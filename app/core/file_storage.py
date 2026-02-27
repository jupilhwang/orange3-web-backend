"""
File Storage Module.

Provides a unified interface for file storage with support for multiple backends:

Supported storage types:
- 'sqlite': SQLite database (default, embedded)
- 'mysql': MySQL/MariaDB database (planned)
- 'postgresql': PostgreSQL database (planned)
- 'oracle': Oracle database (planned)
- 'filesystem' or 'local': Local filesystem

Configuration (priority: config file > env var > default):
    storage.type: 'sqlite' (default), 'mysql', 'postgresql', 'oracle', 'filesystem', 'local'
    storage.max_db_file_size: Maximum file size for DB storage (default: 50MB)
    storage.compression_enabled: Enable zlib compression for DB storage (default: True)
    storage.compression_level: Compression level 1-9 (default: 6)
    storage.compression_min_size: Minimum size to compress (default: 1KB)
"""

import hashlib
import logging
import zlib
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, List, BinaryIO, Union, Tuple
from dataclasses import dataclass

from sqlalchemy import select, delete
from sqlalchemy.orm import load_only
from sqlalchemy.ext.asyncio import AsyncSession

from .config import (
    get_config,
    get_upload_dir,
    get_corpus_dir,
    get_tenant_upload_dir,
    get_tenant_corpus_dir,
)
from .db_models import FileStorageDB, generate_uuid

logger = logging.getLogger(__name__)

# Get storage configuration
_config = get_config()
STORAGE_TYPE = _config.storage.type
MAX_DB_FILE_SIZE = _config.storage.max_db_file_size
COMPRESSION_ENABLED = _config.storage.compression_enabled
COMPRESSION_LEVEL = _config.storage.compression_level
COMPRESSION_MIN_SIZE = _config.storage.compression_min_size


def validate_and_sanitize_path(
    user_input: str, base_dir: Path, description: str = "path"
) -> Path:
    """
    Validate and sanitize a user-provided path to prevent path traversal.

    Args:
        user_input: User-provided path component
        base_dir: Allowed base directory
        description: Description for error messages

    Returns:
        Resolved safe path within base_dir

    Raises:
        ValueError: If path traversal detected
    """
    # Remove any dangerous characters
    sanitized = user_input.replace("../", "").replace("..\\", "").strip()

    # Build full path
    full_path = (base_dir / sanitized).resolve()
    base_resolved = base_dir.resolve()

    # Check if resolved path is within base directory
    try:
        full_path.relative_to(base_resolved)
    except ValueError:
        raise ValueError(f"Path traversal detected in {description}: {user_input}")

    return full_path


@dataclass
class StoredFile:
    """Represents a stored file with metadata."""

    id: str
    filename: str
    original_filename: str
    content_type: str
    file_size: int  # Stored size (compressed if applicable)
    category: str
    tenant_id: str
    checksum: Optional[str] = None
    created_at: Optional[datetime] = None
    # For filesystem storage, this is the file path
    # For database storage, this is None (data is fetched separately)
    file_path: Optional[str] = None
    # Compression info (DB storage only)
    is_compressed: bool = False
    original_size: Optional[int] = None  # Original size before compression


def compress_data(data: bytes) -> Tuple[bytes, bool]:
    """
    Compress data using zlib if enabled and beneficial.

    Returns:
        Tuple of (data, is_compressed) - compressed data if compression is beneficial,
        otherwise original data with is_compressed=False.
    """
    if not COMPRESSION_ENABLED:
        return data, False

    if len(data) < COMPRESSION_MIN_SIZE:
        return data, False

    try:
        compressed = zlib.compress(data, COMPRESSION_LEVEL)
        # Only use compression if it actually reduces size
        if len(compressed) < len(data):
            logger.debug(
                f"Compressed {len(data)} -> {len(compressed)} bytes "
                f"({100 - len(compressed) * 100 // len(data)}% reduction)"
            )
            return compressed, True
        else:
            logger.debug(f"Compression not beneficial for {len(data)} bytes, skipping")
            return data, False
    except Exception as e:
        logger.warning(f"Compression failed: {e}, storing uncompressed")
        return data, False


def decompress_data(data: bytes, is_compressed: bool) -> bytes:
    """
    Decompress data if it was compressed.

    Args:
        data: The stored data (possibly compressed)
        is_compressed: Whether the data is compressed

    Returns:
        Original uncompressed data
    """
    if not is_compressed:
        return data

    try:
        return zlib.decompress(data)
    except Exception as e:
        logger.error(f"Decompression failed: {e}")
        raise ValueError(f"Failed to decompress data: {e}")


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    async def save(
        self,
        tenant_id: str,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        category: str = "upload",
        original_filename: Optional[str] = None,
    ) -> StoredFile:
        """Save a file and return metadata."""
        pass

    @abstractmethod
    async def get(
        self, file_id: str, tenant_id: Optional[str] = None
    ) -> Optional[bytes]:
        """Get file content by ID."""
        pass

    @abstractmethod
    async def get_metadata(
        self, file_id: str, tenant_id: Optional[str] = None
    ) -> Optional[StoredFile]:
        """Get file metadata by ID."""
        pass

    @abstractmethod
    async def delete(self, file_id: str, tenant_id: Optional[str] = None) -> bool:
        """Delete a file by ID. Returns True if deleted."""
        pass

    @abstractmethod
    async def list_files(
        self, tenant_id: str, category: Optional[str] = None
    ) -> List[StoredFile]:
        """List files for a tenant, optionally filtered by category."""
        pass


class FilesystemStorage(StorageBackend):
    """Filesystem-based storage backend."""

    def _get_category_dir(self, tenant_id: str, category: str) -> Path:
        """Get directory path for a category."""
        if category == "corpus":
            return get_tenant_corpus_dir(tenant_id)
        else:
            return get_tenant_upload_dir(tenant_id)

    def _generate_file_id(self) -> str:
        """Generate unique file ID."""
        return generate_uuid()

    async def save(
        self,
        tenant_id: str,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        category: str = "upload",
        original_filename: Optional[str] = None,
    ) -> StoredFile:
        """Save file to filesystem. Uses original filename (tenant separation ensures uniqueness)."""
        if original_filename is None:
            original_filename = filename

        file_id = self._generate_file_id()
        category_dir = self._get_category_dir(tenant_id, category)

        # Validate filename to prevent path traversal
        safe_filename = validate_and_sanitize_path(
            original_filename, category_dir, "filename"
        ).name  # Get just the filename, not full path
        file_path = category_dir / safe_filename

        # Calculate checksum
        checksum = hashlib.sha256(content).hexdigest()

        # Write file (overwrite if exists)
        file_path.write_bytes(content)
        logger.info(f"Saved file to filesystem: {file_path}")

        return StoredFile(
            id=file_id,
            filename=original_filename,
            original_filename=original_filename,
            content_type=content_type,
            file_size=len(content),
            category=category,
            tenant_id=tenant_id,
            checksum=checksum,
            created_at=datetime.utcnow(),
            file_path=str(file_path),
        )

    async def get(
        self, file_id: str, tenant_id: Optional[str] = None
    ) -> Optional[bytes]:
        """Get file content from filesystem. file_id is the filename."""
        search_dirs = []

        if tenant_id:
            search_dirs.extend(
                [
                    get_tenant_upload_dir(tenant_id),
                    get_tenant_corpus_dir(tenant_id),
                ]
            )
        else:
            # Search all tenant directories
            for category_dir in [get_upload_dir(), get_corpus_dir()]:
                if category_dir.exists():
                    for tenant_dir in category_dir.iterdir():
                        if tenant_dir.is_dir():
                            search_dirs.append(tenant_dir)

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            # file_id is the filename
            file_path = search_dir / file_id
            if file_path.is_file():
                return file_path.read_bytes()

        return None

    async def get_metadata(
        self, file_id: str, tenant_id: Optional[str] = None
    ) -> Optional[StoredFile]:
        """Get file metadata from filesystem. file_id is the filename."""
        search_dirs = []

        if tenant_id:
            search_dirs.extend(
                [
                    ("upload", get_tenant_upload_dir(tenant_id)),
                    ("corpus", get_tenant_corpus_dir(tenant_id)),
                ]
            )

        for category, search_dir in search_dirs:
            if not search_dir.exists():
                continue
            # file_id is the filename
            file_path = search_dir / file_id
            if file_path.is_file():
                return StoredFile(
                    id=file_id,
                    filename=file_path.name,
                    original_filename=file_path.name,
                    content_type="application/octet-stream",
                    file_size=file_path.stat().st_size,
                    category=category,
                    tenant_id=tenant_id or "unknown",
                    file_path=str(file_path),
                    created_at=datetime.fromtimestamp(file_path.stat().st_ctime),
                )

        return None

    async def delete(self, file_id: str, tenant_id: Optional[str] = None) -> bool:
        """Delete file from filesystem."""
        metadata = await self.get_metadata(file_id, tenant_id)
        if metadata and metadata.file_path:
            file_path = Path(metadata.file_path)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted file from filesystem: {file_path}")
                return True
        return False

    async def list_files(
        self, tenant_id: str, category: Optional[str] = None
    ) -> List[StoredFile]:
        """List files for a tenant."""
        files = []

        if category:
            dirs = [(category, self._get_category_dir(tenant_id, category))]
        else:
            dirs = [
                ("upload", get_tenant_upload_dir(tenant_id)),
                ("corpus", get_tenant_corpus_dir(tenant_id)),
            ]

        for cat, dir_path in dirs:
            if not dir_path.exists():
                continue
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    # Use filename as ID (original filename is kept as-is)
                    file_id = file_path.name

                    files.append(
                        StoredFile(
                            id=file_id,
                            filename=file_path.name,
                            original_filename=file_path.name,
                            content_type="application/octet-stream",
                            file_size=file_path.stat().st_size,
                            category=cat,
                            tenant_id=tenant_id,
                            file_path=str(file_path),
                            created_at=datetime.fromtimestamp(
                                file_path.stat().st_ctime
                            ),
                        )
                    )

        return files


class DatabaseStorage(StorageBackend):
    """Database-based storage backend using SQLAlchemy."""

    def __init__(self, session_maker):
        """Initialize with async session maker."""
        self._session_maker = session_maker

    async def save(
        self,
        tenant_id: str,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        category: str = "upload",
        original_filename: Optional[str] = None,
    ) -> StoredFile:
        """Save file to database with optional compression."""
        original_size = len(content)

        if original_size > MAX_DB_FILE_SIZE:
            raise ValueError(
                f"File size ({original_size} bytes) exceeds maximum "
                f"({MAX_DB_FILE_SIZE} bytes) for database storage"
            )

        if original_filename is None:
            original_filename = filename

        # Generate UUID for file ID
        file_id = generate_uuid()
        # Checksum of original (uncompressed) content
        checksum = hashlib.sha256(content).hexdigest()

        # Compress if enabled and beneficial
        stored_data, is_compressed = compress_data(content)
        stored_size = len(stored_data)

        async with self._session_maker() as session:
            # Check if file already exists and delete it (overwrite)
            from sqlalchemy import delete

            await session.execute(
                delete(FileStorageDB).where(
                    FileStorageDB.tenant_id == tenant_id,
                    FileStorageDB.filename == original_filename,
                    FileStorageDB.category == category,
                )
            )

            file_entry = FileStorageDB(
                id=file_id,
                tenant_id=tenant_id,
                filename=original_filename,
                original_filename=original_filename,
                content_type=content_type,
                file_size=stored_size,  # Stored (possibly compressed) size
                original_size=original_size,  # Original size before compression
                checksum=checksum,
                is_compressed=is_compressed,
                category=category,
                file_data=stored_data,  # Possibly compressed data
            )
            session.add(file_entry)
            await session.commit()

            compression_info = ""
            if is_compressed:
                ratio = (
                    100 - (stored_size * 100 // original_size)
                    if original_size > 0
                    else 0
                )
                compression_info = f", compressed {ratio}%"

            logger.info(
                f"Saved file to database: {original_filename} (id={file_id}, "
                f"{original_size} -> {stored_size} bytes{compression_info})"
            )

            return StoredFile(
                id=file_id,
                filename=original_filename,
                original_filename=original_filename,
                content_type=content_type,
                file_size=stored_size,
                category=category,
                tenant_id=tenant_id,
                checksum=checksum,
                created_at=file_entry.created_at,
                is_compressed=is_compressed,
                original_size=original_size,
            )

    async def get(
        self, file_id: str, tenant_id: Optional[str] = None
    ) -> Optional[bytes]:
        """Get file content from database, decompressing if necessary."""
        async with self._session_maker() as session:
            query = select(FileStorageDB).where(FileStorageDB.id == file_id)
            if tenant_id:
                query = query.where(FileStorageDB.tenant_id == tenant_id)

            result = await session.execute(query)
            file_entry = result.scalar_one_or_none()

            if file_entry:
                # Decompress if data was stored compressed
                return decompress_data(file_entry.file_data, file_entry.is_compressed)
            return None

    async def get_metadata(
        self, file_id: str, tenant_id: Optional[str] = None
    ) -> Optional[StoredFile]:
        """Get file metadata from database.

        Uses load_only() to exclude the file_data BLOB column — avoids loading
        potentially large binary payloads when only metadata is needed.
        """
        async with self._session_maker() as session:
            query = (
                select(FileStorageDB)
                .where(FileStorageDB.id == file_id)
                .options(
                    load_only(
                        FileStorageDB.id,
                        FileStorageDB.tenant_id,
                        FileStorageDB.filename,
                        FileStorageDB.original_filename,
                        FileStorageDB.content_type,
                        FileStorageDB.file_size,
                        FileStorageDB.original_size,
                        FileStorageDB.checksum,
                        FileStorageDB.is_compressed,
                        FileStorageDB.category,
                        FileStorageDB.created_at,
                    )
                )
            )
            if tenant_id:
                query = query.where(FileStorageDB.tenant_id == tenant_id)

            result = await session.execute(query)
            file_entry = result.scalar_one_or_none()

            if file_entry:
                return StoredFile(
                    id=file_entry.id,
                    filename=file_entry.filename,
                    original_filename=file_entry.original_filename,
                    content_type=file_entry.content_type,
                    file_size=file_entry.file_size,
                    category=file_entry.category,
                    tenant_id=file_entry.tenant_id,
                    checksum=file_entry.checksum,
                    created_at=file_entry.created_at,
                    is_compressed=file_entry.is_compressed,
                    original_size=file_entry.original_size,
                )
            return None

    async def delete(self, file_id: str, tenant_id: Optional[str] = None) -> bool:
        """Delete file from database."""
        async with self._session_maker() as session:
            query = delete(FileStorageDB).where(FileStorageDB.id == file_id)
            if tenant_id:
                query = query.where(FileStorageDB.tenant_id == tenant_id)

            result = await session.execute(query)
            await session.commit()

            deleted = result.rowcount > 0
            if deleted:
                logger.info(f"Deleted file from database: {file_id}")
            return deleted

    async def list_files(
        self, tenant_id: str, category: Optional[str] = None
    ) -> List[StoredFile]:
        """List files for a tenant from database.

        Excludes file_data BLOB from the SELECT to avoid loading large binary
        payloads when only the file listing metadata is needed.
        """
        async with self._session_maker() as session:
            query = (
                select(FileStorageDB)
                .where(FileStorageDB.tenant_id == tenant_id)
                .options(
                    load_only(
                        FileStorageDB.id,
                        FileStorageDB.tenant_id,
                        FileStorageDB.filename,
                        FileStorageDB.original_filename,
                        FileStorageDB.content_type,
                        FileStorageDB.file_size,
                        FileStorageDB.original_size,
                        FileStorageDB.checksum,
                        FileStorageDB.is_compressed,
                        FileStorageDB.category,
                        FileStorageDB.created_at,
                    )
                )
            )
            if category:
                query = query.where(FileStorageDB.category == category)
            query = query.order_by(FileStorageDB.created_at.desc())

            result = await session.execute(query)
            files = []

            for file_entry in result.scalars():
                files.append(
                    StoredFile(
                        id=file_entry.id,
                        filename=file_entry.filename,
                        original_filename=file_entry.original_filename,
                        content_type=file_entry.content_type,
                        file_size=file_entry.file_size,
                        category=file_entry.category,
                        tenant_id=file_entry.tenant_id,
                        checksum=file_entry.checksum,
                        created_at=file_entry.created_at,
                        is_compressed=file_entry.is_compressed,
                        original_size=file_entry.original_size,
                    )
                )

            return files


# Global storage instance (lazy initialization)
_storage: Optional[StorageBackend] = None


def get_storage() -> StorageBackend:
    """
    Get the configured storage backend.

    Supported storage types:
    - 'sqlite': SQLite database (default)
    - 'mysql': MySQL/MariaDB database (uses same DatabaseStorage)
    - 'postgresql': PostgreSQL database (uses same DatabaseStorage)
    - 'oracle': Oracle database (uses same DatabaseStorage)
    - 'filesystem' or 'local': Local filesystem
    """
    global _storage

    if _storage is None:
        # Database storage types
        db_storage_types = {"sqlite", "mysql", "postgresql", "oracle", "database"}
        # Filesystem storage types
        fs_storage_types = {"filesystem", "local"}

        if STORAGE_TYPE in db_storage_types:
            from .database import async_session_maker

            _storage = DatabaseStorage(async_session_maker)
            logger.info(f"Using database storage backend (type: {STORAGE_TYPE})")
        elif STORAGE_TYPE in fs_storage_types:
            _storage = FilesystemStorage()
            logger.info("Using filesystem storage backend")
        else:
            # Default to SQLite if unknown type
            from .database import async_session_maker

            _storage = DatabaseStorage(async_session_maker)
            logger.warning(
                f"Unknown storage type '{STORAGE_TYPE}', defaulting to sqlite"
            )

    return _storage


def reset_storage():
    """Reset storage instance (for testing)."""
    global _storage
    _storage = None


# Convenience functions
async def save_file(
    tenant_id: str,
    filename: str,
    content: bytes,
    content_type: str = "application/octet-stream",
    category: str = "upload",
    original_filename: Optional[str] = None,
) -> StoredFile:
    """Save a file using the configured storage backend."""
    storage = get_storage()
    return await storage.save(
        tenant_id=tenant_id,
        filename=filename,
        content=content,
        content_type=content_type,
        category=category,
        original_filename=original_filename,
    )


async def get_file(file_id: str, tenant_id: Optional[str] = None) -> Optional[bytes]:
    """Get file content using the configured storage backend."""
    storage = get_storage()
    return await storage.get(file_id, tenant_id)


async def get_file_metadata(
    file_id: str, tenant_id: Optional[str] = None
) -> Optional[StoredFile]:
    """Get file metadata using the configured storage backend."""
    storage = get_storage()
    return await storage.get_metadata(file_id, tenant_id)


async def delete_file(file_id: str, tenant_id: Optional[str] = None) -> bool:
    """Delete a file using the configured storage backend."""
    storage = get_storage()
    return await storage.delete(file_id, tenant_id)


async def list_files(
    tenant_id: str, category: Optional[str] = None
) -> List[StoredFile]:
    """List files using the configured storage backend."""
    storage = get_storage()
    return await storage.list_files(tenant_id, category)
