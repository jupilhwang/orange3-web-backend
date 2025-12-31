"""
File Storage Module.

Provides a unified interface for file storage with support for:
- Filesystem storage (default, for single server)
- Database storage (for multi-server without shared filesystem)

Configuration via environment variable:
    STORAGE_TYPE: 'filesystem' (default) or 'database'
"""

import os
import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, List, BinaryIO, Union
from dataclasses import dataclass

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from .paths import get_upload_dir, get_corpus_dir, get_tenant_upload_dir, get_tenant_corpus_dir
from .db_models import FileStorageDB, generate_uuid

logger = logging.getLogger(__name__)

# Storage type configuration
STORAGE_TYPE = os.environ.get('STORAGE_TYPE', 'filesystem')  # 'filesystem' or 'database'

# Maximum file size for database storage (50MB)
MAX_DB_FILE_SIZE = int(os.environ.get('MAX_DB_FILE_SIZE', 50 * 1024 * 1024))


@dataclass
class StoredFile:
    """Represents a stored file with metadata."""
    id: str
    filename: str
    original_filename: str
    content_type: str
    file_size: int
    category: str
    tenant_id: str
    checksum: Optional[str] = None
    created_at: Optional[datetime] = None
    # For filesystem storage, this is the file path
    # For database storage, this is None (data is fetched separately)
    file_path: Optional[str] = None


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
    async def get(self, file_id: str, tenant_id: Optional[str] = None) -> Optional[bytes]:
        """Get file content by ID."""
        pass
    
    @abstractmethod
    async def get_metadata(self, file_id: str, tenant_id: Optional[str] = None) -> Optional[StoredFile]:
        """Get file metadata by ID."""
        pass
    
    @abstractmethod
    async def delete(self, file_id: str, tenant_id: Optional[str] = None) -> bool:
        """Delete a file by ID. Returns True if deleted."""
        pass
    
    @abstractmethod
    async def list_files(
        self, 
        tenant_id: str, 
        category: Optional[str] = None
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
    
    def _generate_filename(self, original_filename: str) -> tuple[str, str]:
        """Generate unique filename with UUID prefix."""
        file_id = generate_uuid()
        safe_filename = f"{file_id}_{original_filename}"
        return file_id, safe_filename
    
    async def save(
        self,
        tenant_id: str,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        category: str = "upload",
        original_filename: Optional[str] = None,
    ) -> StoredFile:
        """Save file to filesystem."""
        if original_filename is None:
            original_filename = filename
        
        file_id, safe_filename = self._generate_filename(original_filename)
        category_dir = self._get_category_dir(tenant_id, category)
        file_path = category_dir / safe_filename
        
        # Calculate checksum
        checksum = hashlib.sha256(content).hexdigest()
        
        # Write file
        file_path.write_bytes(content)
        logger.info(f"Saved file to filesystem: {file_path}")
        
        return StoredFile(
            id=file_id,
            filename=safe_filename,
            original_filename=original_filename,
            content_type=content_type,
            file_size=len(content),
            category=category,
            tenant_id=tenant_id,
            checksum=checksum,
            created_at=datetime.utcnow(),
            file_path=str(file_path),
        )
    
    async def get(self, file_id: str, tenant_id: Optional[str] = None) -> Optional[bytes]:
        """Get file content from filesystem."""
        # Search in all possible locations
        search_dirs = []
        
        if tenant_id:
            search_dirs.extend([
                get_tenant_upload_dir(tenant_id),
                get_tenant_corpus_dir(tenant_id),
            ])
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
            for file_path in search_dir.iterdir():
                if file_path.is_file() and file_path.name.startswith(file_id):
                    return file_path.read_bytes()
        
        return None
    
    async def get_metadata(self, file_id: str, tenant_id: Optional[str] = None) -> Optional[StoredFile]:
        """Get file metadata from filesystem."""
        search_dirs = []
        
        if tenant_id:
            search_dirs.extend([
                ("upload", get_tenant_upload_dir(tenant_id)),
                ("corpus", get_tenant_corpus_dir(tenant_id)),
            ])
        
        for category, search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for file_path in search_dir.iterdir():
                if file_path.is_file() and file_path.name.startswith(file_id):
                    # Extract original filename from stored filename
                    parts = file_path.name.split("_", 1)
                    original_filename = parts[1] if len(parts) > 1 else file_path.name
                    
                    return StoredFile(
                        id=file_id,
                        filename=file_path.name,
                        original_filename=original_filename,
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
        self, 
        tenant_id: str, 
        category: Optional[str] = None
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
                    parts = file_path.name.split("_", 1)
                    file_id = parts[0] if len(parts) > 1 else generate_uuid()
                    original_filename = parts[1] if len(parts) > 1 else file_path.name
                    
                    files.append(StoredFile(
                        id=file_id,
                        filename=file_path.name,
                        original_filename=original_filename,
                        content_type="application/octet-stream",
                        file_size=file_path.stat().st_size,
                        category=cat,
                        tenant_id=tenant_id,
                        file_path=str(file_path),
                        created_at=datetime.fromtimestamp(file_path.stat().st_ctime),
                    ))
        
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
        """Save file to database."""
        if len(content) > MAX_DB_FILE_SIZE:
            raise ValueError(
                f"File size ({len(content)} bytes) exceeds maximum "
                f"({MAX_DB_FILE_SIZE} bytes) for database storage"
            )
        
        if original_filename is None:
            original_filename = filename
        
        file_id = generate_uuid()
        safe_filename = f"{file_id}_{original_filename}"
        checksum = hashlib.sha256(content).hexdigest()
        
        async with self._session_maker() as session:
            file_entry = FileStorageDB(
                id=file_id,
                tenant_id=tenant_id,
                filename=safe_filename,
                original_filename=original_filename,
                content_type=content_type,
                file_size=len(content),
                checksum=checksum,
                category=category,
                file_data=content,
            )
            session.add(file_entry)
            await session.commit()
            
            logger.info(f"Saved file to database: {safe_filename} ({len(content)} bytes)")
            
            return StoredFile(
                id=file_id,
                filename=safe_filename,
                original_filename=original_filename,
                content_type=content_type,
                file_size=len(content),
                category=category,
                tenant_id=tenant_id,
                checksum=checksum,
                created_at=file_entry.created_at,
            )
    
    async def get(self, file_id: str, tenant_id: Optional[str] = None) -> Optional[bytes]:
        """Get file content from database."""
        async with self._session_maker() as session:
            query = select(FileStorageDB).where(FileStorageDB.id == file_id)
            if tenant_id:
                query = query.where(FileStorageDB.tenant_id == tenant_id)
            
            result = await session.execute(query)
            file_entry = result.scalar_one_or_none()
            
            if file_entry:
                return file_entry.file_data
            return None
    
    async def get_metadata(self, file_id: str, tenant_id: Optional[str] = None) -> Optional[StoredFile]:
        """Get file metadata from database."""
        async with self._session_maker() as session:
            query = select(FileStorageDB).where(FileStorageDB.id == file_id)
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
        self, 
        tenant_id: str, 
        category: Optional[str] = None
    ) -> List[StoredFile]:
        """List files for a tenant from database."""
        async with self._session_maker() as session:
            query = select(FileStorageDB).where(FileStorageDB.tenant_id == tenant_id)
            if category:
                query = query.where(FileStorageDB.category == category)
            query = query.order_by(FileStorageDB.created_at.desc())
            
            result = await session.execute(query)
            files = []
            
            for file_entry in result.scalars():
                files.append(StoredFile(
                    id=file_entry.id,
                    filename=file_entry.filename,
                    original_filename=file_entry.original_filename,
                    content_type=file_entry.content_type,
                    file_size=file_entry.file_size,
                    category=file_entry.category,
                    tenant_id=file_entry.tenant_id,
                    checksum=file_entry.checksum,
                    created_at=file_entry.created_at,
                ))
            
            return files


# Global storage instance (lazy initialization)
_storage: Optional[StorageBackend] = None


def get_storage() -> StorageBackend:
    """
    Get the configured storage backend.
    
    Uses STORAGE_TYPE environment variable:
    - 'filesystem': FilesystemStorage (default)
    - 'database': DatabaseStorage
    """
    global _storage
    
    if _storage is None:
        if STORAGE_TYPE == 'database':
            from .database import async_session_maker
            _storage = DatabaseStorage(async_session_maker)
            logger.info("Using database storage backend")
        else:
            _storage = FilesystemStorage()
            logger.info("Using filesystem storage backend")
    
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


async def get_file_metadata(file_id: str, tenant_id: Optional[str] = None) -> Optional[StoredFile]:
    """Get file metadata using the configured storage backend."""
    storage = get_storage()
    return await storage.get_metadata(file_id, tenant_id)


async def delete_file(file_id: str, tenant_id: Optional[str] = None) -> bool:
    """Delete a file using the configured storage backend."""
    storage = get_storage()
    return await storage.delete(file_id, tenant_id)


async def list_files(tenant_id: str, category: Optional[str] = None) -> List[StoredFile]:
    """List files using the configured storage backend."""
    storage = get_storage()
    return await storage.list_files(tenant_id, category)

