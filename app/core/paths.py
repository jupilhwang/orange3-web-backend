"""
Path Configuration Module.

Centralized path management with environment variable support.
For systemd deployment with DynamicUser, set environment variables to writable directories.

Environment Variables:
    DATABASE_DIR: Directory for SQLite database (default: {app_root})
    UPLOAD_DIR: Base directory for file uploads (default: {app_root}/uploads)
    CORPUS_DIR: Directory for corpus files (default: {UPLOAD_DIR}/corpus)
    DATASETS_CACHE_DIR: Directory for datasets cache (default: {app_root}/datasets_cache)

Example systemd configuration:
    Environment="DATABASE_DIR=/var/lib/orange3-web-backend"
    Environment="UPLOAD_DIR=/var/lib/orange3-web-backend/uploads"
    Environment="CORPUS_DIR=/var/lib/orange3-web-backend/uploads/corpus"
    Environment="DATASETS_CACHE_DIR=/var/lib/orange3-web-backend/datasets_cache"
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Application root directory (backend folder)
APP_ROOT = Path(__file__).parent.parent.parent

# Default paths (relative to app root)
_DEFAULT_DATABASE_DIR = APP_ROOT
_DEFAULT_UPLOAD_DIR = APP_ROOT / "uploads"
_DEFAULT_CORPUS_DIR = _DEFAULT_UPLOAD_DIR / "corpus"
_DEFAULT_DATASETS_CACHE_DIR = APP_ROOT / "datasets_cache"


def _get_path_from_env(env_var: str, default: Path) -> Path:
    """
    Get path from environment variable or use default.
    Creates the directory if it doesn't exist.
    
    Args:
        env_var: Environment variable name
        default: Default path if env var is not set
        
    Returns:
        Path object for the directory
    """
    env_value = os.environ.get(env_var)
    
    if env_value:
        path = Path(env_value)
        logger.info(f"Using {env_var}={path}")
    else:
        path = default
        logger.debug(f"Using default path for {env_var}: {path}")
    
    # Create directory if it doesn't exist
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        logger.warning(f"Cannot create directory {path}: {e}")
    except Exception as e:
        logger.warning(f"Error creating directory {path}: {e}")
    
    return path


def get_database_dir() -> Path:
    """
    Get the database directory.
    
    Uses DATABASE_DIR environment variable if set,
    otherwise defaults to {app_root}
    """
    return _get_path_from_env("DATABASE_DIR", _DEFAULT_DATABASE_DIR)


def get_database_url() -> str:
    """
    Get the SQLite database URL.
    
    Uses DATABASE_URL environment variable if set,
    otherwise constructs from DATABASE_DIR.
    """
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url
    
    db_dir = get_database_dir()
    db_path = db_dir / "orange_web.db"
    return f"sqlite+aiosqlite:///{db_path}"


def get_upload_dir() -> Path:
    """
    Get the base upload directory.
    
    Uses UPLOAD_DIR environment variable if set,
    otherwise defaults to {app_root}/uploads
    """
    return _get_path_from_env("UPLOAD_DIR", _DEFAULT_UPLOAD_DIR)


def get_corpus_dir() -> Path:
    """
    Get the corpus files directory.
    
    Uses CORPUS_DIR environment variable if set,
    otherwise defaults to {UPLOAD_DIR}/corpus
    """
    # If CORPUS_DIR is set, use it directly
    if os.environ.get("CORPUS_DIR"):
        return _get_path_from_env("CORPUS_DIR", _DEFAULT_CORPUS_DIR)
    
    # Otherwise, use {UPLOAD_DIR}/corpus
    return _get_path_from_env("CORPUS_DIR", get_upload_dir() / "corpus")


def get_datasets_cache_dir() -> Path:
    """
    Get the datasets cache directory.
    
    Uses DATASETS_CACHE_DIR environment variable if set,
    otherwise defaults to {app_root}/datasets_cache
    """
    return _get_path_from_env("DATASETS_CACHE_DIR", _DEFAULT_DATASETS_CACHE_DIR)


def get_tenant_upload_dir(tenant_id: str) -> Path:
    """
    Get the upload directory for a specific tenant.
    
    Args:
        tenant_id: Tenant identifier
        
    Returns:
        Path to tenant's upload directory
    """
    tenant_dir = get_upload_dir() / tenant_id
    tenant_dir.mkdir(parents=True, exist_ok=True)
    return tenant_dir


def get_tenant_corpus_dir(tenant_id: str) -> Path:
    """
    Get the corpus directory for a specific tenant.
    
    Args:
        tenant_id: Tenant identifier
        
    Returns:
        Path to tenant's corpus directory
    """
    tenant_dir = get_corpus_dir() / tenant_id
    tenant_dir.mkdir(parents=True, exist_ok=True)
    return tenant_dir


# Initialize directories on module load
def init_directories():
    """Initialize all directories. Called on module import."""
    dirs = [
        ("UPLOAD_DIR", get_upload_dir()),
        ("CORPUS_DIR", get_corpus_dir()),
        ("DATASETS_CACHE_DIR", get_datasets_cache_dir()),
    ]
    
    for name, path in dirs:
        if path.exists():
            logger.info(f"Directory ready: {name}={path}")
        else:
            logger.warning(f"Directory not accessible: {name}={path}")


# Auto-initialize on import
init_directories()

