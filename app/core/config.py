"""
Configuration Management Module.

Supports configuration from multiple sources with priority:
    1. Configuration file (orange3-web-backend.properties)
    2. Environment variables
    3. Default values

Configuration file locations (searched in order):
    1. ./orange3-web-backend.properties (current directory)
    2. /etc/orange3-web/orange3-web-backend.properties
    3. ~/.orange3-web/orange3-web-backend.properties

Properties file format:
    # Comment
    key=value
    key.nested=value

Example orange3-web-backend.properties:
    # Database
    database.url=postgresql+asyncpg://user:pass@localhost:5432/orange3
    
    # Server
    server.host=0.0.0.0
    server.port=8000
    server.workers=4
    
    # Paths
    path.upload=/var/lib/orange3-web/uploads
    path.corpus=/var/lib/orange3-web/corpus
    path.datasets_cache=/var/lib/orange3-web/datasets_cache
    path.database=/var/lib/orange3-web
    
    # Storage
    storage.type=database
    storage.max_db_file_size=52428800
    
    # Logging
    log.level=INFO
    log.database_echo=false
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Type
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PropertiesParser:
    """Parser for .properties files."""
    
    @staticmethod
    def parse(file_path: Path) -> Dict[str, str]:
        """
        Parse a properties file into a dictionary.
        
        Args:
            file_path: Path to the properties file
            
        Returns:
            Dictionary of key-value pairs
        """
        config = {}
        
        if not file_path.exists():
            return config
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#') or line.startswith('!'):
                        continue
                    
                    # Handle key=value or key:value
                    if '=' in line:
                        key, value = line.split('=', 1)
                    elif ':' in line:
                        key, value = line.split(':', 1)
                    else:
                        logger.warning(f"Invalid line {line_num} in {file_path}: {line}")
                        continue
                    
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    
                    config[key] = value
                    
            logger.info(f"Loaded {len(config)} settings from {file_path}")
            
        except Exception as e:
            logger.error(f"Error reading config file {file_path}: {e}")
            
        return config


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: Optional[str] = None
    dir: Optional[str] = None
    echo: bool = False
    
    def get_url(self, app_root: Path) -> str:
        """Get database URL, constructing from dir if url not set."""
        if self.url:
            return self.url
        
        db_dir = Path(self.dir) if self.dir else app_root
        db_path = db_dir / "orange_web.db"
        return f"sqlite+aiosqlite:///{db_path}"


@dataclass
class PathConfig:
    """Path configuration."""
    upload: Optional[str] = None
    corpus: Optional[str] = None
    datasets_cache: Optional[str] = None
    database: Optional[str] = None


@dataclass
class StorageConfig:
    """
    Storage configuration.
    
    Supported storage types:
    - 'sqlite': SQLite database (default, embedded)
    - 'mysql': MySQL/MariaDB database
    - 'postgresql': PostgreSQL database
    - 'oracle': Oracle database
    - 'filesystem' or 'local': Local filesystem
    """
    type: str = "sqlite"  # 'sqlite', 'mysql', 'postgresql', 'oracle', 'filesystem', 'local'
    max_db_file_size: int = 50 * 1024 * 1024  # 50MB


@dataclass
class LogConfig:
    """Logging configuration."""
    level: str = "INFO"
    database_echo: bool = False


@dataclass
class AppConfig:
    """Main application configuration."""
    server: ServerConfig = field(default_factory=ServerConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    path: PathConfig = field(default_factory=PathConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    log: LogConfig = field(default_factory=LogConfig)


class ConfigManager:
    """
    Configuration manager that loads settings from file and environment.
    
    Priority (highest to lowest):
        1. Configuration file
        2. Environment variables
        3. Default values
    """
    
    # Configuration file search paths
    CONFIG_FILE_PATHS = [
        Path("./orange3-web-backend.properties"),
        Path("/etc/orange3-web/orange3-web-backend.properties"),
        Path.home() / ".orange3-web" / "orange3-web-backend.properties",
    ]
    
    # Mapping: config file key -> environment variable
    ENV_MAPPING = {
        # Database
        "database.url": "DATABASE_URL",
        "database.dir": "DATABASE_DIR",
        "database.echo": "DATABASE_ECHO",
        
        # Server
        "server.host": "HOST",
        "server.port": "PORT",
        "server.workers": "WORKERS",
        "server.reload": "RELOAD",
        
        # Paths
        "path.upload": "UPLOAD_DIR",
        "path.corpus": "CORPUS_DIR",
        "path.datasets_cache": "DATASETS_CACHE_DIR",
        "path.database": "DATABASE_DIR",
        
        # Storage
        "storage.type": "STORAGE_TYPE",
        "storage.max_db_file_size": "MAX_DB_FILE_SIZE",
        
        # Logging
        "log.level": "LOG_LEVEL",
        "log.database_echo": "DATABASE_ECHO",
    }
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional explicit path to config file
        """
        self._file_config: Dict[str, str] = {}
        self._config_file_path: Optional[Path] = None
        
        # Load config file
        if config_file:
            self._load_config_file(config_file)
        else:
            self._find_and_load_config_file()
        
        # Build configuration
        self.config = self._build_config()
    
    def _find_and_load_config_file(self):
        """Find and load configuration file from standard locations."""
        for path in self.CONFIG_FILE_PATHS:
            if path.exists():
                self._load_config_file(path)
                break
        else:
            logger.info("No configuration file found, using environment variables and defaults")
    
    def _load_config_file(self, path: Path):
        """Load configuration from file."""
        self._config_file_path = path
        self._file_config = PropertiesParser.parse(path)
        logger.info(f"Configuration file loaded: {path}")
    
    def get(self, key: str, default: Any = None, value_type: Type[T] = str) -> T:
        """
        Get configuration value with priority: file > env > default.
        
        Args:
            key: Configuration key (e.g., 'database.url')
            default: Default value if not found
            value_type: Type to convert value to
            
        Returns:
            Configuration value
        """
        value = None
        
        # 1. Check config file (highest priority)
        if key in self._file_config:
            value = self._file_config[key]
            logger.debug(f"Config '{key}' from file: {value}")
        
        # 2. Check environment variable
        elif key in self.ENV_MAPPING:
            env_var = self.ENV_MAPPING[key]
            env_value = os.environ.get(env_var)
            if env_value is not None:
                value = env_value
                logger.debug(f"Config '{key}' from env {env_var}: {value}")
        
        # 3. Use default
        if value is None:
            return default
        
        # Convert to target type
        return self._convert_type(value, value_type, default)
    
    def _convert_type(self, value: str, value_type: Type[T], default: T) -> T:
        """Convert string value to target type."""
        try:
            if value_type == bool:
                return value.lower() in ('true', 'yes', '1', 'on')
            elif value_type == int:
                return int(value)
            elif value_type == float:
                return float(value)
            else:
                return value
        except (ValueError, AttributeError):
            logger.warning(f"Cannot convert '{value}' to {value_type}, using default")
            return default
    
    def _build_config(self) -> AppConfig:
        """Build complete configuration object."""
        return AppConfig(
            server=ServerConfig(
                host=self.get("server.host", "0.0.0.0"),
                port=self.get("server.port", 8000, int),
                workers=self.get("server.workers", 1, int),
                reload=self.get("server.reload", False, bool),
            ),
            database=DatabaseConfig(
                url=self.get("database.url", None),
                dir=self.get("database.dir", None),
                echo=self.get("database.echo", False, bool),
            ),
            path=PathConfig(
                upload=self.get("path.upload", None),
                corpus=self.get("path.corpus", None),
                datasets_cache=self.get("path.datasets_cache", None),
                database=self.get("path.database", None),
            ),
            storage=StorageConfig(
                type=self.get("storage.type", "sqlite"),
                max_db_file_size=self.get("storage.max_db_file_size", 50 * 1024 * 1024, int),
            ),
            log=LogConfig(
                level=self.get("log.level", "INFO"),
                database_echo=self.get("log.database_echo", False, bool),
            ),
        )
    
    def get_config_file_path(self) -> Optional[Path]:
        """Get the path to the loaded config file."""
        return self._config_file_path
    
    def reload(self):
        """Reload configuration from file."""
        if self._config_file_path:
            self._load_config_file(self._config_file_path)
            self.config = self._build_config()
            logger.info("Configuration reloaded")


# =============================================================================
# Global Configuration Instance
# =============================================================================

_config_manager: Optional[ConfigManager] = None

# Application root directory (backend folder)
APP_ROOT = Path(__file__).parent.parent.parent


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> AppConfig:
    """Get the application configuration."""
    return get_config_manager().config


def get_setting(key: str, default: Any = None, value_type: Type[T] = str) -> T:
    """
    Get a configuration setting.
    
    Shortcut for get_config_manager().get(key, default, value_type)
    """
    return get_config_manager().get(key, default, value_type)


# =============================================================================
# Path Helper Functions
# =============================================================================

def _ensure_directory(path: Path, name: str) -> Path:
    """
    Ensure directory exists and is accessible.
    
    Args:
        path: Path to ensure
        name: Name for logging
        
    Returns:
        Path object for the directory
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"{name} directory: {path}")
    except PermissionError as e:
        logger.warning(f"Cannot create {name} directory {path}: {e}")
    except Exception as e:
        logger.warning(f"Error creating {name} directory {path}: {e}")
    
    return path


def get_database_dir() -> Path:
    """
    Get the database directory.
    
    Priority: config file > DATABASE_DIR env > default (app root)
    """
    config = get_config()
    
    if config.path.database:
        path = Path(config.path.database)
        return _ensure_directory(path, "database")
    
    return _ensure_directory(APP_ROOT, "database")


def get_database_url() -> str:
    """
    Get the database URL.
    
    Priority: config file > DATABASE_URL env > construct from database.dir
    """
    config = get_config()
    
    if config.database.url:
        logger.info("Using database URL from config")
        return config.database.url
    
    db_dir = get_database_dir()
    db_path = db_dir / "orange_web.db"
    url = f"sqlite+aiosqlite:///{db_path}"
    logger.info(f"Using SQLite database: {db_path}")
    return url


def get_upload_dir() -> Path:
    """
    Get the base upload directory.
    
    Priority: config file > UPLOAD_DIR env > default (app_root/uploads)
    """
    config = get_config()
    
    if config.path.upload:
        path = Path(config.path.upload)
        return _ensure_directory(path, "upload")
    
    return _ensure_directory(APP_ROOT / "uploads", "upload")


def get_corpus_dir() -> Path:
    """
    Get the corpus files directory.
    
    Priority: config file > CORPUS_DIR env > {upload_dir}/corpus
    """
    config = get_config()
    
    if config.path.corpus:
        path = Path(config.path.corpus)
        return _ensure_directory(path, "corpus")
    
    return _ensure_directory(get_upload_dir() / "corpus", "corpus")


def get_datasets_cache_dir() -> Path:
    """
    Get the datasets cache directory.
    
    Priority: config file > DATASETS_CACHE_DIR env > default (app_root/datasets_cache)
    """
    config = get_config()
    
    if config.path.datasets_cache:
        path = Path(config.path.datasets_cache)
        return _ensure_directory(path, "datasets_cache")
    
    return _ensure_directory(APP_ROOT / "datasets_cache", "datasets_cache")


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


def init_directories():
    """
    Initialize all directories.
    
    Called during application startup to ensure all required directories exist.
    """
    dirs = [
        ("upload", get_upload_dir()),
        ("corpus", get_corpus_dir()),
        ("datasets_cache", get_datasets_cache_dir()),
    ]
    
    for name, path in dirs:
        if path.exists():
            logger.info(f"Directory ready: {name}={path}")
        else:
            logger.warning(f"Directory not accessible: {name}={path}")


# =============================================================================
# Initialize on Import
# =============================================================================

_config_manager = ConfigManager()

