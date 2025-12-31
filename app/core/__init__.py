"""
Core infrastructure modules for Orange3 Web Backend.
- config: Configuration management (file + environment variables)
- database: Database connection and session management
- db_models: SQLAlchemy ORM models
- locks: Async lock utilities
- tenant: Multi-tenant management
- file_storage: File storage abstraction (filesystem/database)
"""

from .config import (
    get_config,
    get_config_manager,
    get_setting,
    get_database_dir,
    get_database_url,
    get_upload_dir,
    get_corpus_dir,
    get_datasets_cache_dir,
    get_tenant_upload_dir,
    get_tenant_corpus_dir,
    init_directories,
    APP_ROOT,
)
from .database import (
    Base,
    engine,
    async_session_maker,
    init_db,
    get_db,
    close_db,
    DATABASE_URL,
)
from .db_models import (
    TenantDB,
    WorkflowDB,
    NodeDB,
    LinkDB,
    AnnotationDB,
    FileStorageDB,
    workflow_db_to_pydantic,
    generate_uuid,
)
from .file_storage import (
    StoredFile,
    get_storage,
    save_file,
    get_file,
    get_file_metadata,
    delete_file,
    list_files,
    STORAGE_TYPE,
)
from .locks import (
    SimpleLockManager,
    lock_workflow,
    lock_tenant,
    update_with_version_check,
    workflow_locks,
)
from .tenant import (
    TenantManager,
    get_current_tenant,
)

__all__ = [
    # config
    "get_config",
    "get_config_manager",
    "get_setting",
    "get_database_dir",
    "get_database_url",
    "get_upload_dir",
    "get_corpus_dir",
    "get_datasets_cache_dir",
    "get_tenant_upload_dir",
    "get_tenant_corpus_dir",
    "init_directories",
    "APP_ROOT",
    # database
    "Base",
    "engine",
    "async_session_maker",
    "init_db",
    "get_db",
    "close_db",
    "DATABASE_URL",
    # db_models
    "TenantDB",
    "WorkflowDB",
    "NodeDB",
    "LinkDB",
    "AnnotationDB",
    "FileStorageDB",
    "workflow_db_to_pydantic",
    "generate_uuid",
    # file_storage
    "StoredFile",
    "get_storage",
    "save_file",
    "get_file",
    "get_file_metadata",
    "delete_file",
    "list_files",
    "STORAGE_TYPE",
    # locks
    "SimpleLockManager",
    "lock_workflow",
    "lock_tenant",
    "update_with_version_check",
    "workflow_locks",
    # tenant
    "TenantManager",
    "get_current_tenant",
]

