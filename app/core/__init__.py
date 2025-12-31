"""
Core infrastructure modules for Orange3 Web Backend.
- database: Database connection and session management
- db_models: SQLAlchemy ORM models
- locks: Async lock utilities
- tenant: Multi-tenant management
- paths: Centralized path configuration with environment variable support
"""

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
    workflow_db_to_pydantic,
    generate_uuid,
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
from .paths import (
    get_upload_dir,
    get_corpus_dir,
    get_datasets_cache_dir,
    get_tenant_upload_dir,
    get_tenant_corpus_dir,
)

__all__ = [
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
    "workflow_db_to_pydantic",
    "generate_uuid",
    # locks
    "SimpleLockManager",
    "lock_workflow",
    "lock_tenant",
    "update_with_version_check",
    "workflow_locks",
    # tenant
    "TenantManager",
    "get_current_tenant",
    # paths
    "get_upload_dir",
    "get_corpus_dir",
    "get_datasets_cache_dir",
    "get_tenant_upload_dir",
    "get_tenant_corpus_dir",
]

