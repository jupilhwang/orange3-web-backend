"""
Core infrastructure modules for Orange3 Web Backend.
- config: Configuration management (file + environment variables)
- database: Database connection and session management
- db_models: SQLAlchemy ORM models
- locks: Async lock utilities
- tenant: Multi-tenant management
- file_storage: File storage abstraction (filesystem/database)
- task_queue: DB-based async task queue
- data_utils: Data loading and session management utilities
- text_mining_utils: Text mining shared utilities
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
    get_engine,
    async_session_maker,
    get_session_maker,
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
from .task_queue import (
    task,
    enqueue_task,
    get_task_status,
    list_tasks,
    cancel_task,
    get_queue_stats,
    get_registered_tasks,
    cleanup_stale_tasks,
    cleanup_old_tasks,
    start_worker,
    stop_worker,
    TaskWorker,
)
from .db_models import (
    TaskQueueDB,
    TaskStatus,
    TaskPriority,
)
from .data_utils import (
    load_data,
    save_data,
    save_data_async,
    resolve_data_path,
    DataSessionManager,
    run_in_process,
    run_in_thread,
    shutdown_executors,
    ORANGE_AVAILABLE as DATA_ORANGE_AVAILABLE,
)
from .text_mining_utils import (
    get_text_cache,
    set_cache_item,
    get_cache_item,
    delete_cache_item,
    ORANGE_TEXT_AVAILABLE,
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
    "get_engine",
    "async_session_maker",
    "get_session_maker",
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
    # task_queue
    "task",
    "enqueue_task",
    "get_task_status",
    "list_tasks",
    "cancel_task",
    "get_queue_stats",
    "get_registered_tasks",
    "cleanup_stale_tasks",
    "cleanup_old_tasks",
    "start_worker",
    "stop_worker",
    "TaskWorker",
    "TaskQueueDB",
    "TaskStatus",
    "TaskPriority",
    # data_utils
    "load_data",
    "save_data",
    "save_data_async",
    "resolve_data_path",
    "DataSessionManager",
    "run_in_process",
    "run_in_thread",
    "shutdown_executors",
    "DATA_ORANGE_AVAILABLE",
    # text_mining_utils
    "get_text_cache",
    "set_cache_item",
    "get_cache_item",
    "delete_cache_item",
    "ORANGE_TEXT_AVAILABLE",
]
