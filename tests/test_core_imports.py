"""
Core 모듈 Import 테스트

리팩토링 후 모든 core 모듈이 정상적으로 import되는지 확인합니다.
"""
import pytest
import sys
import os
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCoreImports:
    """Core 모듈 import 테스트"""
    
    def test_import_config(self):
        """config 모듈 import"""
        from app.core.config import (
            get_config,
            get_config_manager,
            get_upload_dir,
            get_corpus_dir,
            get_datasets_cache_dir,
        )
        assert callable(get_config)
        assert callable(get_upload_dir)
    
    def test_import_database(self):
        """database 모듈 import"""
        from app.core.database import (
            Base,
            engine,
            async_session_maker,
            init_db,
            get_db,
            close_db,
        )
        assert Base is not None
    
    def test_import_db_models(self):
        """db_models 모듈 import"""
        from app.core.db_models import (
            TenantDB,
            WorkflowDB,
            NodeDB,
            LinkDB,
            AnnotationDB,
            FileStorageDB,
            TaskQueueDB,
            TaskStatus,
            TaskPriority,
        )
        assert TenantDB is not None
        assert TaskQueueDB is not None
        assert TaskStatus.PENDING == "pending"
    
    def test_import_file_storage(self):
        """file_storage 모듈 import"""
        from app.core.file_storage import (
            StoredFile,
            get_storage,
            save_file,
            get_file,
            get_file_metadata,
            delete_file,
            list_files,
        )
        assert callable(save_file)
        assert callable(get_file)
    
    def test_import_locks(self):
        """locks 모듈 import"""
        from app.core.locks import (
            SimpleLockManager,
            lock_workflow,
            lock_tenant,
        )
        assert SimpleLockManager is not None
    
    def test_import_tenant(self):
        """tenant 모듈 import"""
        from app.core.tenant import (
            TenantManager,
            get_current_tenant,
        )
        assert TenantManager is not None
    
    def test_import_task_queue(self):
        """task_queue 모듈 import"""
        from app.core.task_queue import (
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
        assert callable(task)
        assert callable(enqueue_task)
        assert TaskWorker is not None
    
    def test_import_data_utils(self):
        """data_utils 모듈 import"""
        from app.core.data_utils import (
            load_data,
            save_data,
            save_data_async,
            resolve_data_path,
            DataSessionManager,
            run_in_process,
            run_in_thread,
            shutdown_executors,
        )
        assert callable(load_data)
        assert callable(save_data)
        assert DataSessionManager is not None
    
    def test_import_text_mining_utils(self):
        """text_mining_utils 모듈 import"""
        from app.core.text_mining_utils import (
            get_text_cache,
            set_cache_item,
            get_cache_item,
            delete_cache_item,
            ORANGE_TEXT_AVAILABLE,
        )
        assert callable(get_text_cache)
        assert isinstance(ORANGE_TEXT_AVAILABLE, bool)
    
    def test_import_from_core_init(self):
        """core __init__ 에서 import"""
        from app.core import (
            # config
            get_config,
            get_upload_dir,
            # database
            Base,
            init_db,
            # db_models
            TenantDB,
            WorkflowDB,
            TaskQueueDB,
            TaskStatus,
            TaskPriority,
            # file_storage
            save_file,
            get_file,
            # locks
            lock_workflow,
            # tenant
            TenantManager,
            get_current_tenant,
            # task_queue
            task,
            enqueue_task,
            TaskWorker,
            # data_utils
            load_data,
            DataSessionManager,
            # text_mining_utils
            get_text_cache,
            ORANGE_TEXT_AVAILABLE,
        )
        assert all([
            get_config, get_upload_dir, Base, init_db,
            TenantDB, WorkflowDB, TaskQueueDB, TaskStatus, TaskPriority,
            save_file, get_file, lock_workflow, TenantManager, get_current_tenant,
            task, enqueue_task, TaskWorker, load_data, DataSessionManager,
            get_text_cache,
        ])


class TestWidgetImports:
    """위젯 import 테스트 (리팩토링 후)"""
    
    def test_import_scatter_plot(self):
        """scatter_plot 위젯 import"""
        from app.widgets.scatter_plot import router
        assert router is not None
    
    def test_import_kmeans(self):
        """kmeans 위젯 import"""
        from app.widgets.kmeans import router
        assert router is not None
    
    def test_import_knn(self):
        """knn 위젯 import"""
        from app.widgets.knn import router
        assert router is not None
    
    def test_import_corpus(self):
        """corpus 위젯 import"""
        from app.widgets.corpus import router
        assert router is not None
    
    def test_import_word_cloud(self):
        """word_cloud 위젯 import"""
        from app.widgets.word_cloud import router
        assert router is not None
    
    def test_import_preprocess_text(self):
        """preprocess_text 위젯 import"""
        from app.widgets.preprocess_text import router
        assert router is not None
    
    def test_import_bag_of_words(self):
        """bag_of_words 위젯 import"""
        from app.widgets.bag_of_words import router
        assert router is not None


class TestTasksImport:
    """tasks 모듈 import 테스트"""
    
    def test_import_tasks(self):
        """tasks 모듈 import"""
        from app.tasks import (
            train_kmeans,
            train_classifier,
            cross_validate,
            process_large_file,
            health_check_task,
        )
        assert callable(train_kmeans)
        assert callable(train_classifier)
        assert hasattr(train_kmeans, 'delay')


class TestMainAppImport:
    """메인 앱 import 테스트"""
    
    def test_import_main_app(self):
        """main 앱 import"""
        from app.main import app
        assert app is not None
        assert app.title == "Orange3 Web API"
    
    def test_import_routes(self):
        """routes 모듈 import"""
        from app.routes import (
            workflow_router,
            widget_registry_router,
        )
        assert workflow_router is not None
        assert widget_registry_router is not None


# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

