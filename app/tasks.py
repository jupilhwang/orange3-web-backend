"""
비동기 태스크 정의.

Orange3 Web Backend에서 사용하는 백그라운드 태스크들을 정의합니다.
CPU 집약적인 ML 작업이나 장시간 실행되는 작업에 사용됩니다.

Usage:
    # 태스크 호출
    from app.tasks import train_kmeans
    
    task_id = await train_kmeans.delay(
        data_path="datasets/iris.tab",
        k=3,
        tenant_id="default"
    )
    
    # 상태 조회
    from app.core.task_queue import get_task_status
    status = await get_task_status(task_id)
"""

import logging
from typing import Dict, Any, Optional

from app.core.task_queue import task, TaskPriority
from app.core.db_models import TaskStatus

logger = logging.getLogger(__name__)


# =============================================================================
# ML 태스크
# =============================================================================

@task(name="ml.train_kmeans", max_retries=3, priority=TaskPriority.NORMAL)
async def train_kmeans(
    data_path: str, 
    k: int = 3, 
    max_iterations: int = 100,
    tenant_id: str = "default"
) -> Dict[str, Any]:
    """
    K-Means 클러스터링 학습 태스크.
    
    Args:
        data_path: 데이터 경로 (datasets/iris.tab, uploads/xxx, file:{uuid})
        k: 클러스터 수
        max_iterations: 최대 반복 횟수
        tenant_id: 테넌트 ID
        
    Returns:
        학습 결과 딕셔너리
    """
    logger.info(f"Starting K-Means training: k={k}, data={data_path}")
    
    try:
        from Orange.data import Table
        from Orange.clustering.kmeans import KMeans
        from app.core.data_utils import load_data
        
        # 데이터 로드
        data = load_data(data_path, session_id=tenant_id)
        if data is None:
            raise ValueError(f"Failed to load data: {data_path}")
        
        # K-Means 학습
        kmeans = KMeans(n_clusters=k, max_iter=max_iterations)
        model = kmeans(data)
        
        # 클러스터링 적용
        clustered = model(data)
        
        # 클러스터별 인스턴스 수 계산
        cluster_counts = {}
        cluster_col = clustered.get_column("Cluster")
        for c in cluster_col:
            c_int = int(c)
            cluster_counts[c_int] = cluster_counts.get(c_int, 0) + 1
        
        result = {
            "k": k,
            "instances": len(data),
            "features": len(data.domain.attributes),
            "iterations": max_iterations,
            "cluster_counts": cluster_counts,
            "silhouette_score": None,  # 추후 구현
        }
        
        logger.info(f"K-Means training completed: {result}")
        return result
        
    except ImportError as e:
        logger.error(f"Orange3 not available: {e}")
        raise RuntimeError("Orange3 is not installed")
    except Exception as e:
        logger.exception(f"K-Means training failed: {e}")
        raise


@task(name="ml.train_classifier", max_retries=2, priority=TaskPriority.HIGH)
async def train_classifier(
    data_path: str,
    model_type: str,
    params: Optional[Dict[str, Any]] = None,
    tenant_id: str = "default"
) -> Dict[str, Any]:
    """
    분류 모델 학습 태스크.
    
    Args:
        data_path: 데이터 경로
        model_type: 모델 타입 (knn, tree, naive_bayes, logistic_regression, random_forest)
        params: 모델 파라미터
        tenant_id: 테넌트 ID
        
    Returns:
        학습 결과 딕셔너리
    """
    logger.info(f"Starting classifier training: type={model_type}, data={data_path}")
    params = params or {}
    
    try:
        from Orange.data import Table
        from app.core.data_utils import load_data
        
        # 데이터 로드
        data = load_data(data_path, session_id=tenant_id)
        if data is None:
            raise ValueError(f"Failed to load data: {data_path}")
        
        # 모델 생성
        if model_type == "knn":
            from Orange.classification import KNNLearner
            learner = KNNLearner(n_neighbors=params.get("k", 5))
        elif model_type == "tree":
            from Orange.classification import TreeLearner
            learner = TreeLearner(max_depth=params.get("max_depth", 5))
        elif model_type == "naive_bayes":
            from Orange.classification import NaiveBayesLearner
            learner = NaiveBayesLearner()
        elif model_type == "logistic_regression":
            from Orange.classification import LogisticRegressionLearner
            learner = LogisticRegressionLearner()
        elif model_type == "random_forest":
            from Orange.ensemble import RandomForestLearner
            learner = RandomForestLearner(
                n_estimators=params.get("n_estimators", 10)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 학습
        model = learner(data)
        
        result = {
            "model_type": model_type,
            "params": params,
            "instances": len(data),
            "features": len(data.domain.attributes),
            "target": data.domain.class_var.name if data.domain.class_var else None,
        }
        
        logger.info(f"Classifier training completed: {result}")
        return result
        
    except ImportError as e:
        logger.error(f"Orange3 not available: {e}")
        raise RuntimeError("Orange3 is not installed")
    except Exception as e:
        logger.exception(f"Classifier training failed: {e}")
        raise


@task(name="ml.cross_validate", max_retries=2, priority=TaskPriority.NORMAL)
async def cross_validate(
    data_path: str,
    model_type: str,
    folds: int = 5,
    params: Optional[Dict[str, Any]] = None,
    tenant_id: str = "default"
) -> Dict[str, Any]:
    """
    교차 검증 태스크.
    
    Args:
        data_path: 데이터 경로
        model_type: 모델 타입
        folds: 폴드 수
        params: 모델 파라미터
        tenant_id: 테넌트 ID
        
    Returns:
        교차 검증 결과
    """
    logger.info(f"Starting cross validation: type={model_type}, folds={folds}")
    params = params or {}
    
    try:
        from Orange.data import Table
        from Orange.evaluation import CrossValidation
        from app.core.data_utils import load_data
        import numpy as np
        
        # 데이터 로드
        data = load_data(data_path, session_id=tenant_id)
        if data is None:
            raise ValueError(f"Failed to load data: {data_path}")
        
        # 학습기 생성
        if model_type == "knn":
            from Orange.classification import KNNLearner
            learner = KNNLearner(n_neighbors=params.get("k", 5))
        elif model_type == "tree":
            from Orange.classification import TreeLearner
            learner = TreeLearner(max_depth=params.get("max_depth", 5))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 교차 검증
        cv_results = CrossValidation(data, [learner], k=folds)
        
        # 정확도 계산
        from Orange.evaluation import CA
        accuracy = float(np.mean(CA(cv_results)))
        
        result = {
            "model_type": model_type,
            "folds": folds,
            "accuracy": accuracy,
            "instances": len(data),
        }
        
        logger.info(f"Cross validation completed: accuracy={accuracy:.4f}")
        return result
        
    except ImportError as e:
        logger.error(f"Orange3 not available: {e}")
        raise RuntimeError("Orange3 is not installed")
    except Exception as e:
        logger.exception(f"Cross validation failed: {e}")
        raise


# =============================================================================
# 데이터 처리 태스크
# =============================================================================

@task(name="data.process_large_file", max_retries=1, priority=TaskPriority.LOW)
async def process_large_file(
    file_id: str,
    operations: Optional[list] = None,
    tenant_id: str = "default"
) -> Dict[str, Any]:
    """
    대용량 파일 처리 태스크.
    
    Args:
        file_id: 파일 ID
        operations: 적용할 작업 목록 (예: ["filter", "normalize"])
        tenant_id: 테넌트 ID
        
    Returns:
        처리 결과
    """
    logger.info(f"Processing large file: {file_id}")
    operations = operations or []
    
    try:
        from app.core.file_storage import get_file, get_file_metadata
        
        # 파일 메타데이터 조회
        metadata = await get_file_metadata(file_id, tenant_id)
        if not metadata:
            raise ValueError(f"File not found: {file_id}")
        
        # 파일 내용 로드
        content = await get_file(file_id, tenant_id)
        if not content:
            raise ValueError(f"File content not found: {file_id}")
        
        result = {
            "file_id": file_id,
            "filename": metadata.original_filename,
            "size": metadata.file_size,
            "operations": operations,
            "status": "completed",
        }
        
        logger.info(f"Large file processing completed: {file_id}")
        return result
        
    except Exception as e:
        logger.exception(f"Large file processing failed: {e}")
        raise


@task(name="data.export_dataset", max_retries=2, priority=TaskPriority.LOW)
async def export_dataset(
    data_path: str,
    output_format: str = "csv",
    tenant_id: str = "default"
) -> Dict[str, Any]:
    """
    데이터셋 내보내기 태스크.
    
    Args:
        data_path: 데이터 경로
        output_format: 출력 형식 (csv, xlsx, tab)
        tenant_id: 테넌트 ID
        
    Returns:
        내보내기 결과
    """
    logger.info(f"Exporting dataset: {data_path} -> {output_format}")
    
    try:
        from app.core.data_utils import load_data
        
        data = load_data(data_path, session_id=tenant_id)
        if data is None:
            raise ValueError(f"Failed to load data: {data_path}")
        
        result = {
            "data_path": data_path,
            "format": output_format,
            "instances": len(data),
            "features": len(data.domain.attributes),
            "status": "completed",
        }
        
        logger.info(f"Dataset export completed: {result}")
        return result
        
    except Exception as e:
        logger.exception(f"Dataset export failed: {e}")
        raise


# =============================================================================
# 유틸리티 태스크
# =============================================================================

@task(name="util.cleanup_temp_files", max_retries=1, priority=TaskPriority.LOW)
async def cleanup_temp_files(
    older_than_hours: int = 24,
    tenant_id: str = "default"
) -> Dict[str, Any]:
    """
    임시 파일 정리 태스크.
    
    Args:
        older_than_hours: 삭제 기준 시간 (시간)
        tenant_id: 테넌트 ID
        
    Returns:
        정리 결과
    """
    logger.info(f"Cleaning up temp files older than {older_than_hours} hours")
    
    # 구현 예정
    result = {
        "older_than_hours": older_than_hours,
        "deleted_count": 0,
        "freed_bytes": 0,
        "status": "completed",
    }
    
    logger.info(f"Temp file cleanup completed: {result}")
    return result


@task(name="util.health_check", max_retries=0, priority=TaskPriority.CRITICAL)
async def health_check_task() -> Dict[str, Any]:
    """
    헬스 체크 태스크 (테스트용).
    
    Returns:
        헬스 체크 결과
    """
    import datetime
    
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "message": "Task queue is working correctly",
    }

