"""
Task Queue API 엔드포인트.

태스크 생성, 상태 조회, 목록 조회, 취소 등의 API를 제공합니다.

Endpoints:
    POST   /api/v1/tasks              - 태스크 생성
    GET    /api/v1/tasks              - 태스크 목록 조회
    GET    /api/v1/tasks/stats        - 큐 통계 조회
    GET    /api/v1/tasks/registered   - 등록된 태스크 목록
    GET    /api/v1/tasks/{task_id}    - 태스크 상태 조회
    POST   /api/v1/tasks/{task_id}/cancel - 태스크 취소
"""

import logging
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field

from app.core import get_current_tenant
from app.core.models import Tenant
from app.core.task_queue import (
    enqueue_task,
    get_task_status,
    list_tasks,
    cancel_task,
    get_queue_stats,
    get_registered_tasks,
    cleanup_stale_tasks,
    cleanup_old_tasks,
)
from app.core.db_models import TaskPriority, TaskStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["Tasks"])


# =============================================================================
# Request/Response 모델
# =============================================================================

class TaskCreateRequest(BaseModel):
    """태스크 생성 요청."""
    task_name: str = Field(..., description="태스크 이름 (예: ml.train_kmeans)")
    args: List[Any] = Field(default=[], description="위치 인자")
    kwargs: Dict[str, Any] = Field(default={}, description="키워드 인자")
    priority: int = Field(
        default=TaskPriority.NORMAL, 
        description="우선순위 (0=LOW, 5=NORMAL, 10=HIGH, 20=CRITICAL)"
    )
    max_retries: int = Field(default=3, description="최대 재시도 횟수")


class TaskResponse(BaseModel):
    """태스크 응답."""
    id: str
    tenant_id: str
    task_name: str
    status: str
    priority: int
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int
    max_retries: int
    worker_id: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class TaskListResponse(BaseModel):
    """태스크 목록 응답."""
    tasks: List[Dict[str, Any]]
    total: int


class QueueStatsResponse(BaseModel):
    """큐 통계 응답."""
    pending: int
    running: int
    completed: int
    failed: int
    cancelled: int
    total: int


class TaskCreateResponse(BaseModel):
    """태스크 생성 응답."""
    task_id: str
    status: str
    message: str


# =============================================================================
# 엔드포인트
# =============================================================================

@router.post("", response_model=TaskCreateResponse)
async def create_task(
    request: TaskCreateRequest,
    tenant: Tenant = Depends(get_current_tenant)
):
    """
    새 태스크 생성.
    
    태스크를 큐에 추가하고 태스크 ID를 반환합니다.
    """
    try:
        # 등록된 태스크인지 확인
        registered = get_registered_tasks()
        if request.task_name not in registered:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown task: {request.task_name}. "
                       f"Available tasks: {list(registered.keys())}"
            )
        
        task_id = await enqueue_task(
            task_name=request.task_name,
            args=tuple(request.args),
            kwargs=request.kwargs,
            tenant_id=tenant.id,
            priority=request.priority,
            max_retries=request.max_retries
        )
        
        return TaskCreateResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="태스크가 큐에 추가되었습니다."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to create task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=TaskListResponse)
async def get_tasks(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    tenant: Tenant = Depends(get_current_tenant)
):
    """
    태스크 목록 조회.
    
    테넌트의 태스크 목록을 조회합니다.
    status 파라미터로 필터링할 수 있습니다.
    """
    try:
        # 유효한 상태인지 확인
        valid_statuses = [
            TaskStatus.PENDING, 
            TaskStatus.RUNNING, 
            TaskStatus.COMPLETED,
            TaskStatus.FAILED, 
            TaskStatus.CANCELLED
        ]
        if status and status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. "
                       f"Valid statuses: {valid_statuses}"
            )
        
        tasks = await list_tasks(
            tenant_id=tenant.id,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return TaskListResponse(
            tasks=tasks,
            total=len(tasks)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to list tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=QueueStatsResponse)
async def get_stats():
    """
    큐 통계 조회.
    
    상태별 태스크 수를 반환합니다.
    """
    try:
        stats = await get_queue_stats()
        return QueueStatsResponse(**stats)
    except Exception as e:
        logger.exception(f"Failed to get queue stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registered")
async def get_registered():
    """
    등록된 태스크 목록 조회.
    
    사용 가능한 태스크 이름과 핸들러 정보를 반환합니다.
    """
    return {
        "tasks": get_registered_tasks(),
        "priorities": {
            "LOW": TaskPriority.LOW,
            "NORMAL": TaskPriority.NORMAL,
            "HIGH": TaskPriority.HIGH,
            "CRITICAL": TaskPriority.CRITICAL,
        },
        "statuses": {
            "PENDING": TaskStatus.PENDING,
            "RUNNING": TaskStatus.RUNNING,
            "COMPLETED": TaskStatus.COMPLETED,
            "FAILED": TaskStatus.FAILED,
            "CANCELLED": TaskStatus.CANCELLED,
        }
    }


@router.get("/{task_id}")
async def get_task(
    task_id: str,
    tenant: Tenant = Depends(get_current_tenant)
):
    """
    태스크 상태 조회.
    
    특정 태스크의 상세 정보를 반환합니다.
    """
    try:
        task = await get_task_status(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # 다른 테넌트의 태스크인지 확인
        if task["tenant_id"] != tenant.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return task
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{task_id}/cancel")
async def cancel_task_endpoint(
    task_id: str,
    tenant: Tenant = Depends(get_current_tenant)
):
    """
    태스크 취소.
    
    PENDING 상태인 태스크만 취소할 수 있습니다.
    """
    try:
        # 태스크 존재 및 권한 확인
        task = await get_task_status(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if task["tenant_id"] != tenant.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if task["status"] != TaskStatus.PENDING:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot cancel task in {task['status']} status"
            )
        
        success = await cancel_task(task_id)
        
        if success:
            return {"message": "태스크가 취소되었습니다.", "task_id": task_id}
        else:
            raise HTTPException(
                status_code=400, 
                detail="Failed to cancel task"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to cancel task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 관리자 엔드포인트
# =============================================================================

@router.post("/admin/cleanup-stale")
async def admin_cleanup_stale(timeout_minutes: int = 30):
    """
    오래된 RUNNING 태스크 정리 (관리자용).
    
    워커가 죽어서 RUNNING 상태로 남은 태스크를 PENDING으로 리셋합니다.
    """
    try:
        count = await cleanup_stale_tasks(timeout_minutes)
        return {
            "message": f"{count}개의 태스크가 리셋되었습니다.",
            "reset_count": count
        }
    except Exception as e:
        logger.exception(f"Failed to cleanup stale tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/cleanup-old")
async def admin_cleanup_old(days: int = 7):
    """
    오래된 완료/실패 태스크 삭제 (관리자용).
    
    지정된 일수보다 오래된 완료/실패/취소된 태스크를 삭제합니다.
    """
    try:
        count = await cleanup_old_tasks(days)
        return {
            "message": f"{count}개의 태스크가 삭제되었습니다.",
            "deleted_count": count
        }
    except Exception as e:
        logger.exception(f"Failed to cleanup old tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 편의 엔드포인트 (특정 태스크 빠른 호출)
# =============================================================================

class KMeansTaskRequest(BaseModel):
    """K-Means 태스크 요청."""
    data_path: str
    k: int = 3
    max_iterations: int = 100


@router.post("/kmeans", response_model=TaskCreateResponse)
async def create_kmeans_task(
    request: KMeansTaskRequest,
    tenant: Tenant = Depends(get_current_tenant)
):
    """
    K-Means 학습 태스크 생성 (편의 엔드포인트).
    """
    try:
        # tasks.py의 train_kmeans 임포트
        from app.tasks import train_kmeans
        
        task_id = await train_kmeans.delay(
            data_path=request.data_path,
            k=request.k,
            max_iterations=request.max_iterations,
            tenant_id=tenant.id
        )
        
        return TaskCreateResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="K-Means 학습 태스크가 큐에 추가되었습니다."
        )
        
    except Exception as e:
        logger.exception(f"Failed to create K-Means task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ClassifierTaskRequest(BaseModel):
    """분류 모델 태스크 요청."""
    data_path: str
    model_type: str = "knn"
    params: Optional[Dict[str, Any]] = None


@router.post("/classifier", response_model=TaskCreateResponse)
async def create_classifier_task(
    request: ClassifierTaskRequest,
    tenant: Tenant = Depends(get_current_tenant)
):
    """
    분류 모델 학습 태스크 생성 (편의 엔드포인트).
    """
    try:
        from app.tasks import train_classifier
        
        task_id = await train_classifier.delay(
            data_path=request.data_path,
            model_type=request.model_type,
            params=request.params or {},
            tenant_id=tenant.id
        )
        
        return TaskCreateResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message=f"{request.model_type} 학습 태스크가 큐에 추가되었습니다."
        )
        
    except Exception as e:
        logger.exception(f"Failed to create classifier task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/health-check", response_model=TaskCreateResponse)
async def create_health_check_task(
    tenant: Tenant = Depends(get_current_tenant)
):
    """
    헬스 체크 태스크 생성 (테스트용).
    """
    try:
        from app.tasks import health_check_task
        
        task_id = await health_check_task.delay(tenant_id=tenant.id)
        
        return TaskCreateResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="헬스 체크 태스크가 큐에 추가되었습니다."
        )
        
    except Exception as e:
        logger.exception(f"Failed to create health check task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

