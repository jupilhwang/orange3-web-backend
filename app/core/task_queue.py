"""
DB 독립적인 Task Queue 서비스.

PostgreSQL, MySQL, SQLite, Oracle 모두 지원합니다.
asyncio 네이티브로 FastAPI와 완벽 호환됩니다.

Features:
- 우선순위 기반 작업 처리
- 자동 재시도
- 작업 상태 추적
- 멀티테넌트 지원
- DB 레벨 락 (FOR UPDATE SKIP LOCKED)

Usage:
    from app.core.task_queue import task, enqueue_task, get_task_status
    
    # 태스크 정의
    @task(name="ml.train", max_retries=3)
    async def train_model(data_path: str, params: dict):
        # ... 학습 로직
        return {"model_id": "..."}
    
    # 태스크 호출
    task_id = await train_model.delay(data_path="...", params={}, tenant_id="default")
    
    # 상태 조회
    status = await get_task_status(task_id)
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TypeVar
from functools import wraps

from sqlalchemy import select, update, and_, delete
from sqlalchemy.ext.asyncio import AsyncSession

from .db_models import TaskQueueDB, TaskStatus, TaskPriority
from .database import async_session_maker

logger = logging.getLogger(__name__)

# Type variable for generic return type
T = TypeVar('T')

# 등록된 태스크 핸들러
_task_handlers: Dict[str, Callable] = {}


def task(
    name: str = None, 
    max_retries: int = 3, 
    priority: int = TaskPriority.NORMAL
):
    """
    태스크 데코레이터.
    
    Args:
        name: 태스크 이름 (기본값: module.function_name)
        max_retries: 최대 재시도 횟수
        priority: 우선순위 (TaskPriority.LOW/NORMAL/HIGH/CRITICAL)
        
    Example:
        @task(name="ml.train_kmeans", max_retries=3, priority=TaskPriority.HIGH)
        async def train_kmeans(data_path: str, k: int):
            # ... 학습 로직
            return {"clusters": k}
    """
    def decorator(func: Callable):
        task_name = name or f"{func.__module__}.{func.__name__}"
        _task_handlers[task_name] = func
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        async def delay(*args, tenant_id: str = "default", **kwargs) -> str:
            """태스크를 큐에 추가하고 task_id 반환."""
            return await enqueue_task(
                task_name=task_name,
                args=args,
                kwargs=kwargs,
                tenant_id=tenant_id,
                priority=priority,
                max_retries=max_retries
            )
        
        wrapper.delay = delay
        wrapper.task_name = task_name
        return wrapper
    
    return decorator


async def enqueue_task(
    task_name: str,
    args: tuple = (),
    kwargs: dict = None,
    tenant_id: str = "default",
    priority: int = TaskPriority.NORMAL,
    max_retries: int = 3
) -> str:
    """
    태스크를 큐에 추가.
    
    Args:
        task_name: 태스크 이름 (등록된 핸들러 이름)
        args: 위치 인자
        kwargs: 키워드 인자
        tenant_id: 테넌트 ID
        priority: 우선순위
        max_retries: 최대 재시도 횟수
        
    Returns:
        task_id: 생성된 태스크 ID
    """
    task_id = str(uuid.uuid4())
    
    async with async_session_maker() as session:
        task_record = TaskQueueDB(
            id=task_id,
            tenant_id=tenant_id,
            task_name=task_name,
            task_args=json.dumps(list(args)) if args else None,
            task_kwargs=json.dumps(kwargs) if kwargs else None,
            status=TaskStatus.PENDING,
            priority=priority,
            max_retries=max_retries
        )
        session.add(task_record)
        await session.commit()
    
    logger.info(f"Task enqueued: {task_id} ({task_name}) [priority={priority}]")
    return task_id


async def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """
    태스크 상태 조회.
    
    Args:
        task_id: 태스크 ID
        
    Returns:
        태스크 상태 딕셔너리 또는 None
    """
    async with async_session_maker() as session:
        result = await session.execute(
            select(TaskQueueDB).where(TaskQueueDB.id == task_id)
        )
        task = result.scalar_one_or_none()
        
        if not task:
            return None
        
        return {
            "id": task.id,
            "tenant_id": task.tenant_id,
            "task_name": task.task_name,
            "status": task.status,
            "priority": task.priority,
            "progress": task.progress,
            "progress_message": task.progress_message,
            "result": json.loads(task.result) if task.result else None,
            "error": task.error,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries,
            "worker_id": task.worker_id,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        }


async def update_task_progress(
    task_id: str, 
    progress: float, 
    message: str = None
) -> bool:
    """
    태스크 진행률 업데이트.
    
    Args:
        task_id: 태스크 ID
        progress: 진행률 (0.0 ~ 100.0)
        message: 진행 상태 메시지 (선택)
        
    Returns:
        성공 여부
    """
    async with async_session_maker() as session:
        values = {"progress": progress}
        if message is not None:
            values["progress_message"] = message
        
        result = await session.execute(
            update(TaskQueueDB)
            .where(TaskQueueDB.id == task_id)
            .values(**values)
        )
        await session.commit()
        
        if result.rowcount > 0:
            logger.debug(f"Task progress updated: {task_id} -> {progress}%")
            # 진행률 변경 시 WebSocket으로 알림
            await _notify_progress(task_id, progress, message)
            return True
        return False


# WebSocket 알림 콜백 (외부에서 설정)
_progress_callback: Optional[Callable] = None
_completion_callback: Optional[Callable] = None


def set_progress_callback(callback: Callable):
    """진행률 알림 콜백 설정."""
    global _progress_callback
    _progress_callback = callback


def set_completion_callback(callback: Callable):
    """완료 알림 콜백 설정."""
    global _completion_callback
    _completion_callback = callback


async def _notify_progress(task_id: str, progress: float, message: str = None):
    """진행률 변경 알림."""
    if _progress_callback:
        try:
            await _progress_callback(task_id, progress, message)
        except Exception as e:
            logger.warning(f"Progress callback error: {e}")


async def _notify_completion(
    task_id: str, 
    status: str, 
    result: Any = None, 
    error: str = None
):
    """완료/실패 알림."""
    if _completion_callback:
        try:
            await _completion_callback(task_id, status, result, error)
        except Exception as e:
            logger.warning(f"Completion callback error: {e}")


async def list_tasks(
    tenant_id: str = None,
    status: str = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    태스크 목록 조회.
    
    Args:
        tenant_id: 테넌트 ID (None이면 전체)
        status: 상태 필터 (None이면 전체)
        limit: 최대 개수
        offset: 시작 위치
        
    Returns:
        태스크 목록
    """
    async with async_session_maker() as session:
        query = select(TaskQueueDB).order_by(
            TaskQueueDB.created_at.desc()
        ).limit(limit).offset(offset)
        
        if tenant_id:
            query = query.where(TaskQueueDB.tenant_id == tenant_id)
        if status:
            query = query.where(TaskQueueDB.status == status)
        
        result = await session.execute(query)
        tasks = result.scalars().all()
        
        return [
            {
                "id": t.id,
                "tenant_id": t.tenant_id,
                "task_name": t.task_name,
                "status": t.status,
                "priority": t.priority,
                "retry_count": t.retry_count,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
            }
            for t in tasks
        ]


async def cancel_task(task_id: str) -> bool:
    """
    태스크 취소.
    
    PENDING 상태인 태스크만 취소 가능합니다.
    
    Args:
        task_id: 태스크 ID
        
    Returns:
        성공 여부
    """
    async with async_session_maker() as session:
        result = await session.execute(
            update(TaskQueueDB)
            .where(
                and_(
                    TaskQueueDB.id == task_id,
                    TaskQueueDB.status == TaskStatus.PENDING
                )
            )
            .values(
                status=TaskStatus.CANCELLED,
                completed_at=datetime.utcnow()
            )
        )
        await session.commit()
        
        if result.rowcount > 0:
            logger.info(f"Task cancelled: {task_id}")
            return True
        return False


async def fetch_next_task(worker_id: str, use_skip_locked: bool = True) -> Optional[TaskQueueDB]:
    """
    다음 실행할 태스크 가져오기 (원자적 연산).
    
    우선순위 높은 것 먼저, 같으면 먼저 생성된 것.
    
    Args:
        worker_id: 워커 ID
        use_skip_locked: FOR UPDATE SKIP LOCKED 사용 여부
                        (SQLite는 지원하지 않으므로 False로 설정)
    
    Returns:
        TaskQueueDB 또는 None
    """
    async with async_session_maker() as session:
        query = (
            select(TaskQueueDB)
            .where(TaskQueueDB.status == TaskStatus.PENDING)
            .order_by(
                TaskQueueDB.priority.desc(),
                TaskQueueDB.created_at.asc()
            )
            .limit(1)
        )
        
        # PostgreSQL/MySQL은 FOR UPDATE SKIP LOCKED 지원
        # SQLite는 지원하지 않으므로 낙관적 락 사용
        if use_skip_locked:
            query = query.with_for_update(skip_locked=True)
        
        result = await session.execute(query)
        task = result.scalar_one_or_none()
        
        if task:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            task.worker_id = worker_id
            await session.commit()
            await session.refresh(task)
            logger.debug(f"Task fetched: {task.id} by {worker_id}")
        
        return task


async def complete_task(task_id: str, result: Any = None):
    """
    태스크 완료 처리.
    
    Args:
        task_id: 태스크 ID
        result: 결과 데이터 (JSON 직렬화 가능해야 함)
    """
    async with async_session_maker() as session:
        await session.execute(
            update(TaskQueueDB)
            .where(TaskQueueDB.id == task_id)
            .values(
                status=TaskStatus.COMPLETED,
                progress=100.0,
                progress_message="완료",
                result=json.dumps(result) if result is not None else None,
                completed_at=datetime.utcnow()
            )
        )
        await session.commit()
    logger.info(f"Task completed: {task_id}")
    
    # 완료 알림 전송
    await _notify_completion(task_id, TaskStatus.COMPLETED, result)


async def fail_task(task_id: str, error: str, retry: bool = True):
    """
    태스크 실패 처리.
    
    Args:
        task_id: 태스크 ID
        error: 에러 메시지
        retry: 재시도 여부
    """
    final_status = None
    async with async_session_maker() as session:
        result = await session.execute(
            select(TaskQueueDB).where(TaskQueueDB.id == task_id)
        )
        task = result.scalar_one_or_none()
        
        if not task:
            return
        
        task.retry_count += 1
        
        if retry and task.retry_count < task.max_retries:
            # 재시도: PENDING으로 되돌림
            task.status = TaskStatus.PENDING
            task.started_at = None
            task.worker_id = None
            task.progress = 0.0
            task.progress_message = f"재시도 {task.retry_count}/{task.max_retries}"
            logger.warning(
                f"Task retry ({task.retry_count}/{task.max_retries}): {task_id}"
            )
        else:
            # 최종 실패
            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = datetime.utcnow()
            task.progress_message = "실패"
            final_status = TaskStatus.FAILED
            logger.error(f"Task failed: {task_id} - {error}")
        
        await session.commit()
    
    # 최종 실패 시 알림
    if final_status:
        await _notify_completion(task_id, final_status, None, error)


async def cleanup_stale_tasks(timeout_minutes: int = 30) -> int:
    """
    오래된 RUNNING 태스크 정리 (워커가 죽은 경우).
    
    Args:
        timeout_minutes: 타임아웃 시간 (분)
        
    Returns:
        정리된 태스크 수
    """
    cutoff = datetime.utcnow() - timedelta(minutes=timeout_minutes)
    
    async with async_session_maker() as session:
        result = await session.execute(
            update(TaskQueueDB)
            .where(
                and_(
                    TaskQueueDB.status == TaskStatus.RUNNING,
                    TaskQueueDB.started_at < cutoff
                )
            )
            .values(
                status=TaskStatus.PENDING,
                started_at=None,
                worker_id=None
            )
        )
        await session.commit()
        
        if result.rowcount > 0:
            logger.warning(f"Reset {result.rowcount} stale tasks")
        
        return result.rowcount


async def cleanup_old_tasks(days: int = 7) -> int:
    """
    오래된 완료/실패 태스크 삭제.
    
    Args:
        days: 보관 기간 (일)
        
    Returns:
        삭제된 태스크 수
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    async with async_session_maker() as session:
        result = await session.execute(
            delete(TaskQueueDB)
            .where(
                and_(
                    TaskQueueDB.status.in_([
                        TaskStatus.COMPLETED, 
                        TaskStatus.FAILED, 
                        TaskStatus.CANCELLED
                    ]),
                    TaskQueueDB.completed_at < cutoff
                )
            )
        )
        await session.commit()
        
        if result.rowcount > 0:
            logger.info(f"Cleaned up {result.rowcount} old tasks")
        
        return result.rowcount


async def get_queue_stats() -> Dict[str, Any]:
    """
    큐 통계 조회.
    
    Returns:
        상태별 태스크 수 및 기타 통계
    """
    from sqlalchemy import func
    
    async with async_session_maker() as session:
        # 상태별 카운트
        result = await session.execute(
            select(
                TaskQueueDB.status,
                func.count(TaskQueueDB.id).label("count")
            ).group_by(TaskQueueDB.status)
        )
        status_counts = {row.status: row.count for row in result}
        
        return {
            "pending": status_counts.get(TaskStatus.PENDING, 0),
            "running": status_counts.get(TaskStatus.RUNNING, 0),
            "completed": status_counts.get(TaskStatus.COMPLETED, 0),
            "failed": status_counts.get(TaskStatus.FAILED, 0),
            "cancelled": status_counts.get(TaskStatus.CANCELLED, 0),
            "total": sum(status_counts.values()),
        }


# =============================================================================
# Task Worker
# =============================================================================

class TaskWorker:
    """
    비동기 태스크 워커.
    
    백그라운드에서 실행되며, 큐에서 태스크를 가져와 실행합니다.
    
    Example:
        worker = TaskWorker(poll_interval=1.0)
        await worker.start()  # 백그라운드에서 실행
        # ...
        await worker.stop()
    """
    
    def __init__(
        self, 
        worker_id: str = None, 
        poll_interval: float = 1.0,
        use_skip_locked: bool = None
    ):
        """
        Args:
            worker_id: 워커 식별자 (기본값: 랜덤 생성)
            poll_interval: 폴링 간격 (초)
            use_skip_locked: FOR UPDATE SKIP LOCKED 사용 여부
                            (None이면 DB 타입에 따라 자동 결정)
        """
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.poll_interval = poll_interval
        self._running = False
        self._use_skip_locked = use_skip_locked
    
    async def _detect_db_type(self) -> bool:
        """DB 타입을 감지하여 skip_locked 지원 여부 결정."""
        from .config import get_config
        config = get_config()
        db_url = config.database.url or ""
        
        # SQLite는 FOR UPDATE SKIP LOCKED 미지원
        if "sqlite" in db_url.lower():
            return False
        # PostgreSQL, MySQL 8+, Oracle은 지원
        return True
    
    async def start(self):
        """워커 시작."""
        self._running = True
        
        # DB 타입 자동 감지
        if self._use_skip_locked is None:
            self._use_skip_locked = await self._detect_db_type()
        
        logger.info(
            f"Worker started: {self.worker_id} "
            f"(skip_locked={self._use_skip_locked})"
        )
        
        while self._running:
            try:
                task = await fetch_next_task(
                    self.worker_id, 
                    use_skip_locked=self._use_skip_locked
                )
                
                if task:
                    await self._execute_task(task)
                else:
                    await asyncio.sleep(self.poll_interval)
                    
            except asyncio.CancelledError:
                logger.info(f"Worker cancelled: {self.worker_id}")
                break
            except Exception as e:
                logger.exception(f"Worker error: {e}")
                await asyncio.sleep(self.poll_interval)
    
    async def stop(self):
        """워커 중지."""
        self._running = False
        logger.info(f"Worker stopping: {self.worker_id}")
    
    async def _execute_task(self, task: TaskQueueDB):
        """태스크 실행."""
        handler = _task_handlers.get(task.task_name)
        
        if not handler:
            await fail_task(
                task.id, 
                f"Unknown task: {task.task_name}", 
                retry=False
            )
            return
        
        try:
            args = json.loads(task.task_args) if task.task_args else []
            kwargs = json.loads(task.task_kwargs) if task.task_kwargs else {}
            
            logger.info(f"Executing task: {task.id} ({task.task_name})")
            
            # 태스크 실행
            if asyncio.iscoroutinefunction(handler):
                result = await handler(*args, **kwargs)
            else:
                # sync 함수는 thread pool에서 실행
                from .data_utils import run_in_thread
                result = await run_in_thread(handler, *args, **kwargs)
            
            await complete_task(task.id, result)
            
        except Exception as e:
            logger.exception(f"Task execution failed: {task.id}")
            await fail_task(task.id, str(e))


# =============================================================================
# Worker Management
# =============================================================================

# 전역 워커 인스턴스
_worker: Optional[TaskWorker] = None
_worker_task: Optional[asyncio.Task] = None


async def start_worker(poll_interval: float = 1.0) -> TaskWorker:
    """
    백그라운드 워커 시작.
    
    Args:
        poll_interval: 폴링 간격 (초)
        
    Returns:
        TaskWorker 인스턴스
    """
    global _worker, _worker_task
    
    if _worker is not None:
        logger.warning("Worker already running")
        return _worker
    
    _worker = TaskWorker(poll_interval=poll_interval)
    _worker_task = asyncio.create_task(_worker.start())
    
    logger.info("Background task worker started")
    return _worker


async def stop_worker():
    """백그라운드 워커 중지."""
    global _worker, _worker_task
    
    if _worker:
        await _worker.stop()
    
    if _worker_task:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
    
    _worker = None
    _worker_task = None
    
    logger.info("Background task worker stopped")


def get_registered_tasks() -> Dict[str, str]:
    """
    등록된 태스크 목록 조회.
    
    Returns:
        {task_name: module.function} 딕셔너리
    """
    return {
        name: f"{handler.__module__}.{handler.__name__}"
        for name, handler in _task_handlers.items()
    }

