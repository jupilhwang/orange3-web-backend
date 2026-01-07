"""
Task Queue 테스트

DB 기반 Task Queue의 기능을 테스트합니다.
"""
import pytest
import asyncio
import sys
import os
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.task_queue import (
    task, 
    enqueue_task, 
    get_task_status, 
    list_tasks,
    cancel_task,
    get_queue_stats,
    get_registered_tasks,
    complete_task,
    fail_task,
    TaskWorker,
)
from app.core.db_models import TaskStatus, TaskPriority


# =============================================================================
# 테스트용 태스크 정의
# =============================================================================

@task(name="test.simple_task", max_retries=1)
async def simple_task(x: int, y: int) -> dict:
    """간단한 덧셈 태스크."""
    return {"result": x + y}


@task(name="test.failing_task", max_retries=2)
async def failing_task() -> dict:
    """항상 실패하는 태스크."""
    raise ValueError("This task always fails")


@task(name="test.slow_task", max_retries=0)
async def slow_task(seconds: float = 0.1) -> dict:
    """느린 태스크."""
    await asyncio.sleep(seconds)
    return {"slept": seconds}


# =============================================================================
# 테스트
# =============================================================================

class TestTaskDecorator:
    """@task 데코레이터 테스트"""
    
    def test_task_registered(self):
        """태스크가 등록되는지 확인"""
        registered = get_registered_tasks()
        assert "test.simple_task" in registered
        assert "test.failing_task" in registered
        assert "test.slow_task" in registered
    
    def test_task_has_delay_method(self):
        """태스크에 delay 메서드가 있는지 확인"""
        assert hasattr(simple_task, 'delay')
        assert callable(simple_task.delay)
    
    def test_task_has_task_name(self):
        """태스크에 task_name 속성이 있는지 확인"""
        assert hasattr(simple_task, 'task_name')
        assert simple_task.task_name == "test.simple_task"


class TestEnqueueTask:
    """태스크 큐 추가 테스트"""
    
    @pytest.mark.asyncio
    async def test_enqueue_returns_task_id(self):
        """enqueue_task가 task_id를 반환하는지 확인"""
        task_id = await enqueue_task(
            task_name="test.simple_task",
            args=(1, 2),
            tenant_id="test_tenant"
        )
        assert task_id is not None
        assert len(task_id) == 36  # UUID 형식
    
    @pytest.mark.asyncio
    async def test_enqueue_with_priority(self):
        """우선순위가 저장되는지 확인"""
        task_id = await enqueue_task(
            task_name="test.simple_task",
            args=(1, 2),
            tenant_id="test_tenant",
            priority=TaskPriority.HIGH
        )
        
        status = await get_task_status(task_id)
        assert status is not None
        assert status["priority"] == TaskPriority.HIGH
    
    @pytest.mark.asyncio
    async def test_enqueue_with_kwargs(self):
        """키워드 인자가 저장되는지 확인"""
        task_id = await enqueue_task(
            task_name="test.slow_task",
            kwargs={"seconds": 0.5},
            tenant_id="test_tenant"
        )
        
        status = await get_task_status(task_id)
        assert status is not None
        assert status["status"] == TaskStatus.PENDING


class TestTaskStatus:
    """태스크 상태 조회 테스트"""
    
    @pytest.mark.asyncio
    async def test_get_status_pending(self):
        """PENDING 상태 확인"""
        task_id = await enqueue_task(
            task_name="test.simple_task",
            args=(1, 2),
            tenant_id="test_tenant"
        )
        
        status = await get_task_status(task_id)
        assert status["status"] == TaskStatus.PENDING
        assert status["retry_count"] == 0
    
    @pytest.mark.asyncio
    async def test_get_status_nonexistent(self):
        """존재하지 않는 태스크 조회"""
        status = await get_task_status("nonexistent-task-id")
        assert status is None


class TestListTasks:
    """태스크 목록 조회 테스트"""
    
    @pytest.mark.asyncio
    async def test_list_tasks_by_tenant(self):
        """테넌트별 태스크 목록 조회"""
        tenant_id = "list_test_tenant"
        
        # 태스크 추가
        await enqueue_task(
            task_name="test.simple_task",
            args=(1, 2),
            tenant_id=tenant_id
        )
        
        tasks = await list_tasks(tenant_id=tenant_id)
        assert len(tasks) >= 1
        assert all(t["tenant_id"] == tenant_id for t in tasks)
    
    @pytest.mark.asyncio
    async def test_list_tasks_by_status(self):
        """상태별 태스크 목록 조회"""
        tasks = await list_tasks(status=TaskStatus.PENDING)
        assert all(t["status"] == TaskStatus.PENDING for t in tasks)


class TestCancelTask:
    """태스크 취소 테스트"""
    
    @pytest.mark.asyncio
    async def test_cancel_pending_task(self):
        """PENDING 태스크 취소"""
        task_id = await enqueue_task(
            task_name="test.slow_task",
            kwargs={"seconds": 10},
            tenant_id="test_tenant"
        )
        
        result = await cancel_task(task_id)
        assert result is True
        
        status = await get_task_status(task_id)
        assert status["status"] == TaskStatus.CANCELLED


class TestCompleteAndFailTask:
    """태스크 완료/실패 처리 테스트"""
    
    @pytest.mark.asyncio
    async def test_complete_task(self):
        """태스크 완료 처리"""
        task_id = await enqueue_task(
            task_name="test.simple_task",
            args=(1, 2),
            tenant_id="test_tenant"
        )
        
        await complete_task(task_id, {"result": 3})
        
        status = await get_task_status(task_id)
        assert status["status"] == TaskStatus.COMPLETED
        assert status["result"] == {"result": 3}
    
    @pytest.mark.asyncio
    async def test_fail_task_with_retry(self):
        """재시도가 있는 태스크 실패"""
        task_id = await enqueue_task(
            task_name="test.failing_task",
            tenant_id="test_tenant",
            max_retries=3
        )
        
        await fail_task(task_id, "Test error", retry=True)
        
        status = await get_task_status(task_id)
        # 재시도가 남았으므로 PENDING으로 돌아감
        assert status["status"] == TaskStatus.PENDING
        assert status["retry_count"] == 1
    
    @pytest.mark.asyncio
    async def test_fail_task_no_retry(self):
        """재시도 없는 태스크 실패"""
        task_id = await enqueue_task(
            task_name="test.failing_task",
            tenant_id="test_tenant",
            max_retries=0
        )
        
        await fail_task(task_id, "Test error", retry=False)
        
        status = await get_task_status(task_id)
        assert status["status"] == TaskStatus.FAILED
        assert status["error"] == "Test error"


class TestQueueStats:
    """큐 통계 테스트"""
    
    @pytest.mark.asyncio
    async def test_get_queue_stats(self):
        """큐 통계 조회"""
        stats = await get_queue_stats()
        
        assert "pending" in stats
        assert "running" in stats
        assert "completed" in stats
        assert "failed" in stats
        assert "cancelled" in stats
        assert "total" in stats


class TestTaskWorker:
    """TaskWorker 테스트"""
    
    def test_worker_creation(self):
        """워커 생성"""
        worker = TaskWorker(poll_interval=0.1)
        
        assert worker.worker_id is not None
        assert worker.poll_interval == 0.1
        assert worker._running is False
    
    @pytest.mark.asyncio
    async def test_worker_executes_task(self):
        """워커가 태스크를 실행하는지 확인"""
        # 태스크 추가
        task_id = await simple_task.delay(10, 20, tenant_id="worker_test")
        
        # 워커 생성 및 실행 (짧은 시간)
        worker = TaskWorker(poll_interval=0.1, use_skip_locked=False)
        
        # 워커를 백그라운드로 실행
        worker_task = asyncio.create_task(worker.start())
        
        # 태스크 완료 대기 (최대 5초)
        for _ in range(50):
            await asyncio.sleep(0.1)
            status = await get_task_status(task_id)
            if status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break
        
        # 워커 중지
        await worker.stop()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        
        # 결과 확인
        status = await get_task_status(task_id)
        assert status["status"] == TaskStatus.COMPLETED
        assert status["result"] == {"result": 30}


class TestDelayMethod:
    """태스크.delay() 메서드 테스트"""
    
    @pytest.mark.asyncio
    async def test_delay_creates_task(self):
        """delay()가 태스크를 생성하는지 확인"""
        task_id = await simple_task.delay(5, 10, tenant_id="delay_test")
        
        assert task_id is not None
        
        status = await get_task_status(task_id)
        assert status is not None
        assert status["task_name"] == "test.simple_task"


# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

