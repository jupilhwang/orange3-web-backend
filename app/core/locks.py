"""
Concurrency utilities for Orange3 Web Backend

설계 원칙:
    Orange3 Web은 각 사용자가 자신만의 워크플로우를 편집하는 패턴입니다.
    따라서 복잡한 분산 락(Advisory Lock, Redis Lock)은 불필요합니다.
    
    현재 구현:
    - DB 트랜잭션으로 데이터 무결성 보장
    - 인메모리 어댑터 보호를 위한 간단한 asyncio.Lock
    - (선택) 낙관적 잠금(version 컬럼)으로 다중 탭 충돌 감지
"""
import asyncio
from typing import Dict, Optional
from contextlib import asynccontextmanager
from fastapi import HTTPException


class SimpleLockManager:
    """
    간단한 인메모리 락 매니저.
    
    주 용도:
    - 인메모리 어댑터(_workflow_adapters) 동시 접근 보호
    - 위젯 실행 중 상태 보호
    
    Note: 이것은 분산 락이 아닙니다. 단일 프로세스 내에서만 동작합니다.
    """
    
    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._meta_lock = asyncio.Lock()
    
    async def _get_lock(self, resource_id: str) -> asyncio.Lock:
        """Get or create a lock for a resource."""
        async with self._meta_lock:
            if resource_id not in self._locks:
                self._locks[resource_id] = asyncio.Lock()
            return self._locks[resource_id]
    
    @asynccontextmanager
    async def lock(self, resource_id: str, timeout: Optional[float] = 30.0):
        """리소스 락 획득."""
        lock = await self._get_lock(resource_id)
        
        try:
            if timeout is not None:
                await asyncio.wait_for(lock.acquire(), timeout=timeout)
            else:
                await lock.acquire()
            yield
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=409,
                detail="리소스가 사용 중입니다. 잠시 후 다시 시도해주세요."
            )
        finally:
            if lock.locked():
                lock.release()


# 전역 락 매니저
_lock_manager = SimpleLockManager()


# =============================================================================
# 인메모리 어댑터 보호용 락
# =============================================================================

@asynccontextmanager
async def lock_workflow(workflow_id: str, timeout: float = 30.0):
    """
    워크플로우 어댑터 동시 접근 보호.
    
    주 용도: _workflow_adapters 딕셔너리 접근 보호
    """
    async with _lock_manager.lock(f"workflow:{workflow_id}", timeout):
        yield


@asynccontextmanager
async def lock_tenant(tenant_id: str, timeout: float = 30.0):
    """
    테넌트 레벨 작업 보호.
    
    주 용도: 워크플로우 생성 시 어댑터 초기화
    """
    async with _lock_manager.lock(f"tenant:{tenant_id}", timeout):
        yield


# 하위 호환성을 위해 유지 (사용하지 않아도 됨)
workflow_locks = _lock_manager


# =============================================================================
# 낙관적 잠금 유틸리티 (Optimistic Locking) - 선택적 사용
# =============================================================================

async def update_with_version_check(
    session,
    model_class,
    id: str,
    current_version: int,
    **updates
) -> bool:
    """
    버전 체크를 통한 낙관적 잠금 업데이트.
    
    사용 시기:
    - 동일 사용자가 여러 탭에서 같은 워크플로우 편집 시 충돌 감지
    - 일반적인 경우에는 불필요
    
    Args:
        session: DB 세션
        model_class: SQLAlchemy 모델 클래스 (version 컬럼 필요)
        id: 레코드 ID
        current_version: 클라이언트가 보유한 버전
        **updates: 업데이트할 필드들
        
    Raises:
        HTTPException: 버전 충돌 시 409 에러
    """
    from sqlalchemy import update
    
    stmt = (
        update(model_class)
        .where(model_class.id == id)
        .where(model_class.version == current_version)
        .values(version=current_version + 1, **updates)
    )
    
    result = await session.execute(stmt)
    
    if result.rowcount == 0:
        raise HTTPException(
            status_code=409,
            detail="다른 곳에서 수정되었습니다. 새로고침 후 다시 시도해주세요."
        )
    
    await session.commit()
    return True


