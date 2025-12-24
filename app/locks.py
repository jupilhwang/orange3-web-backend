"""
Async Lock Manager for Orange3 Web Backend
Provides thread-safe locking for concurrent operations
"""
import asyncio
from typing import Dict, Optional
from contextlib import asynccontextmanager
from fastapi import HTTPException


class AsyncLockManager:
    """
    Manages async locks for resources.
    
    Thread-safe lock management for concurrent access to workflows,
    nodes, and other resources.
    
    Usage:
        async with lock_manager.lock("workflow:123"):
            # Safe to modify workflow 123
            ...
    """
    
    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._meta_lock = asyncio.Lock()
        self._lock_counts: Dict[str, int] = {}  # Reference counting
    
    async def _get_lock(self, resource_id: str) -> asyncio.Lock:
        """Get or create a lock for a resource."""
        async with self._meta_lock:
            if resource_id not in self._locks:
                self._locks[resource_id] = asyncio.Lock()
                self._lock_counts[resource_id] = 0
            return self._locks[resource_id]
    
    async def acquire(self, resource_id: str, timeout: Optional[float] = 30.0) -> bool:
        """
        Acquire a lock for a resource.
        
        Args:
            resource_id: Unique identifier for the resource
            timeout: Maximum time to wait for lock (None = wait forever)
            
        Returns:
            True if lock acquired, False if timeout
        """
        lock = await self._get_lock(resource_id)
        
        try:
            if timeout is not None:
                await asyncio.wait_for(lock.acquire(), timeout=timeout)
            else:
                await lock.acquire()
            
            async with self._meta_lock:
                self._lock_counts[resource_id] = self._lock_counts.get(resource_id, 0) + 1
            
            return True
        except asyncio.TimeoutError:
            return False
    
    async def release(self, resource_id: str):
        """Release a lock for a resource."""
        async with self._meta_lock:
            if resource_id in self._locks and self._locks[resource_id].locked():
                self._locks[resource_id].release()
                self._lock_counts[resource_id] = max(0, self._lock_counts.get(resource_id, 1) - 1)
    
    def is_locked(self, resource_id: str) -> bool:
        """Check if a resource is currently locked."""
        return resource_id in self._locks and self._locks[resource_id].locked()
    
    @asynccontextmanager
    async def lock(self, resource_id: str, timeout: Optional[float] = 30.0):
        """
        Context manager for acquiring and releasing a lock.
        
        Args:
            resource_id: Unique identifier for the resource
            timeout: Maximum time to wait for lock
            
        Raises:
            HTTPException: If lock cannot be acquired (409 Conflict)
            
        Usage:
            async with lock_manager.lock("workflow:123"):
                # Safe to modify
                ...
        """
        acquired = await self.acquire(resource_id, timeout)
        if not acquired:
            raise HTTPException(
                status_code=409,
                detail=f"Resource '{resource_id}' is being modified by another user. Please try again."
            )
        try:
            yield
        finally:
            await self.release(resource_id)
    
    async def cleanup_unused_locks(self):
        """Remove locks that are no longer in use."""
        async with self._meta_lock:
            to_remove = [
                rid for rid, count in self._lock_counts.items()
                if count == 0 and not self._locks[rid].locked()
            ]
            for rid in to_remove:
                del self._locks[rid]
                del self._lock_counts[rid]


# Global lock manager instances
workflow_locks = AsyncLockManager()
tenant_locks = AsyncLockManager()


# Convenience functions
@asynccontextmanager
async def lock_workflow(workflow_id: str, timeout: float = 30.0):
    """Lock a workflow for exclusive access."""
    async with workflow_locks.lock(f"workflow:{workflow_id}", timeout):
        yield


@asynccontextmanager
async def lock_tenant(tenant_id: str, timeout: float = 30.0):
    """Lock tenant-level operations."""
    async with tenant_locks.lock(f"tenant:{tenant_id}", timeout):
        yield


@asynccontextmanager
async def lock_node(workflow_id: str, node_id: str, timeout: float = 30.0):
    """Lock a specific node within a workflow."""
    async with workflow_locks.lock(f"node:{workflow_id}:{node_id}", timeout):
        yield

