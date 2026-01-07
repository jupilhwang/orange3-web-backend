"""
Common data loading utilities for widget backends.
Handles various data path formats including k-Means clustered data.
Provides session-based data isolation for multi-user environments.
"""

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional, Dict, Any, Callable, TypeVar

logger = logging.getLogger(__name__)

# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"

# Check Orange3 availability
try:
    from Orange.data import Table
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False


# =============================================================================
# CPU-bound task executor for Orange3 operations
# =============================================================================

# Process pool for CPU-bound Orange3 operations (GIL bypass)
# Use max 4 workers to avoid memory issues with Orange3
_process_pool: Optional[ProcessPoolExecutor] = None

# Thread pool for I/O-bound operations
_thread_pool: Optional[ThreadPoolExecutor] = None

T = TypeVar('T')


def get_process_pool() -> ProcessPoolExecutor:
    """Get or create the process pool executor."""
    global _process_pool
    if _process_pool is None:
        import os
        max_workers = min(4, os.cpu_count() or 2)
        _process_pool = ProcessPoolExecutor(max_workers=max_workers)
        logger.info(f"Created ProcessPoolExecutor with {max_workers} workers")
    return _process_pool


def get_thread_pool() -> ThreadPoolExecutor:
    """Get or create the thread pool executor."""
    global _thread_pool
    if _thread_pool is None:
        import os
        max_workers = min(8, (os.cpu_count() or 2) * 2)
        _thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"Created ThreadPoolExecutor with {max_workers} workers")
    return _thread_pool


async def run_in_process(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a CPU-bound function in a separate process.
    
    This bypasses the GIL and allows true parallel execution for
    CPU-intensive Orange3 operations like clustering, classification, etc.
    
    Args:
        func: Function to execute (must be picklable)
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result from the function
        
    Example:
        # In an async endpoint:
        result = await run_in_process(heavy_computation, data, k=5)
    """
    loop = asyncio.get_event_loop()
    pool = get_process_pool()
    
    # Create a partial function if kwargs are provided
    if kwargs:
        func = partial(func, **kwargs)
    
    return await loop.run_in_executor(pool, func, *args)


async def run_in_thread(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a blocking I/O function in a thread pool.
    
    Use this for I/O-bound operations that block (file reading, etc.)
    but don't benefit from process-level parallelism.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Result from the function
    """
    loop = asyncio.get_event_loop()
    pool = get_thread_pool()
    
    if kwargs:
        func = partial(func, **kwargs)
    
    return await loop.run_in_executor(pool, func, *args)


def shutdown_executors():
    """Shutdown all executor pools gracefully."""
    global _process_pool, _thread_pool
    
    if _process_pool is not None:
        _process_pool.shutdown(wait=True)
        _process_pool = None
        logger.info("ProcessPoolExecutor shutdown complete")
    
    if _thread_pool is not None:
        _thread_pool.shutdown(wait=True)
        _thread_pool = None
        logger.info("ThreadPoolExecutor shutdown complete")


# =============================================================================
# DataSessionManager - Session-based data isolation
# =============================================================================

class DataSessionManager:
    """
    세션 기반 데이터 격리 관리자 (asyncio 호환).
    
    다중 사용자 환경에서 각 사용자의 데이터를 격리합니다.
    TTL(Time-To-Live) 기반 자동 만료를 지원합니다.
    
    Note:
        모든 메서드는 async입니다. 동기 컨텍스트에서는 
        store_sync/get_sync를 사용하세요.
    
    사용 예시:
        # 데이터 저장 (async)
        await DataSessionManager.store("session_abc", "sampler/sample_123", table)
        
        # 데이터 조회 (async)
        data = await DataSessionManager.get("session_abc", "sampler/sample_123")
        
        # 동기 저장 (sync context)
        DataSessionManager.store_sync("session_abc", "data_id", table)
    """
    
    _sessions: Dict[str, Dict[str, Any]] = {}
    _lock: asyncio.Lock = None  # Lazy initialization
    DEFAULT_TTL = 3600  # 1시간 (초)
    
    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        """Get or create the async lock (lazy initialization)."""
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock
    
    @classmethod
    async def store(cls, session_id: str, data_id: str, data: 'Table', 
                    ttl: int = None, metadata: dict = None) -> str:
        """
        세션에 데이터 저장 (async).
        
        Args:
            session_id: 세션 ID (브라우저/사용자별 고유)
            data_id: 데이터 ID (예: "sampler/sample_123")
            data: Orange.data.Table 객체
            ttl: Time-To-Live (초), 기본값: 3600 (1시간)
            metadata: 추가 메타데이터
            
        Returns:
            저장된 데이터의 전체 경로 (data_id)
        """
        async with cls._get_lock():
            if session_id not in cls._sessions:
                cls._sessions[session_id] = {}
            
            cls._sessions[session_id][data_id] = {
                "data": data,
                "created_at": time.time(),
                "expires_at": time.time() + (ttl or cls.DEFAULT_TTL),
                "metadata": metadata or {}
            }
            
            logger.info(f"Stored data: session={session_id}, id={data_id}")
            return data_id
    
    @classmethod
    def store_sync(cls, session_id: str, data_id: str, data: 'Table', 
                   ttl: int = None, metadata: dict = None) -> str:
        """
        세션에 데이터 저장 (sync 버전 - 동기 컨텍스트용).
        
        Note: 락 없이 동작합니다. 동기 컨텍스트에서만 사용하세요.
        """
        if session_id not in cls._sessions:
            cls._sessions[session_id] = {}
        
        cls._sessions[session_id][data_id] = {
            "data": data,
            "created_at": time.time(),
            "expires_at": time.time() + (ttl or cls.DEFAULT_TTL),
            "metadata": metadata or {}
        }
        
        logger.info(f"Stored data (sync): session={session_id}, id={data_id}")
        return data_id
    
    @classmethod
    async def get(cls, session_id: str, data_id: str) -> Optional['Table']:
        """
        세션에서 데이터 조회 (async).
        
        Args:
            session_id: 세션 ID
            data_id: 데이터 ID
            
        Returns:
            Orange.data.Table 또는 None (없거나 만료됨)
        """
        async with cls._get_lock():
            session_data = cls._sessions.get(session_id, {})
            entry = session_data.get(data_id)
            
            if entry is None:
                return None
            
            # 만료 체크
            if time.time() > entry["expires_at"]:
                logger.info(f"Data expired: session={session_id}, id={data_id}")
                del session_data[data_id]
                return None
            
            return entry.get("data")
    
    @classmethod
    def get_sync(cls, session_id: str, data_id: str) -> Optional['Table']:
        """
        세션에서 데이터 조회 (sync 버전).
        
        Note: 락 없이 동작합니다. 동기 컨텍스트에서만 사용하세요.
        """
        session_data = cls._sessions.get(session_id, {})
        entry = session_data.get(data_id)
        
        if entry is None:
            return None
        
        # 만료 체크
        if time.time() > entry["expires_at"]:
            logger.info(f"Data expired: session={session_id}, id={data_id}")
            del session_data[data_id]
            return None
        
        return entry.get("data")
    
    @classmethod
    async def get_metadata(cls, session_id: str, data_id: str) -> Optional[dict]:
        """데이터의 메타데이터 조회 (async)."""
        async with cls._get_lock():
            session_data = cls._sessions.get(session_id, {})
            entry = session_data.get(data_id)
            
            if entry is None:
                return None
            
            return entry.get("metadata")
    
    @classmethod
    async def cleanup(cls, session_id: str) -> int:
        """
        세션의 모든 데이터 삭제 (async).
        
        Args:
            session_id: 삭제할 세션 ID
            
        Returns:
            삭제된 데이터 수
        """
        async with cls._get_lock():
            if session_id in cls._sessions:
                count = len(cls._sessions[session_id])
                del cls._sessions[session_id]
                logger.info(f"Cleaned up session: {session_id}, removed {count} items")
                return count
            return 0
    
    @classmethod
    async def cleanup_expired(cls) -> int:
        """
        모든 세션에서 만료된 데이터 삭제 (async).
        
        Returns:
            삭제된 데이터 수
        """
        async with cls._get_lock():
            now = time.time()
            count = 0
            
            for session_id in list(cls._sessions.keys()):
                session_data = cls._sessions[session_id]
                expired_ids = [
                    data_id for data_id, entry in session_data.items()
                    if now > entry["expires_at"]
                ]
                
                for data_id in expired_ids:
                    del session_data[data_id]
                    count += 1
                
                # 빈 세션 삭제
                if not session_data:
                    del cls._sessions[session_id]
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired items")
            
            return count
    
    @classmethod
    async def get_stats(cls) -> dict:
        """
        세션 통계 조회 (async).
        
        Returns:
            {"sessions": 세션 수, "total_items": 총 데이터 수}
        """
        async with cls._get_lock():
            total_items = sum(len(s) for s in cls._sessions.values())
            return {
                "sessions": len(cls._sessions),
                "total_items": total_items
            }
    
    @classmethod
    def get_stats_sync(cls) -> dict:
        """세션 통계 조회 (sync 버전)."""
        total_items = sum(len(s) for s in cls._sessions.values())
        return {
            "sessions": len(cls._sessions),
            "total_items": total_items
        }


# =============================================================================
# Legacy global storage (for backward compatibility)
# TODO: Migrate to DataSessionManager
# =============================================================================


def load_data(data_path: str, session_id: str = None) -> Optional['Table']:
    """
    Load data from various path formats (sync version).
    
    Supported formats:
    - "datasets/iris" - Orange3 built-in datasets
    - "uploads/xxx" - Uploaded files
    - "kmeans/{cluster_id}" - k-Means clustered data
    - "sampler/{sample_id}" - Data sampler results
    - Direct file paths
    
    Args:
        data_path: 데이터 경로
        session_id: 세션 ID (sampler/kmeans 데이터의 세션 기반 조회용)
    
    Returns:
        Orange.data.Table or None if loading fails
    """
    if not ORANGE_AVAILABLE:
        logger.error("Orange3 not available")
        return None
    
    try:
        # Session-based data lookup (if session_id provided or special prefix)
        session_prefixes = ("sampler/", "kmeans/", "confusion_selection_")
        if data_path.startswith(session_prefixes):
            # Try session-based lookup first (sync version)
            effective_session = session_id or "default"
            data = DataSessionManager.get_sync(effective_session, data_path)
            if data is not None:
                logger.debug(f"Loaded from session: {effective_session}/{data_path}")
                return data
            # Fallback to legacy storage if not found in session
        
        # Handle k-Means clustered data (legacy global storage)
        if data_path.startswith("kmeans/"):
            return _load_kmeans_data(data_path)
        
        # Handle Data Sampler results (legacy global storage)
        if data_path.startswith("sampler/"):
            return _load_sampler_data(data_path)
        
        # Handle uploads
        if data_path.startswith("uploads/"):
            resolved_path = str(UPLOAD_DIR / data_path.replace("uploads/", ""))
            return Table(resolved_path)
        
        # Handle built-in datasets
        if data_path.startswith("datasets/"):
            dataset_name = data_path.replace("datasets/", "").split(".")[0]
            return Table(dataset_name)
        
        # Direct path or dataset name
        return Table(data_path)
        
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}: {e}")
        return None


def _load_kmeans_data(data_path: str) -> Optional['Table']:
    """
    Load k-Means clustered data.
    
    Args:
        data_path: "kmeans/{cluster_id}" format
        
    Returns:
        Annotated Table with cluster column
    """
    try:
        # Import k-Means results storage
        from .kmeans import _kmeans_results
        
        cluster_id = data_path.replace("kmeans/", "")
        
        if cluster_id not in _kmeans_results:
            logger.error(f"K-Means result not found: {cluster_id}")
            return None
        
        result = _kmeans_results[cluster_id]
        return result.get("data")
        
    except Exception as e:
        logger.error(f"Failed to load k-Means data: {e}")
        return None


def _load_sampler_data(data_path: str) -> Optional['Table']:
    """
    Load Data Sampler results.
    
    Args:
        data_path: "sampler/{sampler_id}" format
        
    Returns:
        Sampled or remaining data Table
    """
    try:
        # Import Data Sampler results storage
        from .data_sampler import _sampler_results
        
        sampler_id = data_path.replace("sampler/", "")
        
        if sampler_id not in _sampler_results:
            logger.error(f"Sampler result not found: {sampler_id}")
            return None
        
        result = _sampler_results[sampler_id]
        return result.get("data")
        
    except Exception as e:
        logger.error(f"Failed to load sampler data: {e}")
        return None


def resolve_data_path(data_path: str) -> str:
    """
    Resolve data path to actual file path (for legacy compatibility).
    
    Note: For k-Means data, use load_data() instead.
    """
    if data_path.startswith("uploads/"):
        return str(UPLOAD_DIR / data_path.replace("uploads/", ""))
    elif data_path.startswith("datasets/"):
        dataset_name = data_path.replace("datasets/", "").split(".")[0]
        return dataset_name
    return data_path


def save_data(data_id: str, data: 'Table', session_id: str = None, 
              ttl: int = None, metadata: dict = None) -> str:
    """
    Save data to session storage (sync version).
    
    This is a convenience wrapper around DataSessionManager.store_sync.
    
    Args:
        data_id: 데이터 ID (예: "confusion_selection_xxx")
        data: Orange.data.Table 객체
        session_id: 세션 ID (None이면 기본 세션 사용)
        ttl: Time-To-Live (초)
        metadata: 추가 메타데이터
        
    Returns:
        저장된 데이터의 전체 경로 (data_id)
    """
    # Use default session if none provided
    if session_id is None:
        session_id = "default"
    
    return DataSessionManager.store_sync(session_id, data_id, data, ttl, metadata)


async def save_data_async(data_id: str, data: 'Table', session_id: str = None, 
                          ttl: int = None, metadata: dict = None) -> str:
    """
    Save data to session storage (async version).
    
    This is a convenience wrapper around DataSessionManager.store.
    
    Args:
        data_id: 데이터 ID (예: "confusion_selection_xxx")
        data: Orange.data.Table 객체
        session_id: 세션 ID (None이면 기본 세션 사용)
        ttl: Time-To-Live (초)
        metadata: 추가 메타데이터
        
    Returns:
        저장된 데이터의 전체 경로 (data_id)
    """
    if session_id is None:
        session_id = "default"
    
    return await DataSessionManager.store(session_id, data_id, data, ttl, metadata)

