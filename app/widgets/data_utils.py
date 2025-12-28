"""
Common data loading utilities for widget backends.
Handles various data path formats including k-Means clustered data.
Provides session-based data isolation for multi-user environments.
"""

import logging
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any

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
# DataSessionManager - Session-based data isolation
# =============================================================================

class DataSessionManager:
    """
    세션 기반 데이터 격리 관리자.
    
    다중 사용자 환경에서 각 사용자의 데이터를 격리합니다.
    TTL(Time-To-Live) 기반 자동 만료를 지원합니다.
    
    사용 예시:
        # 데이터 저장
        DataSessionManager.store("session_abc", "sampler/sample_123", table)
        
        # 데이터 조회
        data = DataSessionManager.get("session_abc", "sampler/sample_123")
        
        # 세션 정리
        DataSessionManager.cleanup("session_abc")
    """
    
    _sessions: Dict[str, Dict[str, Any]] = {}
    _lock = threading.Lock()
    DEFAULT_TTL = 3600  # 1시간 (초)
    
    @classmethod
    def store(cls, session_id: str, data_id: str, data: 'Table', 
              ttl: int = None, metadata: dict = None) -> str:
        """
        세션에 데이터 저장.
        
        Args:
            session_id: 세션 ID (브라우저/사용자별 고유)
            data_id: 데이터 ID (예: "sampler/sample_123")
            data: Orange.data.Table 객체
            ttl: Time-To-Live (초), 기본값: 3600 (1시간)
            metadata: 추가 메타데이터
            
        Returns:
            저장된 데이터의 전체 경로 (data_id)
        """
        with cls._lock:
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
    def get(cls, session_id: str, data_id: str) -> Optional['Table']:
        """
        세션에서 데이터 조회.
        
        Args:
            session_id: 세션 ID
            data_id: 데이터 ID
            
        Returns:
            Orange.data.Table 또는 None (없거나 만료됨)
        """
        with cls._lock:
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
    def get_metadata(cls, session_id: str, data_id: str) -> Optional[dict]:
        """데이터의 메타데이터 조회."""
        with cls._lock:
            session_data = cls._sessions.get(session_id, {})
            entry = session_data.get(data_id)
            
            if entry is None:
                return None
            
            return entry.get("metadata")
    
    @classmethod
    def cleanup(cls, session_id: str) -> int:
        """
        세션의 모든 데이터 삭제.
        
        Args:
            session_id: 삭제할 세션 ID
            
        Returns:
            삭제된 데이터 수
        """
        with cls._lock:
            if session_id in cls._sessions:
                count = len(cls._sessions[session_id])
                del cls._sessions[session_id]
                logger.info(f"Cleaned up session: {session_id}, removed {count} items")
                return count
            return 0
    
    @classmethod
    def cleanup_expired(cls) -> int:
        """
        모든 세션에서 만료된 데이터 삭제.
        
        Returns:
            삭제된 데이터 수
        """
        with cls._lock:
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
    def get_stats(cls) -> dict:
        """
        세션 통계 조회.
        
        Returns:
            {"sessions": 세션 수, "total_items": 총 데이터 수}
        """
        with cls._lock:
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
    Load data from various path formats.
    
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
        # Session-based data lookup (if session_id provided)
        if session_id and (data_path.startswith("sampler/") or data_path.startswith("kmeans/")):
            data = DataSessionManager.get(session_id, data_path)
            if data is not None:
                logger.debug(f"Loaded from session: {session_id}/{data_path}")
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

