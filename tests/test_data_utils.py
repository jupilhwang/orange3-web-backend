"""
data_utils.py 및 DataSessionManager 테스트
테스트 케이스: 20개

NOTE: DataSessionManager는 async 메서드를 사용하므로 pytest-asyncio 필요
"""

import pytest
import time
import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.data_utils import DataSessionManager, load_data, ORANGE_AVAILABLE


class TestDataSessionManager:
    """DataSessionManager 테스트 (12개 케이스)"""
    
    def setup_method(self):
        """각 테스트 전 세션 초기화"""
        DataSessionManager._sessions = {}
    
    # =========================================================================
    # 1-4: 기본 CRUD 테스트
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_01_store_and_get(self):
        """TC01: 데이터 저장 및 조회"""
        session_id = "test_session_1"
        data_id = "sampler/sample_001"
        mock_data = {"test": "data"}  # 실제는 Table 객체
        
        # 저장
        result = await DataSessionManager.store(session_id, data_id, mock_data)
        assert result == data_id
        
        # 조회
        retrieved = await DataSessionManager.get(session_id, data_id)
        assert retrieved == mock_data
        print("✅ TC01 통과: 데이터 저장 및 조회 성공")
    
    @pytest.mark.asyncio
    async def test_02_get_nonexistent_session(self):
        """TC02: 존재하지 않는 세션 조회"""
        result = await DataSessionManager.get("nonexistent_session", "any_id")
        assert result is None
        print("✅ TC02 통과: 존재하지 않는 세션 → None 반환")
    
    @pytest.mark.asyncio
    async def test_03_get_nonexistent_data(self):
        """TC03: 존재하지 않는 데이터 조회"""
        session_id = "test_session_3"
        await DataSessionManager.store(session_id, "existing_id", "data")
        
        result = await DataSessionManager.get(session_id, "nonexistent_id")
        assert result is None
        print("✅ TC03 통과: 존재하지 않는 데이터 → None 반환")
    
    @pytest.mark.asyncio
    async def test_04_store_with_metadata(self):
        """TC04: 메타데이터와 함께 저장"""
        session_id = "test_session_4"
        data_id = "kmeans/cluster_001"
        metadata = {"n_clusters": 3, "type": "kmeans"}
        
        await DataSessionManager.store(session_id, data_id, "data", metadata=metadata)
        
        retrieved_meta = await DataSessionManager.get_metadata(session_id, data_id)
        assert retrieved_meta == metadata
        print("✅ TC04 통과: 메타데이터 저장 및 조회 성공")
    
    # =========================================================================
    # 5-7: TTL(만료) 테스트
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_05_ttl_not_expired(self):
        """TC05: TTL 내 데이터 조회"""
        session_id = "test_session_5"
        await DataSessionManager.store(session_id, "data_id", "data", ttl=10)
        
        result = await DataSessionManager.get(session_id, "data_id")
        assert result == "data"
        print("✅ TC05 통과: TTL 내 데이터 조회 성공")
    
    @pytest.mark.asyncio
    async def test_06_ttl_expired(self):
        """TC06: TTL 만료 후 데이터 조회"""
        session_id = "test_session_6"
        await DataSessionManager.store(session_id, "data_id", "data", ttl=1)
        
        # 2초 대기 (TTL 만료)
        await asyncio.sleep(2)
        
        result = await DataSessionManager.get(session_id, "data_id")
        assert result is None
        print("✅ TC06 통과: TTL 만료 후 → None 반환")
    
    @pytest.mark.asyncio
    async def test_07_cleanup_expired(self):
        """TC07: 만료된 데이터 일괄 정리"""
        await DataSessionManager.store("sess1", "data1", "d1", ttl=1)
        await DataSessionManager.store("sess1", "data2", "d2", ttl=100)
        await DataSessionManager.store("sess2", "data3", "d3", ttl=1)
        
        await asyncio.sleep(2)
        
        count = await DataSessionManager.cleanup_expired()
        assert count == 2  # data1, data3 삭제됨
        
        # data2는 남아있어야 함
        assert await DataSessionManager.get("sess1", "data2") == "d2"
        print("✅ TC07 통과: 만료된 데이터 2개 정리됨")
    
    # =========================================================================
    # 8-9: 세션 관리 테스트
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_08_cleanup_session(self):
        """TC08: 특정 세션 전체 삭제"""
        session_id = "test_session_8"
        await DataSessionManager.store(session_id, "data1", "d1")
        await DataSessionManager.store(session_id, "data2", "d2")
        await DataSessionManager.store(session_id, "data3", "d3")
        
        count = await DataSessionManager.cleanup(session_id)
        assert count == 3
        
        assert await DataSessionManager.get(session_id, "data1") is None
        print("✅ TC08 통과: 세션 전체 삭제 (3개 항목)")
    
    @pytest.mark.asyncio
    async def test_09_get_stats(self):
        """TC09: 통계 조회"""
        await DataSessionManager.store("sess1", "d1", "data")
        await DataSessionManager.store("sess1", "d2", "data")
        await DataSessionManager.store("sess2", "d3", "data")
        
        stats = await DataSessionManager.get_stats()
        assert stats["sessions"] == 2
        assert stats["total_items"] == 3
        print("✅ TC09 통과: 통계 - 2 세션, 3 항목")
    
    # =========================================================================
    # 10-12: 동시성 테스트
    # =========================================================================
    
    @pytest.mark.asyncio
    async def test_10_concurrent_write(self):
        """TC10: 동시 쓰기 테스트"""
        results = []
        
        async def write_data(i):
            await DataSessionManager.store(f"sess_{i}", f"data_{i}", f"value_{i}")
            results.append(i)
        
        tasks = [write_data(i) for i in range(10)]
        await asyncio.gather(*tasks)
        
        assert len(results) == 10
        stats = await DataSessionManager.get_stats()
        assert stats["total_items"] == 10
        print("✅ TC10 통과: 10개 동시 쓰기 성공")
    
    @pytest.mark.asyncio
    async def test_11_concurrent_read_write(self):
        """TC11: 동시 읽기/쓰기 테스트"""
        session_id = "concurrent_session"
        await DataSessionManager.store(session_id, "shared_data", "initial")
        
        read_results = []
        
        async def read_data():
            for _ in range(5):
                result = await DataSessionManager.get(session_id, "shared_data")
                if result:
                    read_results.append(result)
        
        async def write_data():
            for i in range(5):
                await DataSessionManager.store(session_id, "shared_data", f"value_{i}")
        
        tasks = [read_data(), write_data(), read_data()]
        await asyncio.gather(*tasks)
        
        # 읽기가 실패하지 않아야 함
        assert len(read_results) > 0
        print(f"✅ TC11 통과: 동시 읽기/쓰기 - {len(read_results)}회 읽기 성공")
    
    @pytest.mark.asyncio
    async def test_12_isolation_between_sessions(self):
        """TC12: 세션 간 데이터 격리"""
        await DataSessionManager.store("session_A", "private_data", "A's data")
        await DataSessionManager.store("session_B", "private_data", "B's data")
        
        assert await DataSessionManager.get("session_A", "private_data") == "A's data"
        assert await DataSessionManager.get("session_B", "private_data") == "B's data"
        
        # A 세션 삭제 후 B는 영향 없어야 함
        await DataSessionManager.cleanup("session_A")
        assert await DataSessionManager.get("session_B", "private_data") == "B's data"
        print("✅ TC12 통과: 세션 간 데이터 격리 확인")


class TestLoadData:
    """load_data() 함수 테스트 (8개 케이스)"""
    
    def setup_method(self):
        """각 테스트 전 초기화"""
        DataSessionManager._sessions = {}
    
    # =========================================================================
    # 13-16: 경로 형식 테스트
    # =========================================================================
    
    def test_13_load_builtin_dataset(self):
        """TC13: Orange3 내장 데이터셋 로드"""
        if not ORANGE_AVAILABLE:
            pytest.skip("Orange3 not installed")
        
        data = load_data("datasets/iris")
        assert data is not None
        assert len(data) == 150  # Iris는 150개 행
        print("✅ TC13 통과: Iris 데이터셋 로드 성공 (150행)")
    
    def test_14_load_nonexistent_dataset(self):
        """TC14: 존재하지 않는 데이터셋"""
        data = load_data("datasets/nonexistent_xyz")
        assert data is None
        print("✅ TC14 통과: 존재하지 않는 데이터셋 → None")
    
    def test_17_empty_path(self):
        """TC17: 빈 경로"""
        # 빈 경로는 Table()이 실패할 것
        data = load_data("")
        assert data is None
        print("✅ TC17 통과: 빈 경로 → None")
    
    def test_18_invalid_path_format(self):
        """TC18: 잘못된 경로 형식"""
        data = load_data("invalid/unknown/path/format")
        assert data is None
        print("✅ TC18 통과: 잘못된 경로 → None")
    
    def test_20_load_multiple_datasets(self):
        """TC20: 여러 데이터셋 순차 로드"""
        if not ORANGE_AVAILABLE:
            pytest.skip("Orange3 not installed")
        
        datasets = ["datasets/iris", "datasets/housing", "datasets/zoo"]
        loaded = []
        
        for ds in datasets:
            data = load_data(ds)
            if data is not None:
                loaded.append(ds)
        
        # 최소 1개 이상 로드 성공
        assert len(loaded) >= 1
        print(f"✅ TC20 통과: {len(loaded)}개 데이터셋 로드 성공")


class TestExecutors:
    """run_in_process, run_in_thread 테스트"""
    
    @pytest.mark.asyncio
    async def test_run_in_thread(self):
        """TC21: run_in_thread 테스트"""
        from app.core.data_utils import run_in_thread
        
        def slow_io():
            time.sleep(0.1)
            return "done"
        
        result = await run_in_thread(slow_io)
        assert result == "done"
        print("✅ TC21 통과: run_in_thread 성공")
    
    @pytest.mark.asyncio
    async def test_run_in_process(self):
        """TC22: run_in_process 테스트 (pickle 가능한 함수만)"""
        from app.core.data_utils import run_in_process
        import math
        
        # 내장 함수나 모듈 수준 함수는 pickle 가능
        # local function은 pickle 불가능
        result = await run_in_process(math.sqrt, 16)
        assert result == 4.0
        print("✅ TC22 통과: run_in_process 성공")


# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
