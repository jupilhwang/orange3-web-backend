"""
data_utils.py 및 DataSessionManager 테스트
테스트 케이스: 20개
"""

import time
import threading
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.widgets.data_utils import DataSessionManager, load_data, ORANGE_AVAILABLE


class TestDataSessionManager:
    """DataSessionManager 테스트 (12개 케이스)"""
    
    def setup_method(self):
        """각 테스트 전 세션 초기화"""
        DataSessionManager._sessions = {}
    
    # =========================================================================
    # 1-4: 기본 CRUD 테스트
    # =========================================================================
    
    def test_01_store_and_get(self):
        """TC01: 데이터 저장 및 조회"""
        session_id = "test_session_1"
        data_id = "sampler/sample_001"
        mock_data = {"test": "data"}  # 실제는 Table 객체
        
        # 저장
        result = DataSessionManager.store(session_id, data_id, mock_data)
        assert result == data_id
        
        # 조회
        retrieved = DataSessionManager.get(session_id, data_id)
        assert retrieved == mock_data
        print("✅ TC01 통과: 데이터 저장 및 조회 성공")
    
    def test_02_get_nonexistent_session(self):
        """TC02: 존재하지 않는 세션 조회"""
        result = DataSessionManager.get("nonexistent_session", "any_id")
        assert result is None
        print("✅ TC02 통과: 존재하지 않는 세션 → None 반환")
    
    def test_03_get_nonexistent_data(self):
        """TC03: 존재하지 않는 데이터 조회"""
        session_id = "test_session_3"
        DataSessionManager.store(session_id, "existing_id", "data")
        
        result = DataSessionManager.get(session_id, "nonexistent_id")
        assert result is None
        print("✅ TC03 통과: 존재하지 않는 데이터 → None 반환")
    
    def test_04_store_with_metadata(self):
        """TC04: 메타데이터와 함께 저장"""
        session_id = "test_session_4"
        data_id = "kmeans/cluster_001"
        metadata = {"n_clusters": 3, "type": "kmeans"}
        
        DataSessionManager.store(session_id, data_id, "data", metadata=metadata)
        
        retrieved_meta = DataSessionManager.get_metadata(session_id, data_id)
        assert retrieved_meta == metadata
        print("✅ TC04 통과: 메타데이터 저장 및 조회 성공")
    
    # =========================================================================
    # 5-7: TTL(만료) 테스트
    # =========================================================================
    
    def test_05_ttl_not_expired(self):
        """TC05: TTL 내 데이터 조회"""
        session_id = "test_session_5"
        DataSessionManager.store(session_id, "data_id", "data", ttl=10)
        
        result = DataSessionManager.get(session_id, "data_id")
        assert result == "data"
        print("✅ TC05 통과: TTL 내 데이터 조회 성공")
    
    def test_06_ttl_expired(self):
        """TC06: TTL 만료 후 데이터 조회"""
        session_id = "test_session_6"
        DataSessionManager.store(session_id, "data_id", "data", ttl=1)
        
        # 2초 대기 (TTL 만료)
        time.sleep(2)
        
        result = DataSessionManager.get(session_id, "data_id")
        assert result is None
        print("✅ TC06 통과: TTL 만료 후 → None 반환")
    
    def test_07_cleanup_expired(self):
        """TC07: 만료된 데이터 일괄 정리"""
        DataSessionManager.store("sess1", "data1", "d1", ttl=1)
        DataSessionManager.store("sess1", "data2", "d2", ttl=100)
        DataSessionManager.store("sess2", "data3", "d3", ttl=1)
        
        time.sleep(2)
        
        count = DataSessionManager.cleanup_expired()
        assert count == 2  # data1, data3 삭제됨
        
        # data2는 남아있어야 함
        assert DataSessionManager.get("sess1", "data2") == "d2"
        print("✅ TC07 통과: 만료된 데이터 2개 정리됨")
    
    # =========================================================================
    # 8-9: 세션 관리 테스트
    # =========================================================================
    
    def test_08_cleanup_session(self):
        """TC08: 특정 세션 전체 삭제"""
        session_id = "test_session_8"
        DataSessionManager.store(session_id, "data1", "d1")
        DataSessionManager.store(session_id, "data2", "d2")
        DataSessionManager.store(session_id, "data3", "d3")
        
        count = DataSessionManager.cleanup(session_id)
        assert count == 3
        
        assert DataSessionManager.get(session_id, "data1") is None
        print("✅ TC08 통과: 세션 전체 삭제 (3개 항목)")
    
    def test_09_get_stats(self):
        """TC09: 통계 조회"""
        DataSessionManager.store("sess1", "d1", "data")
        DataSessionManager.store("sess1", "d2", "data")
        DataSessionManager.store("sess2", "d3", "data")
        
        stats = DataSessionManager.get_stats()
        assert stats["sessions"] == 2
        assert stats["total_items"] == 3
        print("✅ TC09 통과: 통계 - 2 세션, 3 항목")
    
    # =========================================================================
    # 10-12: 동시성 테스트
    # =========================================================================
    
    def test_10_concurrent_write(self):
        """TC10: 동시 쓰기 테스트"""
        results = []
        
        def write_data(i):
            DataSessionManager.store(f"sess_{i}", f"data_{i}", f"value_{i}")
            results.append(i)
        
        threads = [threading.Thread(target=write_data, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 10
        stats = DataSessionManager.get_stats()
        assert stats["total_items"] == 10
        print("✅ TC10 통과: 10개 스레드 동시 쓰기 성공")
    
    def test_11_concurrent_read_write(self):
        """TC11: 동시 읽기/쓰기 테스트"""
        session_id = "concurrent_session"
        DataSessionManager.store(session_id, "shared_data", "initial")
        
        read_results = []
        
        def read_data():
            for _ in range(5):
                result = DataSessionManager.get(session_id, "shared_data")
                if result:
                    read_results.append(result)
        
        def write_data():
            for i in range(5):
                DataSessionManager.store(session_id, "shared_data", f"value_{i}")
        
        threads = [
            threading.Thread(target=read_data),
            threading.Thread(target=write_data),
            threading.Thread(target=read_data),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 읽기가 실패하지 않아야 함
        assert len(read_results) > 0
        print(f"✅ TC11 통과: 동시 읽기/쓰기 - {len(read_results)}회 읽기 성공")
    
    def test_12_isolation_between_sessions(self):
        """TC12: 세션 간 데이터 격리"""
        DataSessionManager.store("session_A", "private_data", "A's data")
        DataSessionManager.store("session_B", "private_data", "B's data")
        
        assert DataSessionManager.get("session_A", "private_data") == "A's data"
        assert DataSessionManager.get("session_B", "private_data") == "B's data"
        
        # A 세션 삭제 후 B는 영향 없어야 함
        DataSessionManager.cleanup("session_A")
        assert DataSessionManager.get("session_B", "private_data") == "B's data"
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
        data = load_data("datasets/iris")
        assert data is not None
        assert len(data) == 150  # Iris는 150개 행
        print("✅ TC13 통과: Iris 데이터셋 로드 성공 (150행)")
    
    def test_14_load_nonexistent_dataset(self):
        """TC14: 존재하지 않는 데이터셋"""
        data = load_data("datasets/nonexistent_xyz")
        assert data is None
        print("✅ TC14 통과: 존재하지 않는 데이터셋 → None")
    
    def test_15_load_from_session(self):
        """TC15: 세션에서 데이터 로드"""
        session_id = "test_session_15"
        mock_data = {"type": "mock_table"}
        
        DataSessionManager.store(session_id, "sampler/test_sample", mock_data)
        
        result = load_data("sampler/test_sample", session_id=session_id)
        assert result == mock_data
        print("✅ TC15 통과: 세션에서 데이터 로드 성공")
    
    def test_16_load_fallback_to_legacy(self):
        """TC16: 세션에 없으면 레거시 저장소 사용"""
        # 세션에는 없고 레거시에만 있는 경우
        from app.widgets.data_sampler import _sampler_results
        _sampler_results["legacy_sample"] = {"data": "legacy_data"}
        
        result = load_data("sampler/legacy_sample", session_id="empty_session")
        assert result == "legacy_data"
        
        # 정리
        del _sampler_results["legacy_sample"]
        print("✅ TC16 통과: 레거시 저장소 폴백 성공")
    
    # =========================================================================
    # 17-20: 엣지 케이스 테스트
    # =========================================================================
    
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
    
    def test_19_session_id_none(self):
        """TC19: session_id가 None인 경우"""
        # session_id 없이 sampler 경로 → 레거시 사용
        from app.widgets.data_sampler import _sampler_results
        _sampler_results["no_session_sample"] = {"data": "no_session_data"}
        
        result = load_data("sampler/no_session_sample", session_id=None)
        assert result == "no_session_data"
        
        del _sampler_results["no_session_sample"]
        print("✅ TC19 통과: session_id=None → 레거시 사용")
    
    def test_20_load_multiple_datasets(self):
        """TC20: 여러 데이터셋 순차 로드"""
        datasets = ["datasets/iris", "datasets/housing", "datasets/zoo"]
        loaded = []
        
        for ds in datasets:
            data = load_data(ds)
            if data is not None:
                loaded.append(ds)
        
        # 최소 1개 이상 로드 성공
        assert len(loaded) >= 1
        print(f"✅ TC20 통과: {len(loaded)}개 데이터셋 로드 성공")


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "=" * 60)
    print("DataSessionManager & data_utils 테스트")
    print("=" * 60 + "\n")
    
    # DataSessionManager 테스트
    print("📦 DataSessionManager 테스트 (12개)")
    print("-" * 40)
    
    dsm_tests = TestDataSessionManager()
    dsm_tests.setup_method()
    dsm_tests.test_01_store_and_get()
    
    dsm_tests.setup_method()
    dsm_tests.test_02_get_nonexistent_session()
    
    dsm_tests.setup_method()
    dsm_tests.test_03_get_nonexistent_data()
    
    dsm_tests.setup_method()
    dsm_tests.test_04_store_with_metadata()
    
    dsm_tests.setup_method()
    dsm_tests.test_05_ttl_not_expired()
    
    dsm_tests.setup_method()
    dsm_tests.test_06_ttl_expired()
    
    dsm_tests.setup_method()
    dsm_tests.test_07_cleanup_expired()
    
    dsm_tests.setup_method()
    dsm_tests.test_08_cleanup_session()
    
    dsm_tests.setup_method()
    dsm_tests.test_09_get_stats()
    
    dsm_tests.setup_method()
    dsm_tests.test_10_concurrent_write()
    
    dsm_tests.setup_method()
    dsm_tests.test_11_concurrent_read_write()
    
    dsm_tests.setup_method()
    dsm_tests.test_12_isolation_between_sessions()
    
    # load_data 테스트
    print("\n📦 load_data() 테스트 (8개)")
    print("-" * 40)
    
    ld_tests = TestLoadData()
    
    if ORANGE_AVAILABLE:
        ld_tests.setup_method()
        ld_tests.test_13_load_builtin_dataset()
    else:
        print("⏭️  TC13 건너뜀: Orange3 미설치")
    
    ld_tests.setup_method()
    ld_tests.test_14_load_nonexistent_dataset()
    
    ld_tests.setup_method()
    ld_tests.test_15_load_from_session()
    
    ld_tests.setup_method()
    ld_tests.test_16_load_fallback_to_legacy()
    
    ld_tests.setup_method()
    ld_tests.test_17_empty_path()
    
    ld_tests.setup_method()
    ld_tests.test_18_invalid_path_format()
    
    ld_tests.setup_method()
    ld_tests.test_19_session_id_none()
    
    if ORANGE_AVAILABLE:
        ld_tests.setup_method()
        ld_tests.test_20_load_multiple_datasets()
    else:
        print("⏭️  TC20 건너뜀: Orange3 미설치")
    
    print("\n" + "=" * 60)
    print("🎉 모든 테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

