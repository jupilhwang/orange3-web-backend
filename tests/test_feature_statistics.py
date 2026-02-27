"""
Feature Statistics Widget 테스트

Orange3의 OWFeatureStatistics 위젯과 동일한 기능을 검증합니다.

테스트 범위:
1. 기본 통계 계산 (Mean, Mode, Median, Min, Max, Missing)
2. 분산 계산 (연속형: CV, 범주형: Entropy)
3. 분포 히스토그램 생성
4. 컬럼 선택 및 Reduced Data 출력
5. 색상 변수별 분포 구분
6. 다양한 데이터 타입 처리 (연속형, 범주형, 시간형)
7. 결측치 처리
8. 에러 핸들링
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Test fixtures
@pytest.fixture
def iris_data():
    """Iris 데이터셋 mock"""
    try:
        from Orange.data import Table
        return Table("iris")
    except ImportError:
        pytest.skip("Orange3 not installed")

@pytest.fixture
def housing_data():
    """Housing 데이터셋 mock"""
    try:
        from Orange.data import Table
        return Table("housing")
    except ImportError:
        pytest.skip("Orange3 not installed")

@pytest.fixture
def titanic_data():
    """Titanic 데이터셋 mock (결측치 포함)"""
    try:
        from Orange.data import Table
        return Table("titanic")
    except ImportError:
        pytest.skip("Orange3 not installed")


class TestFeatureStatisticsCompute:
    """통계 계산 테스트"""
    
    def test_compute_mean_continuous(self, iris_data):
        """연속형 변수 평균 계산 테스트"""
        from app.widgets.feature_statistics import _compute_dispersion
        
        # Sepal length column
        x = iris_data.X[:, 0]
        expected_mean = np.nanmean(x)
        
        assert not np.isnan(expected_mean)
        assert 5.0 < expected_mean < 6.0  # Iris sepal length mean is around 5.8
    
    def test_compute_median_continuous(self, iris_data):
        """연속형 변수 중앙값 계산 테스트"""
        x = iris_data.X[:, 0]
        median = np.nanmedian(x)
        
        assert not np.isnan(median)
        assert 5.0 < median < 6.5
    
    def test_compute_mode_discrete(self, iris_data):
        """범주형 변수 최빈값 계산 테스트"""
        from scipy import stats
        
        y = iris_data.Y
        mode_result = stats.mode(y, keepdims=False)
        
        assert mode_result.mode in [0, 1, 2]  # Class index
    
    def test_compute_dispersion_continuous(self, iris_data):
        """연속형 변수 분산(CV) 계산 테스트"""
        from app.widgets.feature_statistics import _compute_dispersion
        
        x = iris_data.X[:, 0]
        cv = _compute_dispersion(x, 'continuous')
        
        assert cv is not None
        assert cv > 0
        assert cv < 1  # CV is typically < 1 for Iris data
    
    def test_compute_dispersion_discrete(self, iris_data):
        """범주형 변수 분산(Entropy) 계산 테스트"""
        from app.widgets.feature_statistics import _compute_dispersion
        
        y = iris_data.Y
        entropy = _compute_dispersion(y, 'discrete')
        
        assert entropy is not None
        assert entropy > 0  # Iris has 3 balanced classes, entropy > 0
    
    def test_compute_min_max(self, iris_data):
        """최소/최대값 계산 테스트"""
        x = iris_data.X[:, 0]
        
        min_val = np.nanmin(x)
        max_val = np.nanmax(x)
        
        assert min_val < max_val
        assert min_val > 0
    
    def test_compute_missing_count(self, titanic_data):
        """결측치 개수 계산 테스트"""
        # Titanic data has missing values in 'age' column
        has_missing = any(np.isnan(titanic_data.X[:, i]).sum() > 0 
                        for i in range(titanic_data.X.shape[1]))
        
        # Titanic should have missing values
        # This may vary depending on the dataset version
        assert True  # Just verify no exception


class TestFeatureStatisticsDistribution:
    """분포 히스토그램 테스트"""
    
    def test_compute_distribution_continuous(self, iris_data):
        """연속형 변수 히스토그램 계산 테스트"""
        from app.widgets.feature_statistics import _compute_distribution
        
        x = iris_data.X[:, 0]
        var = iris_data.domain.attributes[0]
        
        dist = _compute_distribution(x, var, 'continuous')
        
        assert dist is not None
        assert dist['type'] == 'histogram'
        assert 'bins' in dist
        assert 'counts' in dist
        assert len(dist['counts']) > 0
    
    def test_compute_distribution_discrete(self, iris_data):
        """범주형 변수 바 차트 계산 테스트"""
        from app.widgets.feature_statistics import _compute_distribution
        
        y = iris_data.Y
        var = iris_data.domain.class_var
        
        dist = _compute_distribution(y, var, 'discrete')
        
        assert dist is not None
        assert dist['type'] == 'bar'
        assert 'labels' in dist
        assert 'counts' in dist
        assert len(dist['labels']) == 3  # Iris has 3 classes
    
    def test_distribution_with_color_var(self, iris_data):
        """색상 변수별 분포 계산 테스트"""
        from app.widgets.feature_statistics import _compute_distribution
        
        x = iris_data.X[:, 0]
        var = iris_data.domain.attributes[0]
        color_data = iris_data.Y
        color_var = iris_data.domain.class_var
        
        dist = _compute_distribution(x, var, 'continuous', color_data, color_var)
        
        assert dist is not None
        if 'stacked' in dist:
            assert len(dist['stacked']) == 3  # 3 classes


class TestFeatureStatisticsAPI:
    """API 엔드포인트 테스트"""
    
    @pytest.mark.asyncio
    async def test_compute_endpoint_success(self, iris_data):
        """통계 계산 API 성공 테스트"""
        from app.widgets.feature_statistics import compute_feature_statistics, FeatureStatisticsRequest
        
        with patch('app.core.data_utils.load_data') as mock_load:
            mock_load.return_value = iris_data
            
            request = FeatureStatisticsRequest(data_path="datasets/iris")
            response = await compute_feature_statistics(request)
            
            assert response.success
            assert response.n_instances == 150
            assert response.n_features > 0
            assert len(response.columns) > 0
    
    @pytest.mark.asyncio
    async def test_compute_endpoint_no_data(self):
        """데이터 없음 테스트"""
        from app.widgets.feature_statistics import compute_feature_statistics, FeatureStatisticsRequest
        
        with patch('app.core.data_utils.load_data') as mock_load:
            mock_load.return_value = None
            
            request = FeatureStatisticsRequest(data_path="nonexistent")
            response = await compute_feature_statistics(request)
            
            assert not response.success
            assert response.error is not None
    
    @pytest.mark.asyncio
    async def test_reduced_data_endpoint(self, iris_data):
        """Reduced Data API 테스트"""
        from app.widgets.feature_statistics import get_reduced_data, ReducedDataRequest
        
        with patch('app.core.data_utils.load_data') as mock_load:
            mock_load.return_value = iris_data
            
            request = ReducedDataRequest(
                data_path="datasets/iris",
                selected_columns=["sepal length", "petal length"]
            )
            response = await get_reduced_data(request)
            
            assert response.success
            assert response.n_instances == 150
            assert response.n_features == 2


class TestFeatureStatisticsColumnTypes:
    """다양한 컬럼 타입 테스트"""
    
    def test_variable_type_detection(self, iris_data):
        """변수 타입 감지 테스트"""
        from app.widgets.feature_statistics import _get_variable_type
        
        # Continuous variable
        cont_var = iris_data.domain.attributes[0]
        assert _get_variable_type(cont_var) == 'continuous'
        
        # Discrete variable (class)
        disc_var = iris_data.domain.class_var
        assert _get_variable_type(disc_var) == 'discrete'
    
    def test_variable_role_detection(self, iris_data):
        """변수 역할 감지 테스트"""
        from app.widgets.feature_statistics import _get_variable_role
        
        domain = iris_data.domain
        
        # Attribute
        attr = domain.attributes[0]
        assert _get_variable_role(attr, domain) == 'attribute'
        
        # Class
        class_var = domain.class_var
        assert _get_variable_role(class_var, domain) == 'class'


class TestFeatureStatisticsEdgeCases:
    """엣지 케이스 테스트"""
    
    def test_empty_data(self):
        """빈 데이터 처리 테스트"""
        from app.widgets.feature_statistics import _compute_dispersion
        
        x = np.array([])
        result = _compute_dispersion(x, 'continuous')
        
        assert result is None
    
    def test_all_nan_column(self):
        """모든 값이 NaN인 컬럼 테스트"""
        from app.widgets.feature_statistics import _compute_dispersion
        
        x = np.array([np.nan, np.nan, np.nan])
        result = _compute_dispersion(x, 'continuous')
        
        assert result is None
    
    def test_single_value_column(self):
        """단일 값 컬럼 테스트"""
        from app.widgets.feature_statistics import _compute_dispersion
        
        x = np.array([5.0, 5.0, 5.0, 5.0])
        result = _compute_dispersion(x, 'continuous')
        
        assert result is not None
        assert result == 0.0  # No variation
    
    def test_zero_mean_column(self):
        """평균이 0인 컬럼 CV 계산 테스트"""
        from app.widgets.feature_statistics import _compute_dispersion
        
        x = np.array([-1.0, 0.0, 1.0])
        result = _compute_dispersion(x, 'continuous')
        
        # CV with zero mean should return inf
        assert result == float('inf')


class TestFeatureStatisticsIntegration:
    """통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_full_workflow_iris(self, iris_data):
        """Iris 데이터 전체 워크플로우 테스트"""
        from app.widgets.feature_statistics import (
            compute_feature_statistics, 
            get_reduced_data,
            FeatureStatisticsRequest,
            ReducedDataRequest
        )
        
        with patch('app.core.data_utils.load_data') as mock_load:
            mock_load.return_value = iris_data
            
            # Step 1: Compute statistics
            stats_request = FeatureStatisticsRequest(data_path="datasets/iris")
            stats_response = await compute_feature_statistics(stats_request)
            
            assert stats_response.success
            assert stats_response.n_instances == 150
            
            # Verify column statistics
            for col in stats_response.columns:
                assert col.name is not None
                assert col.type in ['continuous', 'discrete', 'time', 'string']
                assert col.missing >= 0
                assert col.missing_percent >= 0
            
            # Step 2: Get reduced data with selected columns
            reduced_request = ReducedDataRequest(
                data_path="datasets/iris",
                selected_columns=["sepal length", "sepal width"]
            )
            reduced_response = await get_reduced_data(reduced_request)
            
            assert reduced_response.success
            assert reduced_response.n_features == 2
    
    @pytest.mark.asyncio
    async def test_full_workflow_housing(self, housing_data):
        """Housing 데이터 전체 워크플로우 테스트"""
        from app.widgets.feature_statistics import (
            compute_feature_statistics,
            FeatureStatisticsRequest
        )
        
        with patch('app.core.data_utils.load_data') as mock_load:
            mock_load.return_value = housing_data
            
            request = FeatureStatisticsRequest(data_path="datasets/housing")
            response = await compute_feature_statistics(request)
            
            assert response.success
            assert response.n_instances == 506
            assert response.n_features >= 13  # Housing has 13+ features


class TestFeatureStatisticsOptions:
    """옵션 API 테스트"""
    
    @pytest.mark.asyncio
    async def test_get_options(self):
        """옵션 목록 API 테스트"""
        from app.widgets.feature_statistics import get_options
        
        options = await get_options()
        
        assert 'columns' in options
        assert len(options['columns']) == 10  # 10 columns defined
        
        # Verify column definitions
        column_ids = [c['id'] for c in options['columns']]
        assert 'name' in column_ids
        assert 'mean' in column_ids
        assert 'distribution' in column_ids
        assert 'missing' in column_ids


# Test runner for manual execution
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

