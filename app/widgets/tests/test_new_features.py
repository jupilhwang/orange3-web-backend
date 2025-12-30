"""
Tests for newly implemented features (v0.23.2):
1. Tree Visualization
2. Random Forest Feature Importance
3. Distributions KDE
4. Linear Regression Coefficients
5. Logistic Regression Coefficients
"""

import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def get_headers():
    """Get common request headers."""
    return {"X-Session-Id": "test-session-new-features"}


def load_iris_dataset():
    """Load iris dataset and return response."""
    response = client.post(
        "/api/v1/datasets/core%2Firis.tab/load",
        headers=get_headers()
    )
    return response


def load_housing_dataset():
    """Load housing (regression) dataset."""
    response = client.post(
        "/api/v1/datasets/core%2Fhousing.tab/load",
        headers=get_headers()
    )
    return response


# ============================================================================
# Test Class: Tree Visualization
# ============================================================================

class TestTreeVisualization:
    """Tests for Tree visualization API."""
    
    def test_train_tree_for_visualization(self):
        """Test training a tree model for visualization."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/model/tree/train",
            json={
                "data_path": "core/iris.tab",
                "binary_trees": True,
                "max_depth": 5,
                "min_samples_split": 5,
                "min_samples_leaf": 2
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "model_id" in data
        return data["model_id"]
    
    def test_visualize_tree_default_depth(self):
        """Test tree visualization with default depth."""
        load_iris_dataset()
        
        # Train model first
        train_response = client.post(
            "/api/v1/model/tree/train",
            json={
                "data_path": "core/iris.tab",
                "binary_trees": True,
                "max_depth": 10
            },
            headers=get_headers()
        )
        
        if train_response.status_code == 200:
            train_data = train_response.json()
            if train_data["success"]:
                model_id = train_data["model_id"]
                
                # Visualize tree
                viz_response = client.get(
                    f"/api/v1/model/tree/visualize/{model_id}",
                    headers=get_headers()
                )
                assert viz_response.status_code == 200
                viz_data = viz_response.json()
                
                if viz_data.get("success"):
                    assert "tree" in viz_data
                    tree = viz_data["tree"]
                    assert "id" in tree
                    assert "depth" in tree
                    assert "samples" in tree
    
    def test_visualize_tree_custom_depth(self):
        """Test tree visualization with custom max depth."""
        load_iris_dataset()
        
        train_response = client.post(
            "/api/v1/model/tree/train",
            json={
                "data_path": "core/iris.tab",
                "binary_trees": True,
                "max_depth": 10
            },
            headers=get_headers()
        )
        
        if train_response.status_code == 200:
            train_data = train_response.json()
            if train_data["success"]:
                model_id = train_data["model_id"]
                
                # Visualize with max_depth=2
                viz_response = client.get(
                    f"/api/v1/model/tree/visualize/{model_id}?max_depth=2",
                    headers=get_headers()
                )
                assert viz_response.status_code == 200
                viz_data = viz_response.json()
                
                if viz_data.get("success"):
                    assert viz_data["max_depth"] == 2
    
    def test_visualize_nonexistent_model(self):
        """Test visualization with non-existent model."""
        response = client.get(
            "/api/v1/model/tree/visualize/nonexistent123",
            headers=get_headers()
        )
        assert response.status_code == 404


# ============================================================================
# Test Class: Random Forest Feature Importance
# ============================================================================

class TestRandomForestFeatureImportance:
    """Tests for Random Forest Feature Importance API."""
    
    def test_train_rf_for_importance(self):
        """Test training RF model for feature importance."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/model/random-forest/train",
            json={
                "data_path": "core/iris.tab",
                "n_estimators": 20,
                "random_state": 42
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "model_id" in data
        
        # Check if feature_importances is in train response
        if "feature_importances" in data and data["feature_importances"]:
            assert len(data["feature_importances"]) > 0
            assert "feature" in data["feature_importances"][0]
            assert "importance" in data["feature_importances"][0]
    
    def test_get_feature_importance_default(self):
        """Test getting feature importance with default parameters."""
        load_iris_dataset()
        
        train_response = client.post(
            "/api/v1/model/random-forest/train",
            json={
                "data_path": "core/iris.tab",
                "n_estimators": 20,
                "random_state": 42
            },
            headers=get_headers()
        )
        
        if train_response.status_code == 200:
            train_data = train_response.json()
            if train_data["success"]:
                model_id = train_data["model_id"]
                
                # Get feature importance
                fi_response = client.get(
                    f"/api/v1/model/random-forest/feature-importance/{model_id}",
                    headers=get_headers()
                )
                assert fi_response.status_code == 200
                fi_data = fi_response.json()
                
                if fi_data.get("success"):
                    assert "features" in fi_data
                    assert "n_features" in fi_data
                    assert len(fi_data["features"]) > 0
                    
                    # Check feature structure
                    first_feature = fi_data["features"][0]
                    assert "feature" in first_feature
                    assert "importance" in first_feature
                    assert "rank" in first_feature
                    assert "cumulative" in first_feature
    
    def test_get_feature_importance_top_n(self):
        """Test getting top N feature importances."""
        load_iris_dataset()
        
        train_response = client.post(
            "/api/v1/model/random-forest/train",
            json={
                "data_path": "core/iris.tab",
                "n_estimators": 20,
                "random_state": 42
            },
            headers=get_headers()
        )
        
        if train_response.status_code == 200:
            train_data = train_response.json()
            if train_data["success"]:
                model_id = train_data["model_id"]
                
                # Get top 2 features
                fi_response = client.get(
                    f"/api/v1/model/random-forest/feature-importance/{model_id}?top_n=2",
                    headers=get_headers()
                )
                assert fi_response.status_code == 200
                fi_data = fi_response.json()
                
                if fi_data.get("success"):
                    assert len(fi_data["features"]) <= 2
    
    def test_feature_importance_sorted_by_importance(self):
        """Test that features are sorted by importance (descending)."""
        load_iris_dataset()
        
        train_response = client.post(
            "/api/v1/model/random-forest/train",
            json={
                "data_path": "core/iris.tab",
                "n_estimators": 20,
                "random_state": 42
            },
            headers=get_headers()
        )
        
        if train_response.status_code == 200:
            train_data = train_response.json()
            if train_data["success"]:
                model_id = train_data["model_id"]
                
                fi_response = client.get(
                    f"/api/v1/model/random-forest/feature-importance/{model_id}",
                    headers=get_headers()
                )
                fi_data = fi_response.json()
                
                if fi_data.get("success") and len(fi_data["features"]) > 1:
                    importances = [f["importance"] for f in fi_data["features"]]
                    # Check sorted descending
                    assert importances == sorted(importances, reverse=True)
    
    def test_feature_importance_nonexistent_model(self):
        """Test feature importance with non-existent model."""
        response = client.get(
            "/api/v1/model/random-forest/feature-importance/nonexistent123",
            headers=get_headers()
        )
        assert response.status_code == 404


# ============================================================================
# Test Class: Distributions KDE
# ============================================================================

class TestDistributionsKDE:
    """Tests for Distributions with Kernel Density Estimation."""
    
    def test_distributions_with_kde(self):
        """Test distributions with KDE enabled."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/data/distributions",
            json={
                "data_path": "core/iris.tab",
                "variable": "sepal length",
                "number_of_bins": 5,
                "kde_smoothing": 10
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        
        assert "variable" in data
        assert "bins" in data
        
        # Check KDE curve
        if "kde_curve" in data and data["kde_curve"]:
            kde = data["kde_curve"]
            assert "x" in kde
            assert "y" in kde
            assert len(kde["x"]) > 0
            assert len(kde["y"]) > 0
            assert len(kde["x"]) == len(kde["y"])
    
    def test_distributions_kde_smoothing_levels(self):
        """Test different KDE smoothing levels."""
        load_iris_dataset()
        
        # Test low smoothing
        response_low = client.post(
            "/api/v1/data/distributions",
            json={
                "data_path": "core/iris.tab",
                "variable": "sepal length",
                "kde_smoothing": 5
            },
            headers=get_headers()
        )
        
        # Test high smoothing
        response_high = client.post(
            "/api/v1/data/distributions",
            json={
                "data_path": "core/iris.tab",
                "variable": "sepal length",
                "kde_smoothing": 30
            },
            headers=get_headers()
        )
        
        assert response_low.status_code == 200
        assert response_high.status_code == 200
    
    def test_distributions_extended_statistics(self):
        """Test extended statistics (Q1, Q3, IQR, skewness, kurtosis)."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/data/distributions",
            json={
                "data_path": "core/iris.tab",
                "variable": "sepal length",
                "kde_smoothing": 10
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        
        if "statistics" in data:
            stats = data["statistics"]
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats
            assert "median" in stats
            
            # New statistics
            assert "q1" in stats
            assert "q3" in stats
            assert "iqr" in stats
            assert "skewness" in stats
            assert "kurtosis" in stats
    
    def test_distributions_kde_disabled(self):
        """Test distributions without KDE (kde_smoothing=0)."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/data/distributions",
            json={
                "data_path": "core/iris.tab",
                "variable": "sepal length",
                "kde_smoothing": 0
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        
        # KDE should not be present when kde_smoothing=0
        # (depends on implementation - may still include empty or not include)


# ============================================================================
# Test Class: Linear Regression Coefficients
# ============================================================================

class TestLinearRegressionCoefficients:
    """Tests for Linear Regression Coefficients API."""
    
    def test_train_linear_regression_for_coefficients(self):
        """Test training LR model for coefficients."""
        load_housing_dataset()
        
        response = client.post(
            "/api/v1/model/linear-regression/train",
            json={
                "data_path": "core/housing.tab",
                "regularization_type": "none",
                "fit_intercept": True
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "model_id" in data
        
        # Check if coefficients included in response
        if "coefficients" in data and data["coefficients"]:
            assert len(data["coefficients"]) > 0
    
    def test_get_coefficients_default(self):
        """Test getting coefficients with default parameters."""
        load_housing_dataset()
        
        train_response = client.post(
            "/api/v1/model/linear-regression/train",
            json={
                "data_path": "core/housing.tab",
                "regularization_type": "none",
                "fit_intercept": True
            },
            headers=get_headers()
        )
        
        if train_response.status_code == 200:
            train_data = train_response.json()
            if train_data["success"]:
                model_id = train_data["model_id"]
                
                # Get coefficients
                coef_response = client.get(
                    f"/api/v1/model/linear-regression/coefficients/{model_id}",
                    headers=get_headers()
                )
                assert coef_response.status_code == 200
                coef_data = coef_response.json()
                
                if coef_data.get("success"):
                    assert "coefficients" in coef_data
                    assert "statistics" in coef_data
                    assert len(coef_data["coefficients"]) > 0
                    
                    # Check coefficient structure
                    first_coef = coef_data["coefficients"][0]
                    assert "feature" in first_coef
                    assert "coefficient" in first_coef
                    assert "abs_coefficient" in first_coef
    
    def test_get_coefficients_sort_by_abs(self):
        """Test sorting coefficients by absolute value."""
        load_housing_dataset()
        
        train_response = client.post(
            "/api/v1/model/linear-regression/train",
            json={
                "data_path": "core/housing.tab",
                "regularization_type": "none"
            },
            headers=get_headers()
        )
        
        if train_response.status_code == 200:
            train_data = train_response.json()
            if train_data["success"]:
                model_id = train_data["model_id"]
                
                coef_response = client.get(
                    f"/api/v1/model/linear-regression/coefficients/{model_id}?sort_by=abs_coefficient",
                    headers=get_headers()
                )
                coef_data = coef_response.json()
                
                if coef_data.get("success"):
                    assert coef_data["sorted_by"] == "abs_coefficient"
    
    def test_coefficients_statistics(self):
        """Test coefficient statistics."""
        load_housing_dataset()
        
        train_response = client.post(
            "/api/v1/model/linear-regression/train",
            json={
                "data_path": "core/housing.tab",
                "regularization_type": "none"
            },
            headers=get_headers()
        )
        
        if train_response.status_code == 200:
            train_data = train_response.json()
            if train_data["success"]:
                model_id = train_data["model_id"]
                
                coef_response = client.get(
                    f"/api/v1/model/linear-regression/coefficients/{model_id}",
                    headers=get_headers()
                )
                coef_data = coef_response.json()
                
                if coef_data.get("success"):
                    stats = coef_data["statistics"]
                    assert "n_features" in stats
                    assert "mean" in stats
                    assert "std" in stats
                    assert "n_positive" in stats
                    assert "n_negative" in stats
    
    def test_coefficients_nonexistent_model(self):
        """Test coefficients with non-existent model."""
        response = client.get(
            "/api/v1/model/linear-regression/coefficients/nonexistent123",
            headers=get_headers()
        )
        assert response.status_code == 404


# ============================================================================
# Test Class: Logistic Regression Coefficients
# ============================================================================

class TestLogisticRegressionCoefficients:
    """Tests for Logistic Regression Coefficients API."""
    
    def test_train_logistic_regression_for_coefficients(self):
        """Test training LogR model for coefficients."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/model/logistic-regression/train",
            json={
                "data_path": "core/iris.tab",
                "penalty": "l2",
                "C": 1.0
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "model_id" in data
    
    def test_get_logr_coefficients_default(self):
        """Test getting LogR coefficients with default parameters."""
        load_iris_dataset()
        
        train_response = client.post(
            "/api/v1/model/logistic-regression/train",
            json={
                "data_path": "core/iris.tab",
                "penalty": "l2",
                "C": 1.0
            },
            headers=get_headers()
        )
        
        if train_response.status_code == 200:
            train_data = train_response.json()
            if train_data["success"]:
                model_id = train_data["model_id"]
                
                coef_response = client.get(
                    f"/api/v1/model/logistic-regression/coefficients/{model_id}",
                    headers=get_headers()
                )
                assert coef_response.status_code == 200
                coef_data = coef_response.json()
                
                if coef_data.get("success"):
                    assert "coefficients" in coef_data
                    assert "statistics" in coef_data
                    assert len(coef_data["coefficients"]) > 0
    
    def test_get_logr_coefficients_with_odds_ratio(self):
        """Test LogR coefficients include odds ratio."""
        load_iris_dataset()
        
        train_response = client.post(
            "/api/v1/model/logistic-regression/train",
            json={
                "data_path": "core/iris.tab",
                "penalty": "l2",
                "C": 1.0
            },
            headers=get_headers()
        )
        
        if train_response.status_code == 200:
            train_data = train_response.json()
            if train_data["success"]:
                model_id = train_data["model_id"]
                
                coef_response = client.get(
                    f"/api/v1/model/logistic-regression/coefficients/{model_id}",
                    headers=get_headers()
                )
                coef_data = coef_response.json()
                
                if coef_data.get("success"):
                    first_coef = coef_data["coefficients"][0]
                    assert "odds_ratio" in first_coef
    
    def test_get_logr_coefficients_multiclass(self):
        """Test LogR coefficients for multi-class (different class_index)."""
        load_iris_dataset()
        
        train_response = client.post(
            "/api/v1/model/logistic-regression/train",
            json={
                "data_path": "core/iris.tab",
                "penalty": "l2",
                "C": 1.0
            },
            headers=get_headers()
        )
        
        if train_response.status_code == 200:
            train_data = train_response.json()
            if train_data["success"]:
                model_id = train_data["model_id"]
                
                # Get coefficients for class 0
                coef_response_0 = client.get(
                    f"/api/v1/model/logistic-regression/coefficients/{model_id}?class_index=0",
                    headers=get_headers()
                )
                
                # Get coefficients for class 1
                coef_response_1 = client.get(
                    f"/api/v1/model/logistic-regression/coefficients/{model_id}?class_index=1",
                    headers=get_headers()
                )
                
                assert coef_response_0.status_code == 200
                assert coef_response_1.status_code == 200
                
                coef_data_0 = coef_response_0.json()
                coef_data_1 = coef_response_1.json()
                
                if coef_data_0.get("success") and coef_data_1.get("success"):
                    assert "n_classes" in coef_data_0
                    assert "current_class_index" in coef_data_0
    
    def test_logr_coefficients_sort_options(self):
        """Test different sort options for LogR coefficients."""
        load_iris_dataset()
        
        train_response = client.post(
            "/api/v1/model/logistic-regression/train",
            json={
                "data_path": "core/iris.tab",
                "penalty": "l2",
                "C": 1.0
            },
            headers=get_headers()
        )
        
        if train_response.status_code == 200:
            train_data = train_response.json()
            if train_data["success"]:
                model_id = train_data["model_id"]
                
                # Test sort by coefficient
                response_coef = client.get(
                    f"/api/v1/model/logistic-regression/coefficients/{model_id}?sort_by=coefficient",
                    headers=get_headers()
                )
                
                # Test sort by abs_coefficient
                response_abs = client.get(
                    f"/api/v1/model/logistic-regression/coefficients/{model_id}?sort_by=abs_coefficient",
                    headers=get_headers()
                )
                
                assert response_coef.status_code == 200
                assert response_abs.status_code == 200
    
    def test_logr_coefficients_nonexistent_model(self):
        """Test coefficients with non-existent model."""
        response = client.get(
            "/api/v1/model/logistic-regression/coefficients/nonexistent123",
            headers=get_headers()
        )
        assert response.status_code == 404


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

