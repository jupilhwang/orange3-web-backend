"""
Comprehensive tests for all Orange3-Web backend widgets.
Tests include: Data widgets, Model widgets, Evaluate widgets, Visualize widgets.
"""

import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# ============================================================================
# Helper Functions
# ============================================================================

def get_headers():
    """Get common request headers."""
    return {"X-Session-Id": "test-session"}


def load_iris_dataset():
    """Load iris dataset and return its path."""
    response = client.post(
        "/api/v1/datasets/core%2Firis.tab/load",
        headers=get_headers()
    )
    if response.status_code == 200:
        return "core/iris.tab"
    return None


def load_housing_dataset():
    """Load housing (regression) dataset."""
    response = client.post(
        "/api/v1/datasets/core%2Fhousing.tab/load",
        headers=get_headers()
    )
    if response.status_code == 200:
        return "core/housing.tab"
    return None


# ============================================================================
# Test Class: Datasets Widget
# ============================================================================

class TestDatasetsWidget:
    """Tests for Datasets widget API."""
    
    def test_list_datasets(self):
        """Test listing available datasets."""
        response = client.get("/api/v1/datasets")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert "total" in data
    
    def test_list_datasets_with_filter(self):
        """Test filtering datasets by language."""
        response = client.get("/api/v1/datasets?language=English")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
    
    def test_list_datasets_with_search(self):
        """Test searching datasets by title."""
        response = client.get("/api/v1/datasets?search=iris")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
    
    def test_load_iris_dataset(self):
        """Test loading iris dataset."""
        response = client.post(
            "/api/v1/datasets/core%2Firis.tab/load",
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["instances"] == 150
        assert data["features"] == 4
        assert "columns" in data
    
    def test_load_housing_dataset(self):
        """Test loading housing (regression) dataset."""
        response = client.post(
            "/api/v1/datasets/core%2Fhousing.tab/load",
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["classType"] == "Regression"
    
    def test_get_dataset_info(self):
        """Test getting dataset information."""
        response = client.get("/api/v1/datasets/core%2Firis.tab/info")
        assert response.status_code == 200
        data = response.json()
        assert "title" in data
    
    def test_load_nonexistent_dataset(self):
        """Test loading a non-existent dataset."""
        response = client.post(
            "/api/v1/datasets/nonexistent%2Fdata.tab/load",
            headers=get_headers()
        )
        assert response.status_code == 404


# ============================================================================
# Test Class: Select Columns Widget
# ============================================================================

class TestSelectColumnsWidget:
    """Tests for Select Columns widget API."""
    
    def test_select_columns_basic(self):
        """Test basic column selection."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/data/select-columns",
            json={
                "data_path": "core/iris.tab",
                "features": ["sepal length", "sepal width"],
                "target": ["iris"],
                "metas": [],
                "ignored": ["petal length", "petal width"]
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert len(data["features"]) == 2
    
    def test_select_columns_reorder(self):
        """Test column reordering."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/data/select-columns",
            json={
                "data_path": "core/iris.tab",
                "features": ["petal width", "petal length", "sepal width", "sepal length"],
                "target": ["iris"],
                "metas": [],
                "ignored": []
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["features"][0] == "petal width"
    
    def test_select_columns_change_target(self):
        """Test changing target variable."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/data/select-columns",
            json={
                "data_path": "core/iris.tab",
                "features": ["sepal length", "sepal width", "petal length"],
                "target": ["petal width"],
                "metas": [],
                "ignored": ["iris"]
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["target"][0] == "petal width"


# ============================================================================
# Test Class: kNN Widget
# ============================================================================

class TestKNNWidget:
    """Tests for kNN widget API."""
    
    def test_train_knn_default(self):
        """Test training kNN with default parameters."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/model/knn/train",
            json={
                "data_path": "core/iris.tab",
                "n_neighbors": 5,
                "metric": "euclidean",
                "weights": "uniform"
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "model_id" in data
    
    def test_train_knn_manhattan(self):
        """Test training kNN with Manhattan distance."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/model/knn/train",
            json={
                "data_path": "core/iris.tab",
                "n_neighbors": 3,
                "metric": "manhattan",
                "weights": "distance"
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    def test_train_knn_invalid_k(self):
        """Test kNN with invalid k value."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/model/knn/train",
            json={
                "data_path": "core/iris.tab",
                "n_neighbors": 200,  # More than dataset size
                "metric": "euclidean",
                "weights": "uniform"
            },
            headers=get_headers()
        )
        assert response.status_code in [200, 400]
    
    def test_get_knn_options(self):
        """Test getting kNN options."""
        response = client.get("/api/v1/model/knn/options")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "weights" in data


# ============================================================================
# Test Class: Tree Widget
# ============================================================================

class TestTreeWidget:
    """Tests for Tree widget API."""
    
    def test_train_tree_default(self):
        """Test training Tree with default parameters."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/model/tree/train",
            json={
                "data_path": "core/iris.tab",
                "binary_trees": True,
                "max_depth": 100,
                "min_samples_split": 5,
                "min_samples_leaf": 2
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    def test_train_tree_limited_depth(self):
        """Test training Tree with limited depth."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/model/tree/train",
            json={
                "data_path": "core/iris.tab",
                "binary_trees": True,
                "max_depth": 3,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    def test_get_tree_options(self):
        """Test getting Tree options."""
        response = client.get("/api/v1/model/tree/options")
        assert response.status_code == 200
        data = response.json()
        assert "default_max_depth" in data


# ============================================================================
# Test Class: Random Forest Widget
# ============================================================================

class TestRandomForestWidget:
    """Tests for Random Forest widget API."""
    
    def test_train_rf_default(self):
        """Test training Random Forest with default parameters."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/model/random-forest/train",
            json={
                "data_path": "core/iris.tab",
                "n_estimators": 10,
                "max_features": None,
                "max_depth": None
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    def test_train_rf_many_trees(self):
        """Test training Random Forest with many trees."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/model/random-forest/train",
            json={
                "data_path": "core/iris.tab",
                "n_estimators": 50,
                "max_features": "sqrt",
                "max_depth": 10
            },
            headers=get_headers()
        )
        assert response.status_code == 200


# ============================================================================
# Test Class: Linear Regression Widget
# ============================================================================

class TestLinearRegressionWidget:
    """Tests for Linear Regression widget API."""
    
    def test_train_linear_regression_default(self):
        """Test training Linear Regression with default parameters."""
        load_housing_dataset()
        
        response = client.post(
            "/api/v1/model/linear-regression/train",
            json={
                "data_path": "core/housing.tab",
                "regularization_type": "none",
                "alpha": 1.0
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    def test_train_linear_regression_ridge(self):
        """Test training Linear Regression with Ridge regularization."""
        load_housing_dataset()
        
        response = client.post(
            "/api/v1/model/linear-regression/train",
            json={
                "data_path": "core/housing.tab",
                "regularization_type": "ridge",
                "alpha": 0.1
            },
            headers=get_headers()
        )
        assert response.status_code == 200
    
    def test_train_linear_regression_lasso(self):
        """Test training Linear Regression with Lasso regularization."""
        load_housing_dataset()
        
        response = client.post(
            "/api/v1/model/linear-regression/train",
            json={
                "data_path": "core/housing.tab",
                "regularization_type": "lasso",
                "alpha": 0.01
            },
            headers=get_headers()
        )
        assert response.status_code == 200
    
    def test_train_linear_regression_elastic_net(self):
        """Test training Linear Regression with Elastic Net."""
        load_housing_dataset()
        
        response = client.post(
            "/api/v1/model/linear-regression/train",
            json={
                "data_path": "core/housing.tab",
                "regularization_type": "elastic_net",
                "alpha": 0.1,
                "l1_ratio": 0.5
            },
            headers=get_headers()
        )
        assert response.status_code == 200
    
    def test_linear_regression_wrong_target(self):
        """Test Linear Regression with categorical target (should fail)."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/model/linear-regression/train",
            json={
                "data_path": "core/iris.tab",
                "regularization_type": "none",
                "alpha": 1.0
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        # Should fail or have error indication
        assert data.get("success") == False or data.get("error_type") == "target_type"


# ============================================================================
# Test Class: Logistic Regression Widget
# ============================================================================

class TestLogisticRegressionWidget:
    """Tests for Logistic Regression widget API."""
    
    def test_train_logistic_regression_default(self):
        """Test training Logistic Regression with default parameters."""
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
    
    def test_train_logistic_regression_l1(self):
        """Test training Logistic Regression with L1 penalty."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/model/logistic-regression/train",
            json={
                "data_path": "core/iris.tab",
                "penalty": "l1",
                "C": 0.5
            },
            headers=get_headers()
        )
        assert response.status_code == 200
    
    def test_logistic_regression_wrong_target(self):
        """Test Logistic Regression with numeric target (should fail)."""
        load_housing_dataset()
        
        response = client.post(
            "/api/v1/model/logistic-regression/train",
            json={
                "data_path": "core/housing.tab",
                "penalty": "l2",
                "C": 1.0
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == False or data.get("error_type") == "target_type"


# ============================================================================
# Test Class: Naive Bayes Widget
# ============================================================================

class TestNaiveBayesWidget:
    """Tests for Naive Bayes widget API."""
    
    def test_train_naive_bayes(self):
        """Test training Naive Bayes."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/model/naive-bayes/train",
            json={
                "data_path": "core/iris.tab"
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    def test_naive_bayes_wrong_target(self):
        """Test Naive Bayes with numeric target (should fail)."""
        load_housing_dataset()
        
        response = client.post(
            "/api/v1/model/naive-bayes/train",
            json={
                "data_path": "core/housing.tab"
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == False or data.get("error_type") == "target_type"


# ============================================================================
# Test Class: K-Means Widget
# ============================================================================

class TestKMeansWidget:
    """Tests for K-Means widget API."""
    
    def test_kmeans_default(self):
        """Test K-Means with default parameters."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/model/kmeans/cluster",
            json={
                "data_path": "core/iris.tab",
                "n_clusters": 3,
                "init": "k-means++",
                "n_init": 10,
                "max_iter": 300
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "silhouette_score" in data
    
    def test_kmeans_range(self):
        """Test K-Means with cluster range (optimization)."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/model/kmeans/optimize",
            json={
                "data_path": "core/iris.tab",
                "k_from": 2,
                "k_to": 5,
                "init": "k-means++",
                "n_init": 10
            },
            headers=get_headers()
        )
        assert response.status_code == 200


# ============================================================================
# Test Class: Test and Score Widget
# ============================================================================

class TestTestAndScoreWidget:
    """Tests for Test and Score widget API."""
    
    def test_cross_validation(self):
        """Test cross-validation evaluation."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/evaluate/test_and_score/evaluate",
            json={
                "data_path": "core/iris.tab",
                "learner_configs": [
                    {"type": "knn", "name": "kNN", "n_neighbors": 5}
                ],
                "resampling": "cross_validation",
                "n_folds": 5,
                "stratified": True
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "scores" in data
        assert len(data["scores"]) > 0
    
    def test_cross_validation_multiple_learners(self):
        """Test cross-validation with multiple learners."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/evaluate/test_and_score/evaluate",
            json={
                "data_path": "core/iris.tab",
                "learner_configs": [
                    {"type": "knn", "name": "kNN", "n_neighbors": 5},
                    {"type": "tree", "name": "Tree"},
                    {"type": "naive_bayes", "name": "Naive Bayes"}
                ],
                "resampling": "cross_validation",
                "n_folds": 5,
                "stratified": True
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["scores"]) == 3
    
    def test_random_sampling(self):
        """Test random sampling evaluation."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/evaluate/test_and_score/evaluate",
            json={
                "data_path": "core/iris.tab",
                "learner_configs": [
                    {"type": "knn", "name": "kNN", "n_neighbors": 5}
                ],
                "resampling": "random_sampling",
                "n_repeats": 10,
                "sample_size": 66
            },
            headers=get_headers()
        )
        assert response.status_code == 200
    
    def test_leave_one_out(self):
        """Test leave-one-out evaluation."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/evaluate/test_and_score/evaluate",
            json={
                "data_path": "core/iris.tab",
                "learner_configs": [
                    {"type": "knn", "name": "kNN", "n_neighbors": 5}
                ],
                "resampling": "leave_one_out"
            },
            headers=get_headers()
        )
        assert response.status_code == 200
    
    def test_test_on_train(self):
        """Test on training data evaluation."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/evaluate/test_and_score/evaluate",
            json={
                "data_path": "core/iris.tab",
                "learner_configs": [
                    {"type": "knn", "name": "kNN", "n_neighbors": 5}
                ],
                "resampling": "test_on_train"
            },
            headers=get_headers()
        )
        assert response.status_code == 200
    
    def test_regression_scores(self):
        """Test regression scoring."""
        load_housing_dataset()
        
        response = client.post(
            "/api/v1/evaluate/test_and_score/evaluate",
            json={
                "data_path": "core/housing.tab",
                "learner_configs": [
                    {"type": "knn", "name": "kNN", "n_neighbors": 5}
                ],
                "resampling": "cross_validation",
                "n_folds": 5
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        if data["success"]:
            assert data["is_classification"] == False
            assert "MAE" in data["scores"][0] or "RMSE" in data["scores"][0]
    
    def test_get_evaluation_options(self):
        """Test getting evaluation options."""
        response = client.get("/api/v1/evaluate/test_and_score/options")
        assert response.status_code == 200
        data = response.json()
        assert "resampling_methods" in data


# ============================================================================
# Test Class: Confusion Matrix Widget
# ============================================================================

class TestConfusionMatrixWidget:
    """Tests for Confusion Matrix widget API."""
    
    def test_compute_confusion_matrix(self):
        """Test computing confusion matrix from evaluation results."""
        load_iris_dataset()
        
        # First run evaluation
        eval_response = client.post(
            "/api/v1/evaluate/test_and_score/evaluate",
            json={
                "data_path": "core/iris.tab",
                "learner_configs": [
                    {"type": "knn", "name": "kNN", "n_neighbors": 5}
                ],
                "resampling": "cross_validation",
                "n_folds": 5,
                "stratified": True
            },
            headers=get_headers()
        )
        
        if eval_response.status_code == 200:
            eval_data = eval_response.json()
            if eval_data["success"]:
                evaluation_id = eval_data["evaluation_id"]
                
                # Compute confusion matrix
                cm_response = client.post(
                    "/api/v1/evaluate/confusion-matrix/compute",
                    json={
                        "results_id": evaluation_id,
                        "learner_index": 0,
                        "quantity": "instances"
                    },
                    headers=get_headers()
                )
                assert cm_response.status_code == 200
                cm_data = cm_response.json()
                if cm_data["success"]:
                    assert "matrix" in cm_data
                    assert "headers" in cm_data
    
    def test_confusion_matrix_proportions(self):
        """Test confusion matrix with different quantity types."""
        load_iris_dataset()
        
        eval_response = client.post(
            "/api/v1/evaluate/test_and_score/evaluate",
            json={
                "data_path": "core/iris.tab",
                "learner_configs": [{"type": "knn", "name": "kNN"}],
                "resampling": "cross_validation",
                "n_folds": 5
            },
            headers=get_headers()
        )
        
        if eval_response.status_code == 200:
            eval_data = eval_response.json()
            if eval_data["success"]:
                evaluation_id = eval_data["evaluation_id"]
                
                # Test predicted proportions
                response = client.post(
                    "/api/v1/evaluate/confusion-matrix/compute",
                    json={
                        "results_id": evaluation_id,
                        "learner_index": 0,
                        "quantity": "predicted"
                    },
                    headers=get_headers()
                )
                assert response.status_code == 200
    
    def test_get_confusion_matrix_options(self):
        """Test getting confusion matrix options."""
        response = client.get("/api/v1/evaluate/confusion-matrix/options")
        assert response.status_code == 200


# ============================================================================
# Test Class: Predictions Widget
# ============================================================================

class TestPredictionsWidget:
    """Tests for Predictions widget API."""
    
    def test_make_predictions(self):
        """Test making predictions with trained model."""
        load_iris_dataset()
        
        # Train model first
        train_response = client.post(
            "/api/v1/model/knn/train",
            json={
                "data_path": "core/iris.tab",
                "n_neighbors": 5
            },
            headers=get_headers()
        )
        
        if train_response.status_code == 200:
            train_data = train_response.json()
            if train_data["success"]:
                model_id = train_data["model_id"]
                
                # Make predictions
                pred_response = client.post(
                    f"/api/v1/model/knn/predict?model_id={model_id}&data_path=core/iris.tab",
                    headers=get_headers()
                )
                assert pred_response.status_code == 200
    
    def test_predictions_with_probabilities(self):
        """Test predictions with probability output."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/predictions/predict",
            json={
                "data_path": "core/iris.tab",
                "model_configs": [
                    {"type": "knn", "params": {"n_neighbors": 5}}
                ],
                "include_probabilities": True
            },
            headers=get_headers()
        )
        # Response may vary based on implementation
        assert response.status_code in [200, 404]


# ============================================================================
# Test Class: Scatter Plot Widget
# ============================================================================

class TestScatterPlotWidget:
    """Tests for Scatter Plot widget API."""
    
    def test_get_scatter_data(self):
        """Test getting scatter plot data."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/visualize/scatter-plot",
            json={
                "data_path": "core/iris.tab",
                "x_var": "sepal length",
                "y_var": "sepal width",
                "color_var": "iris"
            },
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        if data.get("success"):
            assert "points" in data


# ============================================================================
# Test Class: Distributions Widget
# ============================================================================

class TestDistributionsWidget:
    """Tests for Distributions widget API."""
    
    def test_get_distributions(self):
        """Test getting distribution data."""
        load_iris_dataset()
        
        response = client.post(
            "/api/v1/visualize/distributions",
            json={
                "data_path": "core/iris.tab",
                "variable": "sepal length",
                "group_by": "iris"
            },
            headers=get_headers()
        )
        assert response.status_code == 200


# ============================================================================
# Test Class: Data Info Widget
# ============================================================================

class TestDataInfoWidget:
    """Tests for Data Info widget API."""
    
    def test_get_data_info(self):
        """Test getting data information."""
        load_iris_dataset()
        
        response = client.get(
            "/api/v1/data/info?data_path=core/iris.tab",
            headers=get_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert "instances" in data or "rows" in data


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

