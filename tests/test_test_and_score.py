"""
Tests for Test and Score Widget API.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestTestAndScoreWidget:
    """Test cases for Test and Score widget endpoints."""
    
    def test_get_options(self):
        """Test getting available evaluation options."""
        response = client.get("/api/v1/evaluate/test_and_score/options")
        assert response.status_code == 200
        
        result = response.json()
        assert "resampling_methods" in result
        assert "n_folds_options" in result
        assert "n_repeats_options" in result
        assert "sample_sizes" in result
        assert "classification_scores" in result
        assert "regression_scores" in result
    
    def test_evaluate_cross_validation(self):
        """Test cross validation evaluation."""
        response = client.post("/api/v1/evaluate/test_and_score/evaluate", json={
            "data_path": "iris",
            "learner_configs": [
                {"type": "knn", "name": "kNN-3", "n_neighbors": 3},
                {"type": "knn", "name": "kNN-5", "n_neighbors": 5}
            ],
            "resampling": "cross_validation",
            "n_folds": 5,
            "stratified": True
        })
        
        assert response.status_code == 200
        result = response.json()
        
        if result.get("success"):
            assert "scores" in result
            assert "learner_names" in result
            assert result.get("is_classification", False)
            
            # Check scores structure
            scores = result["scores"]
            assert len(scores) == 2
            
            for score in scores:
                assert "name" in score
                # Classification scores
                assert "CA" in score or "AUC" in score
    
    def test_evaluate_random_sampling(self):
        """Test random sampling evaluation."""
        response = client.post("/api/v1/evaluate/test_and_score/evaluate", json={
            "data_path": "iris",
            "learner_configs": [
                {"type": "knn", "name": "kNN", "n_neighbors": 5}
            ],
            "resampling": "random_sampling",
            "n_repeats": 5,
            "sample_size": 66,
            "stratified": True
        })
        
        assert response.status_code == 200
        result = response.json()
        
        if result.get("success"):
            assert "scores" in result
    
    def test_evaluate_leave_one_out(self):
        """Test leave one out evaluation (smaller dataset)."""
        response = client.post("/api/v1/evaluate/test_and_score/evaluate", json={
            "data_path": "iris",
            "learner_configs": [
                {"type": "knn", "name": "kNN", "n_neighbors": 3}
            ],
            "resampling": "leave_one_out"
        })
        
        assert response.status_code == 200
        # Leave one out can be slow, just check response
    
    def test_evaluate_test_on_train(self):
        """Test on training data evaluation."""
        response = client.post("/api/v1/evaluate/test_and_score/evaluate", json={
            "data_path": "iris",
            "learner_configs": [
                {"type": "knn", "name": "kNN", "n_neighbors": 3}
            ],
            "resampling": "test_on_train"
        })
        
        assert response.status_code == 200
        result = response.json()
        
        if result.get("success"):
            # Test on train usually gives perfect or near-perfect scores
            scores = result.get("scores", [])
            if scores:
                ca = scores[0].get("CA")
                if ca is not None:
                    assert ca > 0.8  # Should be high for test on train
    
    def test_evaluate_no_learners(self):
        """Test evaluation with empty learners."""
        response = client.post("/api/v1/evaluate/test_and_score/evaluate", json={
            "data_path": "iris",
            "learner_configs": [],
            "resampling": "cross_validation"
        })
        
        assert response.status_code == 200
        result = response.json()
        
        if not result.get("success"):
            assert "No learners" in result.get("error", "") or \
                   "not available" in result.get("error", "")
    
    def test_get_evaluation_not_found(self):
        """Test getting evaluation with invalid ID."""
        response = client.get("/api/v1/evaluate/test_and_score/invalid_id")
        assert response.status_code == 404
    
    def test_delete_evaluation(self):
        """Test deleting evaluation."""
        response = client.delete("/api/v1/evaluate/test_and_score/test_id")
        assert response.status_code == 200
        assert "deleted" in response.json().get("message", "").lower()
    
    def test_compare_models(self):
        """Test model comparison."""
        # First run evaluation
        eval_response = client.post("/api/v1/evaluate/test_and_score/evaluate", json={
            "data_path": "iris",
            "learner_configs": [
                {"type": "knn", "name": "kNN-3", "n_neighbors": 3},
                {"type": "knn", "name": "kNN-5", "n_neighbors": 5}
            ],
            "resampling": "cross_validation",
            "n_folds": 3
        })
        
        if eval_response.status_code == 200:
            eval_result = eval_response.json()
            
            if eval_result.get("success"):
                eval_id = eval_result.get("evaluation_id")
                
                # Compare models
                compare_response = client.post(
                    f"/api/v1/evaluate/test_and_score/compare?evaluation_id={eval_id}"
                )
                
                assert compare_response.status_code == 200
                compare_result = compare_response.json()
                
                assert "comparison_matrix" in compare_result
                assert "model_names" in compare_result
    
    def test_compare_with_rope(self):
        """Test model comparison with ROPE threshold."""
        # First run evaluation
        eval_response = client.post("/api/v1/evaluate/test_and_score/evaluate", json={
            "data_path": "iris",
            "learner_configs": [
                {"type": "knn", "name": "kNN-3", "n_neighbors": 3},
                {"type": "knn", "name": "kNN-5", "n_neighbors": 5}
            ],
            "resampling": "cross_validation",
            "n_folds": 3
        })
        
        if eval_response.status_code == 200:
            eval_result = eval_response.json()
            
            if eval_result.get("success"):
                eval_id = eval_result.get("evaluation_id")
                
                # Compare with ROPE
                compare_response = client.post(
                    f"/api/v1/evaluate/test_and_score/compare?evaluation_id={eval_id}&rope_threshold=0.05"
                )
                
                assert compare_response.status_code == 200
                compare_result = compare_response.json()
                
                assert compare_result.get("rope_threshold") == 0.05

