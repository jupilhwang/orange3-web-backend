"""
Unit tests for SVM Widget API.
Comprehensive tests for Support Vector Machine model training.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestSVMOptions:
    """Test SVM options endpoint."""

    def test_get_svm_options(self):
        """Test getting SVM options returns expected structure."""
        response = client.get("/api/v1/model/svm/options")

        assert response.status_code == 200
        data = response.json()

        assert "kernels" in data
        assert "default_kernel" in data
        assert "C_range" in data
        assert "gamma_options" in data
        assert "degree_range" in data

    def test_svm_kernels_include_required_values(self):
        """All four required kernel types must be present."""
        response = client.get("/api/v1/model/svm/options")

        assert response.status_code == 200
        data = response.json()

        kernel_values = [k["value"] for k in data["kernels"]]
        assert "RBF" in kernel_values
        assert "linear" in kernel_values
        assert "polynomial" in kernel_values
        assert "sigmoid" in kernel_values

    def test_svm_c_range(self):
        """C range should have min, max, and default."""
        response = client.get("/api/v1/model/svm/options")

        assert response.status_code == 200
        c_range = response.json()["C_range"]

        assert "min" in c_range
        assert "max" in c_range
        assert "default" in c_range
        assert c_range["min"] > 0
        assert c_range["max"] >= c_range["min"]


class TestSVMTrainBasic:
    """Basic SVM training tests."""

    def test_train_svm_iris_rbf(self):
        """Train SVM with RBF kernel on iris dataset."""
        request_data = {
            "data_path": "iris",
            "kernel": "RBF",
            "C": 1.0,
            "gamma": "auto",
        }
        response = client.post("/api/v1/model/svm/train", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "model_id" in data
        assert data["model_id"] is not None

    def test_train_svm_default_params(self):
        """Train SVM with only required field (data_path)."""
        response = client.post("/api/v1/model/svm/train", json={"data_path": "iris"})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_train_svm_returns_model_info(self):
        """Training response should include model_info with expected fields."""
        request_data = {
            "data_path": "iris",
            "kernel": "RBF",
            "C": 1.0,
            "gamma": "auto",
        }
        response = client.post("/api/v1/model/svm/train", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        model_info = data.get("model_info", {})
        assert model_info.get("training_instances") == 150
        assert model_info.get("type") == "classification"
        assert "features" in model_info
        assert "target" in model_info


class TestSVMKernels:
    """Test each kernel type."""

    def test_train_svm_rbf(self):
        """Train SVM with RBF kernel."""
        response = client.post(
            "/api/v1/model/svm/train", json={"data_path": "iris", "kernel": "RBF"}
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_train_svm_linear(self):
        """Train SVM with linear kernel."""
        response = client.post(
            "/api/v1/model/svm/train", json={"data_path": "iris", "kernel": "linear"}
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_train_svm_polynomial(self):
        """Train SVM with polynomial kernel."""
        response = client.post(
            "/api/v1/model/svm/train",
            json={"data_path": "iris", "kernel": "polynomial", "degree": 3},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_train_svm_sigmoid(self):
        """Train SVM with sigmoid kernel."""
        response = client.post(
            "/api/v1/model/svm/train", json={"data_path": "iris", "kernel": "sigmoid"}
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_train_svm_invalid_kernel(self):
        """Invalid kernel should be rejected."""
        response = client.post(
            "/api/v1/model/svm/train",
            json={"data_path": "iris", "kernel": "invalid_kernel"},
        )

        # Must be HTTP 400 or success=False
        assert response.status_code == 400 or (
            response.status_code == 200 and response.json().get("success") is False
        )


class TestSVMCParameter:
    """Test C (regularization) parameter."""

    def test_train_svm_small_C(self):
        """Train SVM with small C (high regularization)."""
        response = client.post(
            "/api/v1/model/svm/train",
            json={"data_path": "iris", "kernel": "RBF", "C": 0.01},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_train_svm_large_C(self):
        """Train SVM with large C (low regularization)."""
        response = client.post(
            "/api/v1/model/svm/train",
            json={"data_path": "iris", "kernel": "RBF", "C": 100.0},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_train_svm_invalid_C_zero(self):
        """C=0 is invalid."""
        response = client.post(
            "/api/v1/model/svm/train",
            json={"data_path": "iris", "kernel": "RBF", "C": 0.0},
        )

        assert response.status_code == 400 or (
            response.status_code == 200 and response.json().get("success") is False
        )

    def test_train_svm_invalid_C_negative(self):
        """Negative C is invalid."""
        response = client.post(
            "/api/v1/model/svm/train",
            json={"data_path": "iris", "kernel": "RBF", "C": -1.0},
        )

        assert response.status_code == 400 or (
            response.status_code == 200 and response.json().get("success") is False
        )


class TestSVMGamma:
    """Test gamma parameter variants."""

    def test_train_svm_gamma_auto(self):
        """Train with gamma='auto'."""
        response = client.post(
            "/api/v1/model/svm/train",
            json={"data_path": "iris", "kernel": "RBF", "gamma": "auto"},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_train_svm_gamma_scale(self):
        """Train with gamma='scale'."""
        response = client.post(
            "/api/v1/model/svm/train",
            json={"data_path": "iris", "kernel": "RBF", "gamma": "scale"},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_train_svm_gamma_manual_float(self):
        """Train with manual float gamma value."""
        response = client.post(
            "/api/v1/model/svm/train",
            json={"data_path": "iris", "kernel": "RBF", "gamma": "0.1"},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_train_svm_invalid_gamma_negative(self):
        """Negative manual gamma is invalid."""
        response = client.post(
            "/api/v1/model/svm/train",
            json={"data_path": "iris", "kernel": "RBF", "gamma": "-0.5"},
        )

        assert response.status_code == 400 or (
            response.status_code == 200 and response.json().get("success") is False
        )


class TestSVMDegree:
    """Test degree parameter for polynomial kernel."""

    def test_train_svm_degree_1(self):
        """Polynomial kernel degree=1."""
        response = client.post(
            "/api/v1/model/svm/train",
            json={"data_path": "iris", "kernel": "polynomial", "degree": 1},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_train_svm_degree_5(self):
        """Polynomial kernel degree=5."""
        response = client.post(
            "/api/v1/model/svm/train",
            json={"data_path": "iris", "kernel": "polynomial", "degree": 5},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_train_svm_invalid_degree_zero(self):
        """Degree=0 is invalid."""
        response = client.post(
            "/api/v1/model/svm/train",
            json={"data_path": "iris", "kernel": "polynomial", "degree": 0},
        )

        assert response.status_code == 400 or (
            response.status_code == 200 and response.json().get("success") is False
        )


class TestSVMProbability:
    """Test probability estimates feature."""

    def test_train_svm_probability_disabled(self):
        """Train without probability estimates (default)."""
        response = client.post(
            "/api/v1/model/svm/train",
            json={"data_path": "iris", "kernel": "RBF", "probability": False},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_train_svm_probability_enabled(self):
        """Train with probability estimates enabled."""
        response = client.post(
            "/api/v1/model/svm/train",
            json={"data_path": "iris", "kernel": "RBF", "probability": True},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True


class TestSVMModelInfo:
    """Test model info retrieval."""

    def test_get_model_info_after_training(self):
        """Get model info after training should return metadata."""
        train_response = client.post(
            "/api/v1/model/svm/train", json={"data_path": "iris", "kernel": "RBF"}
        )
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]

        info_response = client.get(f"/api/v1/model/svm/info/{model_id}")
        assert info_response.status_code == 200

        data = info_response.json()
        assert data.get("model_id") == model_id
        assert "learner_params" in data

    def test_get_nonexistent_model_info(self):
        """Requesting info for a non-existent model returns 404."""
        response = client.get("/api/v1/model/svm/info/nonexistent_id")
        assert response.status_code == 404


class TestSVMModelDeletion:
    """Test model deletion."""

    def test_delete_model(self):
        """Delete a trained model successfully."""
        train_response = client.post(
            "/api/v1/model/svm/train", json={"data_path": "iris", "kernel": "RBF"}
        )
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]

        delete_response = client.delete(f"/api/v1/model/svm/{model_id}")
        assert delete_response.status_code == 200

        # Model must be gone
        info_response = client.get(f"/api/v1/model/svm/info/{model_id}")
        assert info_response.status_code == 404

    def test_delete_nonexistent_model(self):
        """Deleting a non-existent model returns 404."""
        response = client.delete("/api/v1/model/svm/nonexistent_id")
        assert response.status_code == 404


class TestSVMParameterCombinations:
    """Test combinations of parameters."""

    def test_all_kernels_default_C(self):
        """All four kernels should train successfully with default C."""
        kernels = ["RBF", "linear", "polynomial", "sigmoid"]
        for kernel in kernels:
            response = client.post(
                "/api/v1/model/svm/train", json={"data_path": "iris", "kernel": kernel}
            )
            assert response.status_code == 200, f"Failed for kernel={kernel}"
            assert response.json()["success"] is True, (
                f"Training failed for kernel={kernel}"
            )

    def test_multiple_trainings_unique_ids(self):
        """Each training call must produce a unique model ID."""
        model_ids = []
        for _ in range(3):
            response = client.post(
                "/api/v1/model/svm/train", json={"data_path": "iris", "kernel": "RBF"}
            )
            assert response.status_code == 200
            model_ids.append(response.json()["model_id"])

        assert len(set(model_ids)) == 3, "Expected 3 unique model IDs"

    def test_rbf_with_various_C_values(self):
        """RBF kernel should work with a range of C values."""
        c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
        for c in c_values:
            response = client.post(
                "/api/v1/model/svm/train",
                json={"data_path": "iris", "kernel": "RBF", "C": c},
            )
            assert response.status_code == 200, f"Failed for C={c}"
            assert response.json()["success"] is True, f"Training failed for C={c}"


class TestSVMLearnerParams:
    """Verify learner_params are echoed back in the response."""

    def test_learner_params_in_response(self):
        """Response must include learner_params matching request."""
        request_data = {
            "data_path": "iris",
            "kernel": "polynomial",
            "C": 2.0,
            "gamma": "scale",
            "degree": 4,
            "probability": True,
        }
        response = client.post("/api/v1/model/svm/train", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        params = data.get("learner_params", {})
        assert params.get("kernel") == "polynomial"
        assert params.get("C") == 2.0
        assert params.get("gamma") == "scale"
        assert params.get("degree") == 4
        assert params.get("probability") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
