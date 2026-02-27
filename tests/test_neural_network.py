"""
Unit tests for Neural Network Widget API.
Comprehensive tests for Multi-Layer Perceptron model training.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestNeuralNetworkOptions:
    """Test Neural Network options endpoint."""

    def test_get_neural_network_options(self):
        """Test getting Neural Network options returns expected structure."""
        response = client.get("/api/v1/model/neural-network/options")

        assert response.status_code == 200
        data = response.json()

        assert "activations" in data
        assert "default_activation" in data
        assert "alpha_range" in data
        assert "learning_rate_range" in data
        assert "max_iter_range" in data

    def test_activations_include_required_values(self):
        """All three required activation functions must be present."""
        response = client.get("/api/v1/model/neural-network/options")

        assert response.status_code == 200
        data = response.json()

        activation_values = [a["value"] for a in data["activations"]]
        assert "relu" in activation_values
        assert "logistic" in activation_values
        assert "tanh" in activation_values

    def test_default_activation_is_relu(self):
        """Default activation should be relu."""
        response = client.get("/api/v1/model/neural-network/options")

        assert response.status_code == 200
        assert response.json()["default_activation"] == "relu"

    def test_alpha_range_is_valid(self):
        """Alpha range should have min, max, and default."""
        response = client.get("/api/v1/model/neural-network/options")

        assert response.status_code == 200
        alpha_range = response.json()["alpha_range"]

        assert "min" in alpha_range
        assert "max" in alpha_range
        assert "default" in alpha_range
        assert alpha_range["min"] >= 0
        assert alpha_range["max"] > alpha_range["min"]


class TestNeuralNetworkTrainBasic:
    """Basic Neural Network training tests."""

    def test_train_neural_network_iris_defaults(self):
        """Train Neural Network with default params on iris dataset."""
        response = client.post(
            "/api/v1/model/neural-network/train", json={"data_path": "iris"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "model_id" in data
        assert data["model_id"] is not None

    def test_train_neural_network_returns_model_info(self):
        """Training response should include model_info with expected fields."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "hidden_layers": [100], "activation": "relu"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        model_info = data.get("model_info", {})
        assert model_info.get("training_instances") == 150
        assert model_info.get("type") == "classification"
        assert "features" in model_info
        assert "target" in model_info
        assert "hidden_layers" in model_info

    def test_train_neural_network_returns_learner_params(self):
        """Training response should echo back learner_params."""
        request_data = {
            "data_path": "iris",
            "hidden_layers": [50, 25],
            "activation": "tanh",
            "alpha": 0.001,
            "learning_rate_init": 0.01,
            "max_iter": 100,
            "early_stopping": True,
        }
        response = client.post("/api/v1/model/neural-network/train", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        params = data.get("learner_params", {})
        assert params.get("hidden_layers") == [50, 25]
        assert params.get("activation") == "tanh"
        assert params.get("alpha") == 0.001
        assert params.get("learning_rate_init") == 0.01
        assert params.get("max_iter") == 100
        assert params.get("early_stopping") is True

    def test_multiple_trainings_produce_unique_ids(self):
        """Each training call must produce a unique model ID."""
        model_ids = []
        for _ in range(3):
            response = client.post(
                "/api/v1/model/neural-network/train", json={"data_path": "iris"}
            )
            assert response.status_code == 200
            model_ids.append(response.json()["model_id"])

        assert len(set(model_ids)) == 3, "Expected 3 unique model IDs"


class TestNeuralNetworkActivations:
    """Test each activation function."""

    def test_train_relu_activation(self):
        """Train Neural Network with ReLU activation."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "activation": "relu"},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_train_logistic_activation(self):
        """Train Neural Network with Logistic activation."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "activation": "logistic"},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_train_tanh_activation(self):
        """Train Neural Network with Tanh activation."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "activation": "tanh"},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_train_invalid_activation(self):
        """Invalid activation function should be rejected."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "activation": "sigmoid"},
        )

        assert response.status_code == 400 or (
            response.status_code == 200 and response.json().get("success") is False
        )


class TestNeuralNetworkHiddenLayers:
    """Test hidden layer configurations."""

    def test_single_layer_100(self):
        """Train with a single hidden layer of 100 neurons."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "hidden_layers": [100]},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_two_layers(self):
        """Train with two hidden layers."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "hidden_layers": [100, 50]},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_three_layers(self):
        """Train with three hidden layers."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "hidden_layers": [100, 50, 25]},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_small_layer(self):
        """Train with a very small hidden layer."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "hidden_layers": [5]},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_empty_hidden_layers_is_invalid(self):
        """Empty hidden_layers should be rejected."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "hidden_layers": []},
        )

        assert response.status_code == 400 or (
            response.status_code == 200 and response.json().get("success") is False
        )

    def test_zero_neurons_is_invalid(self):
        """Layer with 0 neurons should be rejected."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "hidden_layers": [0]},
        )

        assert response.status_code == 400 or (
            response.status_code == 200 and response.json().get("success") is False
        )


class TestNeuralNetworkAlpha:
    """Test alpha (L2 regularization) parameter."""

    def test_small_alpha(self):
        """Train with small alpha (weak regularization)."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "alpha": 0.00001},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_large_alpha(self):
        """Train with large alpha (strong regularization)."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "alpha": 0.5},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_zero_alpha(self):
        """Alpha=0 (no regularization) should be valid."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "alpha": 0.0},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_negative_alpha_is_invalid(self):
        """Negative alpha should be rejected."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "alpha": -0.1},
        )

        assert response.status_code == 400 or (
            response.status_code == 200 and response.json().get("success") is False
        )


class TestNeuralNetworkLearningRate:
    """Test learning rate parameter."""

    def test_small_learning_rate(self):
        """Train with a small learning rate."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "learning_rate_init": 0.0001},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_large_learning_rate(self):
        """Train with a larger learning rate."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "learning_rate_init": 0.1},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_zero_learning_rate_is_invalid(self):
        """Learning rate of 0 should be rejected."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "learning_rate_init": 0.0},
        )

        assert response.status_code == 400 or (
            response.status_code == 200 and response.json().get("success") is False
        )


class TestNeuralNetworkMaxIter:
    """Test max_iter parameter."""

    def test_low_max_iter(self):
        """Train with low max_iter (may not fully converge)."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "max_iter": 10},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_high_max_iter(self):
        """Train with higher max_iter."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "max_iter": 500},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_zero_max_iter_is_invalid(self):
        """max_iter=0 should be rejected."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "max_iter": 0},
        )

        assert response.status_code == 400 or (
            response.status_code == 200 and response.json().get("success") is False
        )


class TestNeuralNetworkEarlyStopping:
    """Test early stopping feature."""

    def test_early_stopping_disabled(self):
        """Train without early stopping (default)."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris", "early_stopping": False},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_early_stopping_enabled(self):
        """Train with early stopping enabled."""
        response = client.post(
            "/api/v1/model/neural-network/train",
            json={
                "data_path": "iris",
                "early_stopping": True,
                "max_iter": 200,
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] is True


class TestNeuralNetworkModelInfo:
    """Test model info retrieval."""

    def test_get_model_info_after_training(self):
        """Get model info after training should return metadata."""
        train_response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris"},
        )
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]

        info_response = client.get(f"/api/v1/model/neural-network/info/{model_id}")
        assert info_response.status_code == 200

        data = info_response.json()
        assert data.get("model_id") == model_id
        assert "learner_params" in data
        params = data["learner_params"]
        assert "hidden_layers" in params
        assert "activation" in params
        assert "alpha" in params

    def test_get_nonexistent_model_info(self):
        """Requesting info for a non-existent model returns 404."""
        response = client.get("/api/v1/model/neural-network/info/nonexistent_id")
        assert response.status_code == 404


class TestNeuralNetworkModelDeletion:
    """Test model deletion."""

    def test_delete_model(self):
        """Delete a trained model successfully."""
        train_response = client.post(
            "/api/v1/model/neural-network/train",
            json={"data_path": "iris"},
        )
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]

        delete_response = client.delete(f"/api/v1/model/neural-network/{model_id}")
        assert delete_response.status_code == 200
        assert delete_response.json().get("success") is True

        # Model must be gone
        info_response = client.get(f"/api/v1/model/neural-network/info/{model_id}")
        assert info_response.status_code == 404

    def test_delete_nonexistent_model(self):
        """Deleting a non-existent model returns 404."""
        response = client.delete("/api/v1/model/neural-network/nonexistent_id")
        assert response.status_code == 404


class TestNeuralNetworkParameterCombinations:
    """Test combinations of parameters."""

    def test_all_activations_default_layers(self):
        """All three activation functions should train successfully."""
        activations = ["relu", "logistic", "tanh"]
        for activation in activations:
            response = client.post(
                "/api/v1/model/neural-network/train",
                json={"data_path": "iris", "activation": activation},
            )
            assert response.status_code == 200, f"Failed for activation={activation}"
            assert response.json()["success"] is True, (
                f"Training failed for activation={activation}"
            )

    def test_various_layer_configs(self):
        """Several hidden layer configurations should all train successfully."""
        layer_configs = [
            [50],
            [100, 50],
            [64, 32, 16],
        ]
        for layers in layer_configs:
            response = client.post(
                "/api/v1/model/neural-network/train",
                json={"data_path": "iris", "hidden_layers": layers},
            )
            assert response.status_code == 200, f"Failed for layers={layers}"
            assert response.json()["success"] is True, (
                f"Training failed for layers={layers}"
            )

    def test_full_parameter_combination(self):
        """Train with all parameters explicitly set."""
        request_data = {
            "data_path": "iris",
            "hidden_layers": [64, 32],
            "activation": "logistic",
            "alpha": 0.001,
            "learning_rate_init": 0.005,
            "max_iter": 50,
            "early_stopping": False,
            "name": "Test NN",
        }
        response = client.post("/api/v1/model/neural-network/train", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        params = data.get("learner_params", {})
        assert params.get("hidden_layers") == [64, 32]
        assert params.get("activation") == "logistic"
        assert params.get("alpha") == 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
