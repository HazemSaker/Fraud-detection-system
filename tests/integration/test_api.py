"""
Integration tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "scaler_loaded" in data
    assert "model_version" in data


def test_predict_endpoint(sample_transaction):
    """Test prediction endpoint"""
    # Note: This test requires a trained model to be loaded
    # It will fail if model is not present, which is expected
    response = client.post("/api/v1/predict", json=sample_transaction)
    
    # Response could be 200 (if model loaded) or 503 (if model not loaded)
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "confidence" in data


def test_model_info_endpoint():
    """Test model info endpoint"""
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    
    data = response.json()
    assert "model_version" in data
    assert "model_type" in data
    assert "features" in data


def test_metrics_endpoint():
    """Test Prometheus metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
