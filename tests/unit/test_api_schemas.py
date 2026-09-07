"""
Unit tests for API schemas
"""
import pytest
from pydantic import ValidationError
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.api.schemas import TransactionRequest, PredictionResponse, HealthResponse


def test_transaction_request_valid():
    """Test valid transaction request"""
    transaction = TransactionRequest(
        Time=0.0,
        Amount=149.62,
        V1=-1.359807,
        V2=-0.072781,
        V3=2.536347,
        V4=1.378155,
        V5=-0.338321,
        V6=0.462388,
        V7=0.239599,
        V8=0.098698,
        V9=0.363787,
        V10=0.090794,
        V11=-0.551600,
        V12=-0.617801,
        V13=-0.991390,
        V14=-0.347886,
        V15=-0.905631,
        V16=-0.683095,
        V17=-0.683095,
        V18=0.014724,
        V19=0.234804,
        V20=0.321546,
        V21=0.036782,
        V22=0.025194,
        V23=0.028357,
        V24=-0.011577,
        V25=0.014651,
        V26=-0.023425,
        V27=0.012321,
        V28=0.003521,
        Amount=149.62
    )
    
    assert transaction.Time == 0.0
    assert transaction.Amount == 149.62


def test_transaction_request_invalid_amount():
    """Test transaction request with invalid amount"""
    with pytest.raises(ValidationError):
        TransactionRequest(
            Time=0.0,
            Amount=-100,  # Invalid: negative amount
            V1=-1.359807,
            V2=-0.072781,
            V3=2.536347,
            V4=1.378155,
            V5=-0.338321,
            V6=0.462388,
            V7=0.239599,
            V8=0.098698,
            V9=0.363787,
            V10=0.090794,
            V11=-0.551600,
            V12=-0.617801,
            V13=-0.991390,
            V14=-0.347886,
            V15=-0.905631,
            V16=-0.683095,
            V17=-0.683095,
            V18=0.014724,
            V19=0.234804,
            V20=0.321546,
            V21=0.036782,
            V22=0.025194,
            V23=0.028357,
            V24=-0.011577,
            V25=0.014651,
            V26=-0.023425,
            V27=0.012321,
            V28=0.003521,
            Amount=149.62
        )


def test_prediction_response():
    """Test prediction response schema"""
    response = PredictionResponse(
        is_fraud=False,
        fraud_probability=0.1234,
        confidence="Low",
        threshold=0.5,
        model_version="latest",
        prediction_id="test-id",
        timestamp="2024-01-01T12:00:00.000000"
    )
    
    assert response.is_fraud is False
    assert response.fraud_probability == 0.1234
    assert response.confidence == "Low"


def test_health_response():
    """Test health response schema"""
    response = HealthResponse(
        status="healthy",
        model_loaded=True,
        scaler_loaded=True,
        model_version="latest",
        uptime_seconds=123.45,
        timestamp="2024-01-01T12:00:00.000000"
    )
    
    assert response.status == "healthy"
    assert response.model_loaded is True
    assert response.scaler_loaded is True
