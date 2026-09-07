"""
API request/response schemas
"""
from pydantic import BaseModel, Field
from typing import Optional


class TransactionRequest(BaseModel):
    """Schema for fraud prediction request"""
    Time: float = Field(..., ge=0, description="Transaction time in seconds")
    Amount: float = Field(..., ge=0, description="Transaction amount")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    transaction_id: Optional[str] = None
    merchant_id: Optional[str] = None


class PredictionResponse(BaseModel):
    """Schema for fraud prediction response"""
    is_fraud: bool
    fraud_probability: float
    confidence: str
    threshold: float
    model_version: str
    prediction_id: str
    timestamp: str


class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str
    model_loaded: bool
    scaler_loaded: bool
    model_version: str
    uptime_seconds: float
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Schema for model information response"""
    model_version: str
    model_type: str
    features: list
    threshold: float
    last_trained: Optional[str] = None
