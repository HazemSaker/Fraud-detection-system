"""
FastAPI application for fraud detection
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from datetime import datetime
import uuid
import joblib
import numpy as np
import pandas as pd
import time

from config.settings import settings
from app.api.schemas import TransactionRequest, PredictionResponse, HealthResponse, ModelInfoResponse
from app.api.rate_limiter import check_rate_limit
from app.api.monitoring import PredictionTimer, APITimer, record_prediction, get_metrics

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

# Global variables
model = None
scaler = None
threshold = settings.THRESHOLD
feature_names = None
start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Load model and scaler on startup"""
    global model, scaler, feature_names
    try:
        model = joblib.load(settings.MODEL_PATH)
        scaler = joblib.load(settings.SCALER_PATH)
        feature_names = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Hour', 'Amount_log']
        print("Model and scaler loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Fraud Detection API", "version": settings.APP_VERSION}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        scaler_loaded=scaler is not None,
        model_version=settings.MODEL_VERSION,
        uptime_seconds=time.time() - start_time,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionRequest, request: Request):
    """Predict fraud for a single transaction"""
    # Rate limiting
    client_ip = request.client.host
    check_rate_limit(client_ip)
    
    with APITimer("predict"):
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Prepare features
        df = pd.DataFrame([transaction.dict()])
        df["Hour"] = (df["Time"] % 86400) // 3600
        df["Amount_log"] = np.log1p(df["Amount"])
        df = df[feature_names]
        
        # Scale
        df_scaled = scaler.transform(df)
        
        # Predict
        with PredictionTimer():
            fraud_proba = model.predict_proba(df_scaled)[0, 1]
            is_fraud = fraud_proba >= threshold
        
        # Confidence level
        if fraud_proba >= 0.9:
            confidence = "Very High"
        elif fraud_proba >= 0.7:
            confidence = "High"
        elif fraud_proba >= 0.5:
            confidence = "Medium"
        elif fraud_proba >= 0.3:
            confidence = "Low"
        else:
            confidence = "Very Low"
        
        record_prediction(confidence)
        
        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=float(fraud_proba),
            confidence=confidence,
            threshold=threshold,
            model_version=settings.MODEL_VERSION,
            prediction_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat()
        )


@app.get("/api/v1/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information"""
    return ModelInfoResponse(
        model_version=settings.MODEL_VERSION,
        model_type="LightGBM",
        features=feature_names if feature_names else [],
        threshold=threshold
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return get_metrics()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )
