"""
FastAPI application for fraud detection
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from app.api.auth import (
    create_access_token,
    get_password_hash,
    verify_password,
    verify_token,
)
from app.api.monitoring import APITimer, PredictionTimer, get_metrics, record_prediction
from app.api.rate_limiter import check_rate_limit
from app.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)
from config.settings import settings

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

# Global variables
model = None
scaler = None
threshold = (
    settings.THRESHOLD
)  # overwritten at startup from model_metadata.json if present
model_type = "unknown"
feature_names = None
start_time = time.time()

# --- Auth -----------------------------------------------------------------
# auto_error=False so the dependency can decide what "no token" means based
# on settings.ENABLE_AUTH, instead of FastAPI auto-rejecting every request.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# DEMO CREDENTIALS ONLY -- replace with a real user store (DB row, env var,
# whatever) before this ever sits anywhere but your own machine.
# Hashed lazily in startup_event() rather than here at import time: hashing
# at module load means any backend issue (e.g. a passlib/bcrypt version
# mismatch -- see requirements.txt) crashes the whole process on import,
# before uvicorn even starts logging. Doing it in startup_event() instead
# surfaces the same failure as a clear one-line log message.
DEMO_USERNAME = "admin"
DEMO_PASSWORD_HASH = None


async def require_auth(token: str = Depends(oauth2_scheme)) -> None:
    """Gate on settings.ENABLE_AUTH so the API can run open for local
    testing/demos, while still being able to actually enforce the JWT
    issued by POST /token when auth is turned on."""
    if not settings.ENABLE_AUTH:
        return
    if token is None:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    verify_token(token)


@app.on_event("startup")
async def startup_event():
    """Load model and scaler on startup"""
    global model, scaler, feature_names, threshold, model_type, DEMO_PASSWORD_HASH

    if settings.ENABLE_AUTH:
        try:
            DEMO_PASSWORD_HASH = get_password_hash("changeme")
        except Exception as e:
            # Don't take down model loading over an auth backend issue --
            # log it clearly and leave auth effectively unusable (login
            # will fail cleanly) rather than crashing the whole process.
            print(
                f"Could not hash demo password (check passlib/bcrypt versions in requirements.txt): {e}"
            )

    try:
        model = joblib.load(settings.MODEL_PATH)
        scaler = joblib.load(settings.SCALER_PATH)
        # This must exactly match the column order preprocessor.py's
        # engineer_features() produces: it drops Time (after deriving
        # Hour from it) but does NOT drop Amount -- it only adds
        # Amount_log alongside the original Amount column. So the scaler
        # was actually fit on 31 columns, including raw Amount, not 30.
        feature_names = [
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "V7",
            "V8",
            "V9",
            "V10",
            "V11",
            "V12",
            "V13",
            "V14",
            "V15",
            "V16",
            "V17",
            "V18",
            "V19",
            "V20",
            "V21",
            "V22",
            "V23",
            "V24",
            "V25",
            "V26",
            "V27",
            "V28",
            "Amount",
            "Hour",
            "Amount_log",
        ]

        # Read the metadata main.py saves alongside best_model.pkl, so the
        # API reports the model that was actually saved (not a hardcoded
        # guess) and uses the threshold that was actually tuned for it (not
        # whatever static number happens to be in .env).
        metadata_path = Path(settings.MODEL_PATH).parent / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            threshold = meta["threshold"]
            model_type = meta["model_name"]
            print(
                f"Loaded {model_type} (threshold={threshold}) from model_metadata.json"
            )
        else:
            print(
                f"No model_metadata.json found -- falling back to .env THRESHOLD={threshold}"
            )

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
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Issue a JWT for the demo user. See DEMO_USERNAME/DEMO_PASSWORD_HASH
    above -- swap this check for a real user lookup before deploying."""
    if DEMO_PASSWORD_HASH is None:
        raise HTTPException(
            status_code=503, detail="Auth backend unavailable (check server logs)"
        )
    if form_data.username != DEMO_USERNAME or not verify_password(
        form_data.password, DEMO_PASSWORD_HASH
    ):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token({"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}


def _score_transaction(transaction: TransactionRequest) -> PredictionResponse:
    """Core scoring logic shared by /predict and /predict/batch. Assumes
    the caller has already checked model/scaler are loaded and handled
    rate limiting -- this only does feature prep + inference."""
    df = pd.DataFrame([transaction.dict()])
    df["Hour"] = (df["Time"] % 86400) // 3600
    df["Amount_log"] = np.log1p(df["Amount"])
    df = df[feature_names]

    df_scaled = scaler.transform(df)

    with PredictionTimer():
        fraud_proba = model.predict_proba(df_scaled)[0, 1]
        is_fraud = fraud_proba >= threshold

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
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(
    transaction: TransactionRequest, request: Request, _: None = Depends(require_auth)
):
    """Predict fraud for a single transaction"""
    # Rate limiting
    client_ip = request.client.host
    check_rate_limit(client_ip)

    with APITimer("predict"):
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return _score_transaction(transaction)


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch: BatchPredictionRequest, request: Request, _: None = Depends(require_auth)
):
    """Predict fraud for a batch of transactions (up to 100 per request, see
    BatchPredictionRequest). One rate-limit check per call, not per
    transaction, and one HTTP round trip instead of N."""
    client_ip = request.client.host
    check_rate_limit(client_ip)

    with APITimer("predict_batch"):
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        predictions = [_score_transaction(t) for t in batch.transactions]
        fraud_count = sum(1 for p in predictions if p.is_fraud)

        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            fraud_count=fraud_count,
            timestamp=datetime.utcnow().isoformat(),
        )


@app.get("/api/v1/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information"""
    return ModelInfoResponse(
        model_version=settings.MODEL_VERSION,
        model_type=model_type,
        features=feature_names if feature_names else [],
        threshold=threshold,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return get_metrics()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    return JSONResponse(status_code=500, content={"detail": str(exc)})
