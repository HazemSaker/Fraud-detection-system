# API Documentation - Fraud Detection System

This document provides comprehensive documentation for the Fraud Detection API endpoints.

## Base URL

```
http://localhost:8000
```

## Authentication

Authentication is implemented using JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer <token>
```

## Endpoints

### 1. Health Check

Check the health status of the API and model.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "model_version": "latest",
  "uptime_seconds": 123.45,
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

### 2. Single Prediction

Predict fraud for a single transaction.

**Endpoint**: `POST /api/v1/predict`

**Request Body**:
```json
{
  "Time": 0.0,
  "Amount": 149.62,
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536347,
  "V4": 1.378155,
  "V5": -0.338321,
  "V6": 0.462388,
  "V7": 0.239599,
  "V8": 0.098698,
  "V9": 0.363787,
  "V10": 0.090794,
  "V11": -0.551600,
  "V12": -0.617801,
  "V13": -0.991390,
  "V14": -0.347886,
  "V15": -0.905631,
  "V16": -0.683095,
  "V17": -0.683095,
  "V18": 0.014724,
  "V19": 0.234804,
  "V20": 0.321546,
  "V21": 0.036782,
  "V22": 0.025194,
  "V23": 0.028357,
  "V24": -0.011577,
  "V25": 0.014651,
  "V26": -0.023425,
  "V27": 0.012321,
  "V28": 0.003521,
  "transaction_id": "optional_transaction_id",
  "merchant_id": "optional_merchant_id"
}
```

**Response**:
```json
{
  "is_fraud": false,
  "fraud_probability": 0.1234,
  "confidence": "Very Low",
  "threshold": 0.708,
  "model_version": "latest",
  "prediction_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

**Confidence Levels**:
- Very High: ≥ 0.9
- High: 0.7 - 0.9
- Medium: 0.5 - 0.7
- Low: 0.3 - 0.5
- Very Low: < 0.3

### 3. Model Information

Get information about the loaded model.

**Endpoint**: `GET /api/v1/model/info`

**Response**:
```json
{
  "model_version": "latest",
  "model_type": "LightGBM",
  "features": ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Hour", "Amount_log"],
  "threshold": 0.708,
  "last_trained": "2024-01-01T12:00:00.000000"
}
```

### 4. Metrics

Get Prometheus metrics for monitoring.

**Endpoint**: `GET /metrics`

**Response**: Prometheus metrics in text format

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message"
}
```

### Common Error Codes

- **400 Bad Request**: Invalid input data
- **422 Unprocessable Entity**: Validation error
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error
- **503 Service Unavailable**: Model not loaded

## Rate Limiting

- **Default**: 100 requests per 60 seconds per IP address
- **Headers**: Rate limit information is included in response headers

## Example Usage

### cURL

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0.0,
    "Amount": 149.62,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.347886,
    "V15": -0.905631,
    "V16": -0.683095,
    "V17": -0.683095,
    "V18": 0.014724,
    "V19": 0.234804,
    "V20": 0.321546,
    "V21": 0.036782,
    "V22": 0.025194,
    "V23": 0.028357,
    "V24": -0.011577,
    "V25": 0.014651,
    "V26": -0.023425,
    "V27": 0.012321,
    "V28": 0.003521,
    "Amount": 149.62
  }'
```

### Python (requests)

```python
import requests

transaction = {
    "Time": 0.0,
    "Amount": 149.62,
    "V1": -1.359807,
    "V2": -0.072781,
    # ... other features
}

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json=transaction
)

result = response.json()
print(f"Is Fraud: {result['is_fraud']}")
print(f"Probability: {result['fraud_probability']}")
```

### JavaScript (fetch)

```javascript
const transaction = {
  Time: 0.0,
  Amount: 149.62,
  V1: -1.359807,
  V2: -0.072781,
  // ... other features
};

fetch('http://localhost:8000/api/v1/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(transaction)
})
  .then(response => response.json())
  .then(data => console.log(data));
```

## Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
