# Quick Start Guide - Fraud Detection System

Get the fraud detection system up and running in minutes.

## Prerequisites

- Python 3.10+
- pip package manager
- Credit card fraud dataset (optional, for training)

## Installation

### 1. Clone and Navigate

```bash
cd "c:/Users/Asus/vs code files/Fraud detection system/latest"
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` if needed (defaults should work for most cases).

## Training the Model

### Option 1: Quick Training (Recommended)

```bash
python main.py
```

This will:
- Load and validate data
- Train 4 models (XGBoost, LightGBM, Random Forest, Logistic Regression)
- Perform hyperparameter optimization
- Select the best model
- Save model artifacts

**Expected time**: 5-10 minutes

### Option 2: Custom Training

```python
from main import run_fraud_detection_pipeline

results = run_fraud_detection_pipeline(
    run_tuning=True,
    n_trials=100,  # More trials for better results
    n_fraud_samples=200  # More samples for explainability
)
```

## Starting the API

### 1. Start the Server

```bash
uvicorn app.api.app:app --host 0.0.0.0 --port 8000
```

### 2. Verify Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "model_version": "latest",
  "uptime_seconds": 0.0,
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

## Making Predictions

### Using cURL

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

### Using Python

```python
import requests

transaction = {
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
}

response = requests.post("http://localhost:8000/api/v1/predict", json=transaction)
result = response.json()

print(f"Is Fraud: {result['is_fraud']}")
print(f"Probability: {result['fraud_probability']:.4f}")
print(f"Confidence: {result['confidence']}")
```

## Interactive Documentation

Open your browser and navigate to:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive forms to test all API endpoints.

## Docker Deployment (Optional)

### Using Docker Compose

```bash
docker-compose up --build
```

This starts:
- Fraud Detection API (port 8000)
- Prometheus (port 9090)
- Grafana (port 3000)

### Using Docker

```bash
# Build image
docker build -t fraud-detection-api .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  fraud-detection-api
```

## Monitoring

### View Metrics

```bash
curl http://localhost:8000/metrics
```

### Access Grafana

1. Open http://localhost:3000
2. Login with admin/admin
3. Add Prometheus as data source
4. Create dashboards

## Troubleshooting

### Model not loading

**Error**: "Model not loaded"

**Solution**: 
- Ensure you've trained the model first: `python main.py`
- Check that `outputs/best_model.pkl` exists
- Verify MODEL_PATH in `.env`

### Import errors

**Error**: "Module not found"

**Solution**:
- Ensure virtual environment is activated
- Install dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.10+)

### Port already in use

**Error**: "Address already in use"

**Solution**:
- Change port in `.env`: `API_PORT=8001`
- Or stop the process using port 8000

## Next Steps

1. **Explore the code**: Read through the source code
2. **Modify parameters**: Experiment with different hyperparameters
3. **Add new models**: Implement additional ML algorithms
4. **Deploy to cloud**: Follow the deployment guide
5. **Set up monitoring**: Configure alerts and dashboards

## Getting Help

- Check the full documentation in the `docs/` folder
- Review the learning guide for in-depth explanations
- Check logs in `logs/app.log` for errors
- Create an issue for bugs or questions
