# Fraud Detection System - Production Edition

A production-ready machine learning pipeline for real-time credit card fraud detection with comprehensive monitoring, explainability, and deployment capabilities.

## Features

### Machine Learning Pipeline
- **Multi-model Training**: XGBoost, LightGBM, Random Forest, Logistic Regression
- **Hyperparameter Optimization**: Optuna with TPE sampler and median pruning
- **Class Imbalance Handling**: SMOTE, undersampling, and hybrid strategies
- **Early Stopping**: Prevents overfitting during training
- **Optimal Threshold Selection**: F1-score based threshold optimization
- **Explainability**: SHAP (global/local) and LIME (local) explanations
- **Experiment Tracking**: MLflow with SQLite backend

### Production API
- **FastAPI Framework**: High-performance async API
- **Data Validation**: Pydantic schemas with comprehensive validation
- **Authentication**: JWT-based, enforced by default (`ENABLE_AUTH=true`); toggle off for local testing
- **Rate Limiting**: In-memory rate limiter for API protection
- **Monitoring**: Prometheus metrics integration
- **Health Checks**: Comprehensive health check endpoints
- **Batch Processing**: `/api/v1/predict/batch`, up to 100 transactions per request
- **Error Handling**: Structured error responses with proper HTTP status codes

### DevOps & Infrastructure
- **Docker Support**: Multi-stage Dockerfile for optimized images
- **Docker Compose**: Complete stack with Prometheus and Grafana
- **CI/CD Pipeline**: GitHub Actions workflow for testing and deployment
- **Model Registry**: Version-controlled model management
- **DVC Integration**: Data version control and pipeline orchestration
- **Structured Logging**: JSON-formatted logs for production
- **Configuration Management**: Environment-based configuration with Pydantic

### Testing
- **Unit Tests**: Comprehensive unit test coverage
- **Integration Tests**: API endpoint testing
- **Test Coverage**: Pytest with coverage reporting

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training Pipeline](#training-pipeline)
- [API Deployment](#api-deployment)
- [Monitoring](#monitoring)
- [Testing](#testing)
- [API Documentation](#api-documentation)
- [Model Explainability](#model-explainability)
- [Performance Metrics](#performance-metrics)

## Installation

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)
- Git
- DVC (optional, for data version control)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/<your-username>/<new-repo-name>.git
cd <new-repo-name>
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Initialize DVC (optional but recommended)**
```bash
dvc init
dvc remote add -d local_storage ./dvc_storage
```

6. **Prepare data**
```bash
# Place your creditcard.csv in data/raw/
# Or use the Kaggle dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud

# Track data with DVC
dvc add data/raw/creditcard.csv
git add data/raw/creditcard.csv.dvc .gitignore
git commit -m "Add raw data"
```

## Quick Start

### Training the Model

**Option 1: Direct Python**
```bash
python main.py
```

**Option 2: Using DVC Pipeline**
```bash
# Prepare data
dvc repro prepare_data

# Train model
dvc repro train

# Evaluate model
dvc repro evaluate

# Or run entire pipeline
dvc repro
```

This will:
1. Load and validate the dataset
2. Split data into train/validation/test sets
3. Preprocess features (scaling, SMOTE)
4. Train baseline models
5. Perform hyperparameter optimization
6. Evaluate all models
7. Generate explainability reports
8. Save model artifacts

### Starting the API

```bash
# Option 1: Direct Python
uvicorn app.api.app:app --host 0.0.0.0 --port 8000

# Option 2: Docker Compose (includes monitoring)
docker-compose up --build
```

### Making Predictions

Authentication is on by default (`ENABLE_AUTH=true` in `.env`). Get a token first:

```bash
curl -X POST "http://localhost:8000/token" \
  -d "username=admin&password=changeme"
# -> {"access_token": "...", "token_type": "bearer"}
```

> Demo credentials only -- see `app/api/app.py`. Replace with a real user
> store, or set `ENABLE_AUTH=false` for local testing without a token.

Then use it on `/predict`:

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0.0,
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

Batch predictions (up to 100 transactions per request) work the same way,
against `/api/v1/predict/batch`, with a `{"transactions": [...]}` body.

## Project Structure

```
latest/
├── app/
│   ├── api/                    # FastAPI application
│   │   ├── app.py             # Main API application
│   │   ├── schemas.py         # Pydantic schemas
│   │   ├── auth.py            # Authentication logic
│   │   ├── rate_limiter.py    # Rate limiting
│   │   └── monitoring.py      # Prometheus metrics
│   ├── core/                   # Core functionality
│   │   ├── logging_config.py   # Structured logging
│   │   └── exceptions.py      # Custom exceptions
│   ├── pipelines/              # ML pipelines
│   │   ├── steps/             # Pipeline steps
│   │   │   ├── data_loader.py
│   │   │   ├── preprocessor.py
│   │   │   ├── trainer.py
│   │   │   ├── evaluator.py
│   │   │   ├── tuner.py
│   │   │   └── explainer.py
│   │   └── explainability/    # SHAP/LIME
│   │       ├── shap_explainer.py
│   │       └── lime_explainer.py
│   ├── services/               # Business services
│   │   └── model_registry.py  # Model versioning
│   └── models/                 # ML models
├── config/                     # Configuration
│   └── settings.py            # Application settings
├── tests/                      # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── conftest.py            # Pytest configuration
├── monitoring/                 # Monitoring configs
│   └── prometheus.yml         # Prometheus configuration
├── data/                       # Data directory
│   └── raw/                   # Raw datasets
├── outputs/                    # Model outputs
│   ├── explanations/          # SHAP/LIME reports
│   ├── comparisons/           # Model comparisons
│   └── optuna_plots/          # Optimization plots
├── logs/                       # Application logs
├── .github/                    # GitHub workflows
│   └── workflows/
│       └── ci-cd.yml          # CI/CD pipeline
├── main.py                     # Training pipeline
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image
├── docker-compose.yml          # Docker stack
├── .env.example               # Environment template
└── README.md                  # This file
```

## Configuration

Configuration is managed through environment variables and the `.env` file. Key configuration options:

### Application Settings
- `APP_NAME`: Application name
- `APP_VERSION`: Application version
- `ENVIRONMENT`: Environment (development/production)
- `DEBUG`: Debug mode

### API Settings
- `API_HOST`: API host address
- `API_PORT`: API port
- `API_PREFIX`: API URL prefix

### Model Settings
- `MODEL_PATH`: Path to trained model
- `SCALER_PATH`: Path to scaler
- `THRESHOLD`: Fraud detection threshold
- `MODEL_VERSION`: Model version identifier

### ML Settings
- `N_TRIALS`: Number of Optuna trials
- `N_FRAUD_SAMPLES`: Samples for explainability
- `SMOTE_RATIO`: Target ratio for SMOTE

### Security
- `SECRET_KEY`: JWT secret key
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration

## Training Pipeline

The training pipeline follows these steps:

1. **Data Loading**: Load and validate the dataset
2. **Data Splitting**: Stratified split into train/val/test
3. **Preprocessing**: Feature scaling, SMOTE, feature engineering
4. **Baseline Training**: Train multiple baseline models
5. **Hyperparameter Tuning**: Optimize with Optuna
6. **Model Evaluation**: Comprehensive evaluation on test set
7. **Explainability**: Generate SHAP and LIME reports
8. **Artifact Saving**: Save model, scaler, and metadata

### Running with Custom Parameters

```python
from main import run_fraud_detection_pipeline

results = run_fraud_detection_pipeline(
    run_tuning=True,
    n_trials=100,
    n_fraud_samples=200
)
```

## API Deployment

### Docker Deployment

```bash
# Build image
docker build -t fraud-detection-api .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -e MODEL_PATH=/app/outputs/best_model.pkl \
  fraud-detection-api
```

### Docker Compose (Recommended)

```bash
docker-compose up --build
```

This starts:
- Fraud Detection API (port 8000)
- Prometheus (port 9090)
- Grafana (port 3000)

## Monitoring

### Prometheus Metrics

The API exposes Prometheus metrics at `/metrics`:

- `fraud_detection_prediction_requests_total`: Total prediction requests
- `fraud_detection_prediction_duration_seconds`: Prediction latency
- `fraud_detection_fraud_predictions_total`: Fraud predictions by confidence
- `fraud_detection_api_requests_total`: Total API requests
- `fraud_detection_api_duration_seconds`: API latency

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/admin)

## Testing

### Run All Tests

```bash
pytest tests/ -v --cov=app --cov-report=html
```

### Run Unit Tests Only

```bash
pytest tests/unit/ -v
```

### Run Integration Tests Only

```bash
pytest tests/integration/ -v
```

## API Documentation

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Key Endpoints

#### Health Check
```
GET /health
```

#### Get Auth Token
```
POST /token
Content-Type: application/x-www-form-urlencoded
Body: username=admin&password=changeme
```

#### Single Prediction
```
POST /api/v1/predict
Content-Type: application/json
Authorization: Bearer <token>   (unless ENABLE_AUTH=false)
```

#### Batch Prediction
```
POST /api/v1/predict/batch
Content-Type: application/json
Authorization: Bearer <token>   (unless ENABLE_AUTH=false)
Body: {"transactions": [...]}   (max 100 per request)
```

#### Model Info
```
GET /api/v1/model/info
```

#### Metrics
```
GET /metrics
```

## Model Explainability

### SHAP Explanations

- **Global**: Feature importance across all predictions
- **Local**: Individual prediction explanations
- **Visualizations**: Summary plots, waterfall plots

### LIME Explanations

- **Local**: Linear approximations for individual predictions
- **Top Features**: Most influential features per prediction

### Reports

HTML reports are generated in `outputs/explanations/`:
- `shap_summary.png`: SHAP analysis report
- `lime_explanation_*.html`: LIME analysis report

## Performance Metrics

### Evaluation Metrics

- **PR-AUC**: Precision-Recall AUC (primary metric for imbalanced data)
- **ROC-AUC**: Receiver Operating Characteristic AUC
- **Recall**: True positive rate
- **Precision**: Positive predictive value
- **F1-Score**: Harmonic mean of precision and recall
- **Business Cost**: Custom cost function (FP: $5, FN: $50)

### Verified Performance

Results from the reproducible pipeline (seeded Optuna hyperparameter search,
so these numbers reproduce run-to-run) on the credit card fraud dataset
(284,807 transactions, 492 fraud cases, 0.17% fraud ratio; 56,962 in the
held-out test set, 99 of them fraud):

| Model | PR-AUC | ROC-AUC | Recall | Precision | F1-Score | Business Cost | Threshold |
|-------|--------|---------|--------|-----------|----------|---------------|-----------|
| **XGBoost (Tuned)** (Best) | 0.8352 | 0.9668 | 0.7980 | 0.9186 | 0.8541 | $1035 | 0.70 |
| LightGBM | 0.8318 | 0.9666 | 0.8081 | 0.9091 | 0.8556 | $990 | 0.75 |
| Random Forest | 0.8316 | 0.9480 | 0.7475 | 0.9610 | 0.8409 | $1265 | 0.65 |
| XGBoost (Baseline) | 0.8096 | 0.9642 | 0.7778 | 0.8953 | 0.8324 | $1145 | 0.80 |
| Logistic Regression | 0.7410 | 0.9688 | 0.8485 | 0.1949 | 0.3170 | $2485 | 0.85 |

**Best Model (selected automatically by PR-AUC): XGBoost, tuned**
- Hyperparameter search: Optuna, 50 trials, TPE sampler (seeded, reproducible)
- Threshold: 0.70 (optimized for F1-score)
- Business Cost: $1035 (FP: $5, FN: $50)

Worth noting: the pipeline selects "best" by PR-AUC, where XGBoost-tuned
and LightGBM are within 0.003 of each other -- but LightGBM actually has a
lower business cost ($990 vs $1035), since the cost function penalizes
false negatives more heavily than PR-AUC alone reflects. Both are
reasonable picks depending on what you're optimizing for.

## Security

### Authentication

JWT-based authentication, enforced by default. `POST /token` with the demo
credentials (see `app/api/app.py` -- replace before deploying anywhere but
your own machine) returns a bearer token; `/predict` and `/predict/batch`
require it. Set `ENABLE_AUTH=false` in `.env` to disable enforcement for
local testing without a token.

### Rate Limiting

Default: 100 requests per 60 seconds per IP address.

### Data Validation

All inputs are validated using Pydantic schemas with type checking and range validation.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

This project is for educational and demonstration purposes.

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.
