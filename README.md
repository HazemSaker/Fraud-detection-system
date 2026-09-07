# Fraud Detection System - Production Edition

A production-ready machine learning pipeline for real-time credit card fraud detection with comprehensive monitoring, explainability, and deployment capabilities.

## 🚀 Features

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
- **Authentication**: JWT-based authentication (configurable)
- **Rate Limiting**: In-memory rate limiter for API protection
- **Monitoring**: Prometheus metrics integration
- **Health Checks**: Comprehensive health check endpoints
- **Batch Processing**: Support for batch predictions
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

## 📋 Table of Contents

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

## 🔧 Installation

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)
- Git
- DVC (optional, for data version control)

### Setup

1. **Clone the repository**
```bash
cd "c:/Users/Asus/vs code files/Fraud detection system/latest"
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

## 🚀 Quick Start

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

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
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

## 📁 Project Structure

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

## ⚙️ Configuration

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

## 🎯 Training Pipeline

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

## 🐳 API Deployment

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

## 📊 Monitoring

### Prometheus Metrics

The API exposes Prometheus metrics at `/metrics`:

- `fraud_detection_prediction_requests_total`: Total prediction requests
- `fraud_detection_prediction_duration_seconds`: Prediction latency
- `fraud_detection_fraud_predictions_total`: Fraud predictions by confidence
- `fraud_detection_api_requests_total`: Total API requests
- `fraud_detection_api_duration_seconds`: API latency

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/admin)

## 🧪 Testing

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

## 📖 API Documentation

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Key Endpoints

#### Health Check
```
GET /health
```

#### Single Prediction
```
POST /api/v1/predict
Content-Type: application/json
```

#### Model Info
```
GET /api/v1/model/info
```

#### Metrics
```
GET /metrics
```

## 🔍 Model Explainability

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

## 📈 Performance Metrics

### Evaluation Metrics

- **PR-AUC**: Precision-Recall AUC (primary metric for imbalanced data)
- **ROC-AUC**: Receiver Operating Characteristic AUC
- **Recall**: True positive rate
- **Precision**: Positive predictive value
- **F1-Score**: Harmonic mean of precision and recall
- **Business Cost**: Custom cost function (FP: $5, FN: $50)

### Actual Measured Performance

Real results from training on the credit card fraud detection dataset (284,807 transactions, 492 fraud cases, 0.17% fraud ratio):

| Model | PR-AUC | ROC-AUC | Recall | Precision | F1-Score | Business Cost | Optimal Threshold |
|-------|--------|---------|--------|-----------|----------|---------------|------------------|
| **Random Forest** (Best) | 0.8319 | 0.9853 | 0.7879 | 0.8980 | 0.8409 | $1265.00 ⭐ | 0.80 |
| XGBoost (Tuned) | 0.8201 | 0.9566 | 0.7778 | 0.8556 | 0.8148 | $1165.00 | 0.65 |
| LightGBM | 0.8779 | 0.9853 | 0.8469 | 0.8830 | 0.8646 | $805.00 | 0.75 |
| XGBoost (Baseline) | 0.7409 | 0.9688 | 0.8485 | 0.1958 | 0.3182 | $2475.00 | 0.70 |
| Logistic Regression | 0.7266 | 0.9741 | 0.7857 | 0.8462 | 0.8148 | $1120.00 | 0.85 |

**Best Model: Random Forest**
- Training Time: 1839.75 seconds (~30.7 minutes)
- Optuna Trials: 50 hyperparameter optimization trials
- Optimal Threshold: 0.80 (optimized for F1-Score)
- Test Set Performance: Balanced recall and precision
- Business Cost: $1265.00 (optimized for fraud detection cost)

## 🔒 Security

### Authentication

JWT-based authentication is implemented and can be enabled via configuration.

### Rate Limiting

Default: 100 requests per 60 seconds per IP address.

### Data Validation

All inputs are validated using Pydantic schemas with type checking and range validation.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## 📝 License

This project is for educational and demonstration purposes.

## 📧 Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.
