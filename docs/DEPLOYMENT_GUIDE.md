# Deployment Guide - Fraud Detection System

This guide provides step-by-step instructions for deploying the fraud detection system to production.

## Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Git
- Cloud provider account (AWS, GCP, or Azure) - optional
- Domain name (optional)

## Deployment Options

### Option 1: Docker Compose (Local/On-Premises)

#### 1. Build and Start Services

```bash
docker-compose up --build
```

This starts:
- Fraud Detection API (port 8000)
- Prometheus (port 9090)
- Grafana (port 3000)

#### 2. Access Services

- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

#### 3. Stop Services

```bash
docker-compose down
```

### Option 2: Docker (Single Container)

#### 1. Build Docker Image

```bash
docker build -t fraud-detection-api .
```

#### 2. Run Container

```bash
docker run -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -e MODEL_PATH=/app/outputs/best_model.pkl \
  fraud-detection-api
```

### Option 3: Cloud Deployment

#### AWS Deployment

**Using AWS ECS**

1. **Push to Docker Hub or ECR**

```bash
# Tag for ECR
docker tag fraud-detection-api:latest <your-ecr-repo-url>
docker push <your-ecr-repo-url>
```

2. **Create ECS Task Definition**

```json
{
  "family": "fraud-detection-api",
  "containerDefinitions": [
    {
      "name": "fraud-api",
      "image": "<your-ecr-repo-url>",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "memory": 512,
      "cpu": 256
    }
  ]
}
```

3. **Deploy to ECS**

```bash
aws ecs run-task --cluster fraud-detection --task-definition fraud-detection-api
```

#### GCP Deployment

**Using Google Cloud Run**

1. **Build and Push to GCR**

```bash
gcloud builds submit --tag gcr.io/<project-id>/fraud-detection-api
```

2. **Deploy to Cloud Run**

```bash
gcloud run deploy fraud-detection-api \
  --image gcr.io/<project-id>/fraud-detection-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure Deployment

**Using Azure Container Instances**

```bash
az container create \
  --resource-group fraud-detection-rg \
  --name fraud-api \
  --image <your-registry>/fraud-detection-api \
  --cpu 1 \
  --memory 2 \
  --ports 8000
```

## Configuration

### Environment Variables

Set these environment variables for production:

```bash
# Application
APP_NAME=Fraud Detection API
APP_VERSION=2.0.0
ENVIRONMENT=production
DEBUG=false

# API
API_HOST=0.0.0.0
API_PORT=8000

# Model
MODEL_PATH=/app/outputs/best_model.pkl
SCALER_PATH=/app/outputs/scaler.pkl
THRESHOLD=0.708

# Security
SECRET_KEY=<your-secret-key>

# Rate Limiting
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_PERIOD=60
```

### SSL/TLS Configuration

For production, use SSL/TLS:

**Using Nginx Reverse Proxy**

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring Setup

### Prometheus Configuration

Update `monitoring/prometheus.yml` for production:

```yaml
global:
  scrape_interval: 15s
  external_labels:
    cluster: 'fraud-detection-prod'

scrape_configs:
  - job_name: 'fraud-detection-api'
    static_configs:
      - targets: ['fraud-api:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboards

1. Access Grafana at http://localhost:3000
2. Add Prometheus as data source
3. Import dashboards from Grafana.com
4. Create custom dashboards for:
   - Prediction latency
   - Fraud detection rate
   - API request rate
   - Error rate

## Scaling

### Horizontal Scaling

**Using Kubernetes**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-api
  template:
    metadata:
      labels:
        app: fraud-api
    spec:
      containers:
      - name: fraud-api
        image: fraud-detection-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### Load Balancing

**Using Nginx**

```nginx
upstream fraud_api {
    least_conn;
    server fraud-api-1:8000;
    server fraud-api-2:8000;
    server fraud-api-3:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://fraud_api;
    }
}
```

## Security Best Practices

1. **Use HTTPS** in production
2. **Rotate secrets** regularly
3. **Implement rate limiting** to prevent abuse
4. **Use firewall rules** to restrict access
5. **Enable audit logging** for compliance
6. **Regular security updates** for dependencies
7. **Network isolation** for database access

## Backup and Recovery

### Model Artifacts Backup

```bash
# Backup model artifacts
tar -czf model-backup-$(date +%Y%m%d).tar.gz outputs/

# Upload to S3
aws s3 cp model-backup-$(date +%Y%m%d).tar.gz s3://your-backup-bucket/
```

### Database Backup

```bash
# Backup MLflow database
sqlite3 mlflow.db .dump > mlflow-backup-$(date +%Y%m%d).sql
```

## Troubleshooting

### Common Issues

**Model not loading**
- Check MODEL_PATH environment variable
- Verify model file exists
- Check logs for error messages

**High latency**
- Check system resources (CPU, memory)
- Review model size and complexity
- Consider model optimization or quantization

**Memory issues**
- Increase container memory limits
- Implement batch processing
- Use model pruning

## Performance Optimization

### Model Optimization

```python
# Reduce model size
import joblib
joblib.dump(model, "best_model.pkl", compress=3)

# Use quantization
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(model, method='isotonic')
```

### API Optimization

```python
# Enable caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_prediction(features):
    return model.predict_proba(features)
```

## Maintenance

### Regular Tasks

1. **Model Retraining**: Weekly or monthly
2. **Performance Monitoring**: Daily
3. **Log Analysis**: Weekly
4. **Security Updates**: Monthly
5. **Dependency Updates**: Monthly

### Model Drift Detection

Monitor for:
- Decreasing prediction confidence
- Changing feature distributions
- Increasing false positive rates
- Business metric degradation

## Support

For issues or questions:
- Check logs in `logs/app.log`
- Review Prometheus metrics
- Consult documentation
- Create issue in repository
