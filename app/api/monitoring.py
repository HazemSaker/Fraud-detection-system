"""
Prometheus metrics implementation
"""
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response
import time

# Metrics
prediction_requests = Counter('fraud_detection_prediction_requests_total', 'Total prediction requests')
prediction_duration = Histogram('fraud_detection_prediction_duration_seconds', 'Prediction latency')
fraud_predictions = Counter('fraud_detection_fraud_predictions_total', 'Fraud predictions', ['confidence'])
api_requests = Counter('fraud_detection_api_requests_total', 'Total API requests', ['endpoint'])
api_duration = Histogram('fraud_detection_api_duration_seconds', 'API latency')


def record_prediction(confidence: str):
    """Record a fraud prediction"""
    fraud_predictions.labels(confidence=confidence).inc()


def record_api_request(endpoint: str):
    """Record an API request"""
    api_requests.labels(endpoint=endpoint).inc()


class PredictionTimer:
    """Context manager for timing predictions"""
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        duration = time.time() - self.start
        prediction_duration.observe(duration)


class APITimer:
    """Context manager for timing API requests"""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        duration = time.time() - self.start
        api_duration.observe(duration)
        record_api_request(self.endpoint)


def get_metrics():
    """Get Prometheus metrics"""
    return Response(generate_latest(), media_type="text/plain")
