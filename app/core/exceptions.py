"""
Custom exceptions for Fraud Detection System
"""
from typing import Optional
from fastapi import HTTPException, status


class FraudDetectionException(Exception):
    """Base exception for fraud detection errors"""
    def __init__(self, message: str, error_code: str, status_code: int = 500):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(message)


class DataValidationError(FraudDetectionException):
    def __init__(self, message: str):
        super().__init__(message, "DATA_VALIDATION_ERROR", 422)


class PredictionError(FraudDetectionException):
    def __init__(self, message: str):
        super().__init__(message, "PREDICTION_ERROR", 500)


class ModelLoadError(FraudDetectionException):
    def __init__(self, message: str):
        super().__init__(message, "MODEL_LOAD_ERROR", 503)


class TrainingError(FraudDetectionException):
    def __init__(self, message: str):
        super().__init__(message, "TRAINING_ERROR", 500)


def to_http_exception(exc: FraudDetectionException) -> HTTPException:
    """Convert custom exception to HTTPException"""
    return HTTPException(
        status_code=exc.status_code,
        detail={
            "error_code": exc.error_code,
            "message": exc.message
        }
    )
