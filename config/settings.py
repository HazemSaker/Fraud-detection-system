"""
Configuration management for Fraud Detection System
"""
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Any
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "Fraud Detection API"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    
    @field_validator('DEBUG', mode='before')
    @classmethod
    def parse_debug(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return bool(v)
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # Model Configuration
    MODEL_PATH: str = "outputs/best_model.pkl"
    SCALER_PATH: str = "outputs/scaler.pkl"
    THRESHOLD: float = 0.5
    MODEL_VERSION: str = "latest"
    
    # Data Configuration
    DATA_PATH: str = "data/raw/creditcard.csv"
    TARGET_COLUMN: str = "Class"
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.1
    RANDOM_STATE: int = 42
    
    # ML Configuration
    N_TRIALS: int = 50
    N_FRAUD_SAMPLES: int = 100
    SMOTE_RATIO: float = 0.1
    K_NEIGHBORS: int = 5
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "fraud_detection_pipeline"
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "app.log"
    
    # Security Configuration
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    OUTPUTS_DIR: Path = BASE_DIR / "outputs"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "protected_namespaces": ()
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance"""
    return settings
