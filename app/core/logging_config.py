"""
Structured logging configuration for Fraud Detection System
"""
import logging
import json
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from config.settings import settings


def setup_logging():
    """Setup structured JSON logging"""
    LOGS_DIR = settings.LOGS_DIR
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Ensure LOG_FILE doesn't include directory prefix
    log_file_name = settings.LOG_FILE
    if '/' in log_file_name or '\\' in log_file_name:
        log_file_name = log_file_name.split('/')[-1].split('\\')[-1]
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # JSON formatter
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_obj = {
                'timestamp': self.formatTime(record),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            if record.exc_info:
                log_obj['exception'] = self.formatException(record.exc_info)
            return json.dumps(log_obj)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JsonFormatter())
    logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file_path = LOGS_DIR / log_file_name
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)
