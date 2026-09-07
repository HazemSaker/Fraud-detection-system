"""
Data loading and validation functions
"""
import pandas as pd
import logging
from pathlib import Path
from config.settings import settings

logger = logging.getLogger(__name__)


def load_data(data_path: str) -> pd.DataFrame:
    """Load and validate data from CSV file"""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, stratify: bool = True):
    """Split data into train/val/test sets"""
    from sklearn.model_selection import train_test_split
    
    target_col = settings.TARGET_COLUMN
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if stratify:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=settings.RANDOM_STATE, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(test_size / (test_size + val_size)), random_state=settings.RANDOM_STATE, stratify=y_temp
        )
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=settings.RANDOM_STATE
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(test_size / (test_size + val_size)), random_state=settings.RANDOM_STATE
        )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    logger.info(f"Split data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df


def validate_data_quality(df: pd.DataFrame) -> dict:
    """Validate data quality and return metrics"""
    metrics = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "null_values": df.isnull().sum().sum(),
        "duplicate_rows": df.duplicated().sum(),
        "fraud_ratio": df[settings.TARGET_COLUMN].mean()
    }
    logger.info(f"Data quality metrics: {metrics}")
    return metrics
