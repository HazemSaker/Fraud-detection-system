"""
Unit tests for preprocessor module
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.pipelines.steps.preprocessor import preprocess_data


def test_preprocess_data():
    """Test data preprocessing functionality"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    train_data = pd.DataFrame({
        'Time': np.random.uniform(0, 172800, n_samples),
        'Amount': np.random.uniform(0, 1000, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
    })
    
    for i in range(1, 29):
        train_data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    val_data = train_data.copy()
    test_data = train_data.copy()
    
    # Preprocess
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, scaler = preprocess_data(
        train_data, val_data, test_data
    )
    
    # Check shapes
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    
    # Check feature names
    assert len(feature_names) > 0
    assert 'Hour' in feature_names
    assert 'Amount_log' in feature_names
    
    # Check scaler
    assert scaler is not None
