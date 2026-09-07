"""
Pytest configuration and fixtures
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_data():
    """Create sample transaction data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Time': np.random.uniform(0, 172800, n_samples),
        'Amount': np.random.uniform(0, 1000, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
    }
    
    # Add V1-V28 features
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_transaction():
    """Create a single transaction for API testing"""
    return {
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
