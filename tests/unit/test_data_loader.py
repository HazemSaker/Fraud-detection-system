"""
Unit tests for data loader module
"""
import pytest
import pandas as pd
import tempfile
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.pipelines.steps.data_loader import load_data, split_data, validate_data_quality


def test_load_data(sample_data):
    """Test data loading functionality"""
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        # Load data
        df = load_data(temp_path)
        
        # Assertions
        assert len(df) == 1000
        assert 'Class' in df.columns
        assert 'Time' in df.columns
        assert 'Amount' in df.columns
    finally:
        # Cleanup
        Path(temp_path).unlink()


def test_load_data_file_not_found():
    """Test error handling when file doesn't exist"""
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent_file.csv")


def test_split_data(sample_data):
    """Test data splitting functionality"""
    train_df, val_df, test_df = split_data(sample_data, test_size=0.2, val_size=0.1)
    
    # Check that all data is accounted for
    total_samples = len(sample_data)
    assert len(train_df) + len(val_df) + len(test_df) == total_samples
    
    # Check that all splits have the required columns
    for split in [train_df, val_df, test_df]:
        assert 'Class' in split.columns
        assert len(split) > 0


def test_validate_data_quality(sample_data):
    """Test data quality validation"""
    metrics = validate_data_quality(sample_data)
    
    # Check that metrics are returned
    assert 'total_rows' in metrics
    assert 'total_columns' in metrics
    assert 'null_values' in metrics
    assert 'duplicate_rows' in metrics
    assert 'fraud_ratio' in metrics
    
    # Check values
    assert metrics['total_rows'] == 1000
    assert metrics['null_values'] == 0
    assert 0 <= metrics['fraud_ratio'] <= 1
