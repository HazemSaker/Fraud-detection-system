"""
Data preparation script for DVC pipeline
"""
import pandas as pd
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.pipelines.steps import load_data, split_data, validate_data_quality
from config.settings import settings


def prepare_data(input_path: str, output_dir: str):
    """
    Prepare and split data for training
    
    Args:
        input_path: Path to raw data
        output_dir: Directory to save processed data
    """
    print("Loading raw data...")
    df = load_data(input_path)
    
    print("Validating data quality...")
    metrics = validate_data_quality(df)
    print(f"Data quality metrics: {metrics}")
    
    print("Splitting data...")
    train_df, val_df, test_df = split_data(df)
    
    # Save processed data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    print(f"Data saved to {output_dir}")
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=settings.DATA_PATH, help="Input data path")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    args = parser.parse_args()
    
    prepare_data(args.input, args.output)
