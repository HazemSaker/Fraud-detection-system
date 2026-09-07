"""
Model evaluation script for DVC pipeline
"""
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.pipelines.steps.evaluator import evaluate_model
from config.settings import settings


def evaluate_model_pipeline(
    model_path: str,
    scaler_path: str,
    test_data_path: str,
    output_path: str
):
    """
    Evaluate trained model on test set
    
    Args:
        model_path: Path to trained model
        scaler_path: Path to scaler
        test_data_path: Path to test data
        output_path: Path to save evaluation metrics
    """
    print("Loading model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print("Loading test data...")
    test_df = pd.read_csv(test_data_path)
    
    # Preprocess test data
    TARGET_COLUMN = settings.TARGET_COLUMN
    X_test = test_df.drop(columns=[TARGET_COLUMN]).copy()
    y_test = test_df[TARGET_COLUMN].copy()
    
    # Feature engineering
    if "Time" in X_test.columns:
        X_test["Hour"] = (X_test["Time"] % 86400) // 3600
        X_test = X_test.drop(columns=["Time"])
    
    if "Amount" in X_test.columns:
        X_test["Amount_log"] = np.log1p(X_test["Amount"])
    
    # Scale features
    feature_names = X_test.columns.tolist()
    X_test.loc[:, feature_names] = scaler.transform(X_test[feature_names])
    
    # Load threshold
    threshold_path = Path(settings.OUTPUTS_DIR) / "threshold.txt"
    with open(threshold_path, 'r') as f:
        threshold = float(f.read().strip())
    
    print("Evaluating model...")
    metrics = evaluate_model(
        model=model,
        X_test=X_test.values,
        y_test=y_test.values,
        model_name="final_model",
        threshold=threshold
    )
    
    # Save metrics
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Evaluation metrics saved to {output_file}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Business Cost: ${metrics['business_cost']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(settings.OUTPUTS_DIR / "best_model.pkl"))
    parser.add_argument("--scaler", default=str(settings.OUTPUTS_DIR / "scaler.pkl"))
    parser.add_argument("--test-data", default="data/processed/test.csv")
    parser.add_argument("--output", default=str(settings.OUTPUTS_DIR / "evaluation_metrics.json"))
    args = parser.parse_args()
    
    evaluate_model_pipeline(args.model, args.scaler, args.test_data, args.output)
