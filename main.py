"""
Main training pipeline orchestrator for Fraud Detection System
"""
import mlflow
import logging
import json
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

from config.settings import settings
from app.core.logging_config import get_logger, setup_logging
from app.core.exceptions import TrainingError


class NumpyJSONEncoder(json.JSONEncoder):
    """evaluator.py's business_cost is computed from raw numpy int64 values
    (fp * 5 + fn * 50) before the *other* confusion-matrix fields get cast
    to native int, so it slips through as numpy.int64 -- which json.dump()
    doesn't know how to serialize. Handling numpy scalar types generically
    here (rather than casting business_cost by hand in evaluator.py) also
    covers any other numpy type that ends up in metadata later."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from app.pipelines.steps import (
    load_data,
    split_data,
    validate_data_quality,
    preprocess_data,
    train_multiple_models,
    evaluate_all_models,
    tune_hyperparameters,
    generate_explanations
)

logger = get_logger(__name__)


def run_fraud_detection_pipeline(
    run_tuning: bool = True,
    n_trials: int = None,
    n_fraud_samples: int = None
) -> dict:
    """
    Run the complete fraud detection pipeline
    
    Args:
        run_tuning: Whether to run hyperparameter tuning
        n_trials: Number of Optuna trials
        n_fraud_samples: Number of fraud samples for explainability
        
    Returns:
        Dictionary with pipeline results and metrics
    """
    if n_trials is None:
        n_trials = settings.N_TRIALS
    if n_fraud_samples is None:
        n_fraud_samples = settings.N_FRAUD_SAMPLES
    
    # Setup MLflow
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    
    logger.info("=" * 80)
    logger.info("STARTING FRAUD DETECTION PIPELINE")
    logger.info("=" * 80)
    
    with mlflow.start_run():
        start_time = datetime.now()
        
        try:
            # Step 1: Load data
            logger.info("=" * 80)
            logger.info("STEP 1: Loading data...")
            logger.info("=" * 80)
            df = load_data(settings.DATA_PATH)
            
            # Step 2: Validate data quality
            logger.info("=" * 80)
            logger.info("STEP 2: Validating data quality...")
            logger.info("=" * 80)
            quality_metrics = validate_data_quality(df)
            mlflow.log_params(quality_metrics)
            
            # Step 3: Split data
            logger.info("=" * 80)
            logger.info("STEP 3: Splitting data...")
            logger.info("=" * 80)
            train_df, val_df, test_df = split_data(df)
            
            # Step 4: Preprocess data
            logger.info("=" * 80)
            logger.info("STEP 4: Preprocessing data...")
            logger.info("=" * 80)
            X_train, y_train, X_val, y_val, X_test, y_test, feature_names, scaler = preprocess_data(
                train_df, val_df, test_df
            )
            
            # Step 5: Train baseline models
            logger.info("=" * 80)
            logger.info("STEP 5: Training baseline models...")
            logger.info("=" * 80)
            models, thresholds = train_multiple_models(X_train, y_train, X_val, y_val)
            
            # Step 6: Hyperparameter tuning
            if run_tuning:
                logger.info("=" * 80)
                logger.info("STEP 6: Running hyperparameter tuning...")
                logger.info("=" * 80)
                best_params, best_score, tuned_model, optimal_threshold = tune_hyperparameters(
                    X_train, y_train, X_val, y_val, n_trials=n_trials
                )
                mlflow.log_params(best_params)
                # Add tuned model to models dictionary for evaluation
                models["xgboost_tuned"] = tuned_model
                thresholds["xgboost_tuned"] = optimal_threshold
            
            # Step 7: Evaluate models
            logger.info("=" * 80)
            logger.info("STEP 7: Evaluating models...")
            logger.info("=" * 80)
            all_metrics, best_model_name = evaluate_all_models(models, X_test, y_test, thresholds)
            
            # Log metrics
            for model_name, metrics in all_metrics.items():
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)
            
            logger.info(f"Best Model: {best_model_name}")
            logger.info(f"Best PR-AUC: {all_metrics[best_model_name]['pr_auc']:.4f}")
            logger.info(f"Best F1-Score: {all_metrics[best_model_name]['f1_score']:.4f}")
            logger.info(f"Business Cost: ${all_metrics[best_model_name]['business_cost']:.2f}")
            
            # Step 8: Generate explainability
            logger.info("=" * 80)
            logger.info("STEP 8: Generating explainability reports...")
            logger.info("=" * 80)
            generate_explanations(models[best_model_name], X_test, feature_names, n_fraud_samples)
            
            # Step 9: Save model and artifacts
            logger.info("=" * 80)
            logger.info("STEP 9: Saving model and artifacts...")
            logger.info("=" * 80)
            
            # Create outputs directory
            settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Save best model
            best_model = models[best_model_name]
            model_path = settings.OUTPUTS_DIR / "best_model.pkl"
            joblib.dump(best_model, model_path)
            mlflow.log_artifact(str(model_path))
            
            # Save scaler
            scaler_path = settings.OUTPUTS_DIR / "scaler.pkl"
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(str(scaler_path))
            
            # Save threshold
            threshold_path = settings.OUTPUTS_DIR / "threshold.txt"
            with open(threshold_path, 'w') as f:
                f.write(str(thresholds[best_model_name]))
            mlflow.log_artifact(str(threshold_path))

            # Save metadata (model name/class + threshold + metrics together).
            # This is what the API reads at startup instead of guessing the
            # model type and instead of trusting a possibly-stale THRESHOLD
            # in .env -- previously those two things could silently drift
            # apart from whatever best_model.pkl actually was.
            metadata = {
                "model_name": best_model_name,
                "model_class": type(best_model).__name__,
                "threshold": thresholds[best_model_name],
                "pr_auc": all_metrics[best_model_name]["pr_auc"],
                "f1_score": all_metrics[best_model_name]["f1_score"],
                "business_cost": all_metrics[best_model_name]["business_cost"],
                "trained_at": datetime.now().isoformat(),
            }
            metadata_path = settings.OUTPUTS_DIR / "model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, cls=NumpyJSONEncoder)
            mlflow.log_artifact(str(metadata_path))

            logger.info(f"Model saved to {model_path}")
            logger.info(f"Scaler saved to {scaler_path}")
            logger.info(f"Threshold saved to {threshold_path}")
            logger.info(f"Metadata saved to {metadata_path}")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Total duration: {duration:.2f} seconds")
            logger.info(f"Best model: {best_model_name}")
            logger.info(f"Best PR-AUC: {all_metrics[best_model_name]['pr_auc']:.4f}")
            logger.info(f"Best F1-Score: {all_metrics[best_model_name]['f1_score']:.4f}")
            logger.info(f"Business Cost: ${all_metrics[best_model_name]['business_cost']:.2f}")
            logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
            logger.info("Check MLflow UI for detailed metrics and artifacts")
            
            return {
                "status": "completed",
                "duration": duration,
                "best_model": best_model_name,
                "metrics": all_metrics[best_model_name],
                "mlflow_run_id": mlflow.active_run().info.run_id
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise TrainingError(f"Pipeline failed: {str(e)}")


if __name__ == "__main__":
    setup_logging()
    results = run_fraud_detection_pipeline()
    
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Status: {results['status']}")
    print(f"Duration: {results['duration']:.2f} seconds")
    print(f"Best Model: {results['best_model']}")
    print(f"Best PR-AUC: {results['metrics']['pr_auc']:.4f}")
    print(f"Best F1-Score: {results['metrics']['f1_score']:.4f}")
    print(f"Business Cost: ${results['metrics']['business_cost']:.2f}")
    print("=" * 80)
