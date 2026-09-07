"""
Hyperparameter tuning with Optuna
"""
import optuna
import xgboost as xgb
from sklearn.metrics import average_precision_score
import logging

logger = logging.getLogger(__name__)


def tune_hyperparameters(X_train, y_train, X_val, y_val, model_type="xgboost", n_trials=50):
    """Tune hyperparameters using Optuna"""
    def objective(trial):
        if model_type == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            model = xgb.XGBClassifier(**params, random_state=42, eval_metric='aucpr')
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
            y_proba = model.predict_proba(X_val)[:, 1]
            return average_precision_score(y_val, y_proba)
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best trial: {study.best_trial.params}")
    logger.info(f"Best score: {study.best_value}")
    
    # Train final model with best parameters on full training data
    best_params = study.best_trial.params
    best_model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='aucpr')
    best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    
    return best_params, study.best_value, best_model
