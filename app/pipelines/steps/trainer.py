"""
Model training functions
"""
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)


def train_multiple_models(X_train, y_train, X_val, y_val):
    """Train multiple baseline models"""
    models = {}
    thresholds = {}
    
    # XGBoost
    logger.info("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='aucpr'
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    models["xgboost"] = xgb_model
    thresholds["xgboost"] = 0.5
    
    # LightGBM
    logger.info("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
    models["lightgbm"] = lgb_model
    thresholds["lightgbm"] = 0.5
    
    # Random Forest
    logger.info("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf_model.fit(X_train, y_train)
    models["random_forest"] = rf_model
    thresholds["random_forest"] = 0.5
    
    # Logistic Regression
    logger.info("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
    lr_model.fit(X_train, y_train)
    models["logistic_regression"] = lr_model
    thresholds["logistic_regression"] = 0.5
    
    logger.info("All baseline models trained")
    return models, thresholds
