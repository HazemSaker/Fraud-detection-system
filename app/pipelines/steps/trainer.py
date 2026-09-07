"""
Model training functions
"""
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np
import logging

logger = logging.getLogger(__name__)


def find_optimal_threshold(y_val, y_proba, metric='f1'):
    """Find optimal threshold based on validation set"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(y_val, y_pred)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    logger.info(f"Optimal threshold: {best_threshold:.2f} (F1: {best_score:.4f})")
    return best_threshold


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
    y_proba = xgb_model.predict_proba(X_val)[:, 1]
    thresholds["xgboost"] = find_optimal_threshold(y_val, y_proba)
    
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
    y_proba = lgb_model.predict_proba(X_val)[:, 1]
    thresholds["lightgbm"] = find_optimal_threshold(y_val, y_proba)
    
    # Random Forest
    logger.info("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf_model.fit(X_train, y_train)
    models["random_forest"] = rf_model
    y_proba = rf_model.predict_proba(X_val)[:, 1]
    thresholds["random_forest"] = find_optimal_threshold(y_val, y_proba)
    
    # Logistic Regression
    logger.info("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
    lr_model.fit(X_train, y_train)
    models["logistic_regression"] = lr_model
    y_proba = lr_model.predict_proba(X_val)[:, 1]
    thresholds["logistic_regression"] = find_optimal_threshold(y_val, y_proba)
    
    logger.info("All baseline models trained")
    return models, thresholds
