"""
Model evaluation functions
"""
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, model_name, threshold=0.5):
    """Evaluate a single model"""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Metrics
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Business cost (FP: $5, FN: $50)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    business_cost = fp * 5 + fn * 50
    
    metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "recall": rec,
        "precision": prec,
        "f1_score": f1,
        "business_cost": business_cost,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "threshold": threshold
    }
    
    logger.info(f"{model_name} metrics: {metrics}")
    return metrics


def evaluate_all_models(models, X_test, y_test, thresholds):
    """Evaluate all models"""
    all_metrics = {}
    for name, model in models.items():
        threshold = thresholds.get(name, 0.5)
        metrics = evaluate_model(model, X_test, y_test, name, threshold)
        all_metrics[name] = metrics
    
    # Select best model by PR-AUC
    best_model_name = max(all_metrics, key=lambda x: all_metrics[x]["pr_auc"])
    return all_metrics, best_model_name
