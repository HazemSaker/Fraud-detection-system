"""
Model explainability functions
"""
import logging

logger = logging.getLogger(__name__)


def generate_explanations(model, X_test, feature_names, n_samples=100):
    """Generate SHAP and LIME explanations"""
    logger.info("Generating model explanations...")
    
    try:
        # SHAP explanations
        from app.pipelines.explainability.shap_explainer import generate_shap_explanations
        generate_shap_explanations(model, X_test, feature_names, n_samples)
        logger.info("SHAP explanations generated")
    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {e}")
    
    try:
        # LIME explanations
        from app.pipelines.explainability.lime_explainer import generate_lime_explanations
        generate_lime_explanations(model, X_test, feature_names, n_samples)
        logger.info("LIME explanations generated")
    except Exception as e:
        logger.error(f"Error generating LIME explanations: {e}")
    
    logger.info("Model explanations completed")
