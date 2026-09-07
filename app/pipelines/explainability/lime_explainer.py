"""
LIME explainability implementation
"""
import lime.lime_tabular
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_lime_explanations(model, X_test, feature_names, n_samples=100):
    """Generate LIME explanations for local interpretability"""
    logger.info("Creating LIME explainer...")
    
    # Create explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_test,
        feature_names=feature_names,
        mode='classification'
    )
    
    # Explain a few instances
    output_dir = Path("outputs/explanations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(min(5, len(X_test))):
        exp = explainer.explain_instance(X_test[i], model.predict_proba, num_features=10)
        exp.save_to_file(output_dir / f"lime_explanation_{i}.html")
    
    logger.info("LIME explanations saved to {}".format(output_dir))
