"""
SHAP explainability implementation
"""
import shap
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_shap_explanations(model, X_test, feature_names, n_samples=100):
    """Generate SHAP explanations for model interpretability"""
    logger.info("Creating SHAP explainer with {} background samples...".format(n_samples))
    
    # Create explainer
    explainer = shap.TreeExplainer(model, shap.sample(X_test, n_samples))
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # Save summary plot
    output_dir = Path("outputs/explanations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    import matplotlib.pyplot as plt
    plt.savefig(output_dir / "shap_summary.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    logger.info("SHAP explanations saved to {}".format(output_dir))
