"""
SHAP explainability implementation
"""

import builtins
import contextlib
import logging
from pathlib import Path

import shap

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _tolerate_bracketed_base_score():
    """Work around a shap==0.43.0 / XGBoost 2.x incompatibility.

    XGBoost 2.x serializes base_score as a bracketed vector string (e.g.
    "[9.090909E-2]") to support multi-output models. Somewhere inside
    shap 0.43's model loader, that value is parsed with a plain
    float(...), which raises "could not convert string to float" on the
    brackets.

    Two earlier attempts patched this upstream -- first via
    booster.save_config()/load_config() (only edits training
    hyperparameters, not the state SHAP reads), then via a
    save_model()/load_model() round-trip on the actual model file (still
    didn't take: SHAP kept reading the original bracketed string
    afterwards, meaning it isn't re-reading base_score from either of
    those representations). Rather than keep guessing which internal
    path shap's closed-source-to-us version reads it from, this
    intercepts the exact, confirmed point of failure instead: it
    temporarily wraps the builtin float() so that IF a bracketed string
    like "[9.090909E-2]" is ever passed to it, it strips the brackets
    first. Every other float() call anywhere in this process, including
    ones raising legitimate errors on genuinely bad input, behaves
    exactly as before -- the wrapper only special-cases the one bracketed
    pattern, and only for the duration of the `with` block. The original
    float() is always restored in the `finally`, even if SHAP raises.
    """
    original_float = builtins.float

    def _patched_float(x, *args, **kwargs):
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("[") and s.endswith("]"):
                return original_float(s[1:-1])
        return original_float(x, *args, **kwargs)

    builtins.float = _patched_float
    try:
        yield
    finally:
        builtins.float = original_float


def generate_shap_explanations(model, X_test, feature_names, n_samples=100):
    """Generate SHAP explanations for model interpretability"""
    logger.info(
        "Creating SHAP explainer with {} background samples...".format(n_samples)
    )

    with _tolerate_bracketed_base_score():
        explainer = shap.TreeExplainer(model, shap.sample(X_test, n_samples))
        shap_values = explainer.shap_values(X_test)

    # Save summary plot
    output_dir = Path("outputs/explanations")
    output_dir.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    import matplotlib.pyplot as plt

    plt.savefig(output_dir / "shap_summary.png", bbox_inches="tight", dpi=150)
    plt.close()

    logger.info("SHAP explanations saved to {}".format(output_dir))
