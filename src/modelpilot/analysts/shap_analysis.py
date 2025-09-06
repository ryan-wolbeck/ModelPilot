import importlib, json
from pathlib import Path
import numpy as np

def compute_shap_values(model, X, out_dir="artifacts"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    have_shap = importlib.util.find_spec("shap") is not None
    if not have_shap:
        # Fallback: permutation importance summary
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model.model if hasattr(model, "model") else model, X, np.zeros(len(X)), n_repeats=3, random_state=42)
        summary = {"feature_names": list(getattr(X, "columns", range(X.shape[1]))),
                   "mean_abs": r["importances_mean"].tolist()}
    else:
        import shap
        try:
            explainer = shap.TreeExplainer(model.model if hasattr(model, "model") else model)
            vals = explainer.shap_values(X)
            mean_abs = np.abs(vals).mean(0).tolist()
        except Exception:
            explainer = shap.KernelExplainer(lambda data: model.predict(data), X.sample(min(256, len(X)), random_state=42))
            vals = explainer.shap_values(X.sample(min(512, len(X)), random_state=42), nsamples=100)
            mean_abs = np.abs(vals).mean(0).tolist()
        summary = {"feature_names": list(getattr(X, "columns", range(X.shape[1]))),
                   "mean_abs": mean_abs}
    Path(f"{out_dir}/shap_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
