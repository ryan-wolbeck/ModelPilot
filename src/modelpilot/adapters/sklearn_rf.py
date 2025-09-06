from dataclasses import dataclass
from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor
import numpy as np

@dataclass
class SKLearnRFAdapter:
    params: Dict[str, Any]

    def build(self, params: Dict[str, Any]):
        p = {"n_estimators": params.get("n_estimators", 200),
             "max_depth": params.get("base_max_depth", None),
             "min_samples_leaf": params.get("base_min_samples_leaf", 1),
             "random_state": 42}
        class ModelWrapper:
            def __init__(self):
                self.model = RandomForestRegressor(**p)
            def fit(self, X, y):
                self.model.fit(X, y); return self
            def predict(self, X):
                return self.model.predict(X)
            def predict_dist(self, X):
                # Simple Gaussian approx with leaf variance (stub)
                preds = self.model.predict(X)
                return _Gaussian(preds, np.full_like(preds, fill_value=np.std(preds) + 1e-6))
        return ModelWrapper()

    def param_space(self, search_cfg):
        # Map to generic names used in examples
        return {
            "n_estimators": ("int", 100, 800),
            "base_max_depth": ("int", 2, 12),
            "base_min_samples_leaf": ("int", 1, 16),
            "learning_rate": ("fixed", 0.05)  # ignored here, for API parity
        }

class _Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu; self.sigma = sigma
    def logpdf(self, y):
        var = self.sigma ** 2
        return -0.5*(np.log(2*np.pi*var) + ((y - self.mu)**2)/var)
