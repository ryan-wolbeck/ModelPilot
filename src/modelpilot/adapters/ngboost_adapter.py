from dataclasses import dataclass
from typing import Dict, Any
import importlib

@dataclass
class NGBoostAdapter:
    params: Dict[str, Any]

    def _ngb(self):
        if importlib.util.find_spec("ngboost") is None:
            raise ImportError("ngboost not installed; install with: pip install modelpilot[extras]")
        from ngboost import NGBRegressor
        from ngboost.scores import LogScore
        from ngboost.distns import Normal
        base_max_depth = self.params.get("base_max_depth", 3)
        base_min_samples_leaf = self.params.get("base_min_samples_leaf", 5)
        from sklearn.tree import DecisionTreeRegressor
        base = DecisionTreeRegressor(max_depth=base_max_depth, min_samples_leaf=base_min_samples_leaf, random_state=42)
        return NGBRegressor(Base=base, Dist=Normal, Score=LogScore, random_state=42)

    def build(self, params: Dict[str, Any]):
        model = self._ngb()
        # Map common params
        if hasattr(model, "n_estimators") and "n_estimators" in params:
            model.n_estimators = params["n_estimators"]
        if hasattr(model, "learning_rate") and "learning_rate" in params:
            model.learning_rate = params["learning_rate"]
        # Update base learner depth/leaf if provided
        return model

    def param_space(self, search_cfg):
        return {
            "n_estimators": ("int", 200, 1500),
            "learning_rate": ("loguniform", 1e-3, 1e-1),
            "base_max_depth": ("int", 2, 8),
            "base_min_samples_leaf": ("int", 1, 32)
        }
