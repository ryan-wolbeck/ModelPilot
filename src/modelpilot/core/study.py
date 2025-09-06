import os, uuid, json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from modelpilot.adapters.registry import get_adapter
from modelpilot.searchers.registry import get_searcher
from modelpilot.evaluators.metrics import negative_log_likelihood_stub
from modelpilot.storage.registry import Registry

@dataclass
class Study:
    cfg: Dict[str, Any]
    registry: Registry

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]):
        reg = Registry(artifacts_dir=cfg.get("storage", {}).get("artifacts_dir", "artifacts"))
        return cls(cfg=cfg, registry=reg)

    def _load_data(self):
        dcfg = self.cfg.get("data", {})
        if dcfg.get("kind") == "synthetic_regression":
            X, y = make_regression(n_samples=dcfg.get("n_samples", 2000),
                                   n_features=dcfg.get("n_features", 20),
                                   noise=dcfg.get("noise", 0.2),
                                   random_state=self.cfg.get("seed", 42))
            return pd.DataFrame(X), pd.Series(y)
        raise NotImplementedError("Only synthetic_regression in MVP")

    def run(self):
        np.random.seed(self.cfg.get("seed", 42))
        run_id = str(uuid.uuid4())

        X, y = self._load_data()

        # Splits
        scfg = self.cfg.get("split", {"type": "kfold", "n_splits": 5})
        kf = KFold(n_splits=scfg.get("n_splits", 5), shuffle=True, random_state=self.cfg.get("seed", 42))

        # Adapter
        adapter = get_adapter(self.cfg.get("adapter", {"name": "sklearn_random_forest"}))

        # Searcher
        searcher = get_searcher(self.cfg.get("search", {"name": "optuna", "n_trials": 10}))

        # Objective
        def objective(params):
            rmses = []
            nlls = []
            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                model = adapter.build(params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                rmse = mean_squared_error(y_val, preds, squared=False)
                # NLL stub: if model has predict_dist, use it; else fallback
                try:
                    dist = model.predict_dist(X_val)  # NGBoost-like
                    nll = -np.mean(dist.logpdf(y_val.values))
                except Exception:
                    nll = negative_log_likelihood_stub(y_val.values, preds)
                rmses.append(rmse); nlls.append(nll)
            return {"rmse": float(np.mean(rmses)), "nll": float(np.mean(nlls))}

        best = searcher.search(objective, adapter.param_space(self.cfg.get("search", {})))

        # Save manifest
        manifest = {"run_id": run_id, "best_params": best["params"], "metrics": best["metrics"]}
        self.registry.save_manifest(run_id, manifest)
        return {"run_id": run_id, "best_params": best["params"], "metrics": best["metrics"]}
