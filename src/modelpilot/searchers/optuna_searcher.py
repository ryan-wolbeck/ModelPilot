from typing import Callable, Dict, Any
import importlib
import math
import random

class OptunaSearcher:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def _suggest(self, trial, name, spec):
        kind = spec[0]
        if kind == "int":
            return trial.suggest_int(name, spec[1], spec[2])
        if kind == "loguniform":
            return trial.suggest_float(name, spec[1], spec[2], log=True)
        if kind == "fixed":
            return spec[1]
        if kind == "float":
            return trial.suggest_float(name, spec[1], spec[2])
        raise ValueError(f"Unsupported space type: {kind}")

    def search(self, objective: Callable[[Dict[str, Any]], Dict[str, float]], space: Dict[str, Any]):
        if importlib.util.find_spec("optuna") is None:
            # Fallback: random search if Optuna not installed
            n_trials = int(self.cfg.get("n_trials", 10))
            best = None
            for _ in range(n_trials):
                params = {}
                for k, v in space.items():
                    if v[0] == "int":
                        params[k] = random.randint(v[1], v[2])
                    elif v[0] == "loguniform":
                        import math, random
                        lo, hi = math.log(v[1]), math.log(v[2])
                        params[k] = math.exp(random.uniform(lo, hi))
                    elif v[0] == "fixed":
                        params[k] = v[1]
                    elif v[0] == "float":
                        params[k] = random.uniform(v[1], v[2])
                metrics = objective(params)
                score = metrics.get("rmse", float("inf"))
                if best is None or score < best["metrics"]["rmse"]:
                    best = {"params": params, "metrics": metrics}
            return best

        import optuna
        direction = "minimize"  # using RMSE as primary
        study = optuna.create_study(direction=direction)

        def _obj(trial):
            params = {k: self._suggest(trial, k, spec) for k, spec in space.items()}
            metrics = objective(params)
            trial.set_user_attr("metrics", metrics)
            return metrics["rmse"]

        study.optimize(_obj, n_trials=int(self.cfg.get("n_trials", 30)))
        best_trial = study.best_trial
        return {"params": {k: best_trial.params.get(k) for k in space.keys()},
                "metrics": best_trial.user_attrs.get("metrics", {})}
