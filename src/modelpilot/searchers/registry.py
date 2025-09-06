from typing import Dict, Any
from .optuna_searcher import OptunaSearcher

def get_searcher(cfg: Dict[str, Any]):
    name = cfg.get("name", "optuna").lower()
    if name == "optuna":
        return OptunaSearcher(cfg)
    raise NotImplementedError(f"Unknown searcher: {name}")
