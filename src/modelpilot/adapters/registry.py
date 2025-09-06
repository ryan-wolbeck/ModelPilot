from typing import Dict, Any
from .sklearn_rf import SKLearnRFAdapter
from .ngboost_adapter import NGBoostAdapter

def get_adapter(cfg: Dict[str, Any]):
    name = cfg.get("name", "sklearn_random_forest").lower()
    if name == "ngboost":
        return NGBoostAdapter(cfg.get("params", {}))
    return SKLearnRFAdapter(cfg.get("params", {}))
