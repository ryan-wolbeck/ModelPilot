from pathlib import Path
import json, yaml

class Gate:
    @staticmethod
    def evaluate(run_id: str = "LAST", policy_path: str = None) -> bool:
        mani_dir = Path("artifacts")
        if run_id == "LAST":
            runs = sorted([p for p in mani_dir.glob("*/manifest.json")], key=lambda p: p.stat().st_mtime)
            manifest = json.loads(runs[-1].read_text())
        else:
            manifest = json.loads((mani_dir / run_id / "manifest.json").read_text())

        metrics = manifest.get("metrics", {})
        rules = []
        if policy_path:
            rules = yaml.safe_load(Path(policy_path).read_text()).get("rules", [])
        # minimal: pass if metrics exist; extend with real rules
        return bool(metrics)
