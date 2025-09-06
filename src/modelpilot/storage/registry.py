from pathlib import Path
import json

class Registry:
    def __init__(self, artifacts_dir="artifacts"):
        self.root = Path(artifacts_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_manifest(self, run_id, manifest):
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return run_dir
