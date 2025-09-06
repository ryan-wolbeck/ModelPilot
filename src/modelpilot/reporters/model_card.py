from pathlib import Path
import json, os

TEMPLATE = '''
<!doctype html>
<html><head><meta charset="utf-8"><title>Model Card - {{run_id}}</title></head>
<body style="font-family: system-ui, sans-serif; margin: 2rem;">
<h1>Model Card</h1>
<p><strong>Run ID:</strong> {{run_id}}</p>
<h2>Best Parameters</h2>
<pre>{{best_params}}</pre>
<h2>Metrics</h2>
<pre>{{metrics}}</pre>
<p>Artifacts directory: artifacts/{{run_id}}/</p>
</body></html>
'''

def render_model_card(run_id: str = "LAST", out: str = "reports/model_card.html"):
    out_path = Path(out); out_path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal: load last manifest if LAST
    mani_dir = Path("artifacts")
    if run_id == "LAST":
        runs = sorted([p for p in mani_dir.glob("*/manifest.json")], key=lambda p: p.stat().st_mtime)
        if not runs:
            raise FileNotFoundError("No runs found.")
        manifest = json.loads(runs[-1].read_text())
    else:
        manifest_path = mani_dir / run_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
    html = TEMPLATE.replace("{{run_id}}", manifest["run_id"])\
                   .replace("{{best_params}}", json.dumps(manifest["best_params"], indent=2))\
                   .replace("{{metrics}}", json.dumps(manifest["metrics"], indent=2))
    out_path.write_text(html)
    return out
