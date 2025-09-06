import typer, json, uuid, time
from pathlib import Path
from typing import Optional
import yaml

from modelpilot.core.study import Study
from modelpilot.reporters.model_card import render_model_card
from modelpilot.governance.gates import Gate

app = typer.Typer(add_completion=False)

@app.command()
def run(config: str = typer.Option(..., help="Path to YAML config")):
    cfg = yaml.safe_load(Path(config).read_text())
    study = Study.from_config(cfg)
    result = study.run()
    run_id = result.get("run_id", str(uuid.uuid4()))
    print(json.dumps({"run_id": run_id, "best_params": result.get("best_params")}, indent=2))

@app.command()
def report(run_id: str = typer.Option("LAST", help="Run ID or LAST"),
           out: str = typer.Option("reports/model_card.html")):
    render_model_card(run_id=run_id, out=out)
    print(f"Report written to {out}")

@app.command()
def gate(run_id: str = typer.Option("LAST"),
         policy: Optional[str] = typer.Option(None, help="YAML policy; defaults to config's governance")):
    passed = Gate.evaluate(run_id=run_id, policy_path=policy)
    print(json.dumps({"run_id": run_id, "passed": passed}, indent=2))

if __name__ == "__main__":
    app()
