# modelpilot

**modelpilot** is a small, opinionated framework for **generic, rigorous model tuning** with:
- pluggable **Adapters** (NGBoost, XGBoost, sklearn, etc.)
- swappable **Searchers** (Optuna now; Ray Tune/skopt later)
- first-class **Evaluation** (proper scoring rules, calibration)
- automatic **Reporting** (Model Card w/ plots)
- policy-as-code **Governance** gates
- optional **Agentic AI** to analyze runs and generate reports
- a **Streamlit reviewer chatbot** (RAG over run artifacts) for interactive review

> This is an MVP scaffold. Pieces are stubs you can extend.

## Quick start

```bash
# create and activate env (example)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# run an example (uses sklearn synthetic data by default)
modelpilot run --config examples/clearwater.yaml

# generate a model card for the last run (id printed at the end)
modelpilot report --run-id LAST

# open the reviewer app
streamlit run streamlit_app/app.py
```

## Project layout
```
src/modelpilot/
  adapters/       # model wrappers
  searchers/      # HPO engines
  evaluators/     # metrics + calibration
  analysts/       # SHAP, drift, diagnostics
  reporters/      # model card generator
  governance/     # gates and policy checks
  storage/        # artifact registry helpers
  cli/            # Typer-based CLI
  rag/            # indexing + retrieval for chatbot
streamlit_app/    # Streamlit reviewer UI
examples/         # configs + scripts
tests/            # unit tests
```

## Status
- ✅ Minimal CLI: `run`, `report`, `gate`
- ✅ NGBoost adapter (optional)
- ✅ Optuna searcher (stubbed if Optuna missing)
- ✅ SHAP analyst (falls back to permutation importances if SHAP missing)
- ✅ Streamlit reviewer (RAG over JSON artifacts)
- ⏳ Ray backend, fairness metrics, more adapters

## License
Apache-2.0
