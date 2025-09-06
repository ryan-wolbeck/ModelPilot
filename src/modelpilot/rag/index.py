from pathlib import Path
import json, re

def collect_docs(artifacts_dir="artifacts"):
    docs = []
    for mani in Path(artifacts_dir).glob("*/manifest.json"):
        d = json.loads(mani.read_text())
        docs.append({"run_id": d["run_id"], "text": json.dumps(d, indent=2)})
    return docs

def simple_search(query, docs):
    q = query.lower()
    scored = []
    for d in docs:
        txt = d["text"].lower()
        score = sum(1 for w in q.split() if w in txt)
        scored.append((score, d))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [d for s, d in scored if s > 0][:5]
