import streamlit as st
from modelpilot.rag.index import collect_docs, simple_search

st.set_page_config(page_title="modelpilot reviewer", layout="wide")
st.title("modelpilot Reviewer (MVP)")

query = st.text_input("Ask about your runs (e.g., 'best params' or 'rmse')")
docs = collect_docs()
if query:
    hits = simple_search(query, docs)
    if not hits:
        st.info("No hits.")
    for d in hits:
        with st.expander(f"Run {d['run_id']}"):
            st.code(d["text"], language="json")
else:
    st.write("Index contains", len(docs), "runs.")
