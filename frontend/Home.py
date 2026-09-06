"""
EvalLoop Pipeline Control — Home.

A thin UI over the FastAPI backend for Phases 1, 2, 3, and 5. The
existing review app (src/review/streamlit_app.py) and dashboard
(dashboard/dataset_health.py) remain separate and unchanged — this app
covers the pipeline-generation steps that previously required the
command line.

Requires the API to be running:
    uvicorn src.api.main:app --reload

Run this app:
    streamlit run frontend/Home.py
"""

import streamlit as st

st.set_page_config(page_title="EvalLoop Pipeline Control", layout="wide")

st.title("EvalLoop — Pipeline Control")
st.caption("Drive the pipeline end-to-end through the API instead of the command line.")

st.markdown(
    """
Use the pages in the sidebar to run each stage of the pipeline:

1. **Generate Logs** — create a fresh batch of synthetic production logs (Phase 1)
2. **Cluster Logs** — sample and cluster the current log set (Phase 2)
3. **Generate Eval Cases** — auto-label candidates into draft/approved eval cases (Phase 3)
5. **Export & Run Eval** — export approved cases to JSONL and run them against the target model (Phase 5)

For human review of draft cases, use the existing review app:
```
streamlit run src/review/streamlit_app.py
```

For dataset health and pass-rate trends, use the existing dashboard:
```
streamlit run dashboard/dataset_health.py
```

**Before using this app**, make sure the API is running in another terminal:
```
uvicorn src.api.main:app --reload
```
"""
)

st.info("This app calls http://localhost:8000 — make sure the API is up before running a stage.")
