import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from api_client import api_get, run_pipeline_job_with_live_log

st.title("1. Generate Synthetic Logs")
st.caption("Phase 1 — creates a fresh batch of synthetic production logs via Groq.")

try:
    summary = api_get("/logs/summary")
    st.write("Current log set:", summary)
except Exception:
    st.caption("No existing log set found yet, or API unreachable.")

st.warning("This calls the Groq API and may take a while depending on batch size and quota.")

if st.button("Generate logs", type="primary"):
    job = run_pipeline_job_with_live_log(
        start_path="/pipeline/generate-logs",
        status_path_template="/pipeline/jobs/{job_id}",
    )
    if job["status"] == "completed":
        st.success("Log generation completed.")
    else:
        st.error(f"Log generation failed (exit code {job.get('returncode')}). See log above.")
