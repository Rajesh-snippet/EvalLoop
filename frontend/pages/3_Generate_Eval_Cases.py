import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from api_client import api_get, run_pipeline_job_with_live_log

st.title("3. Generate Eval Case Labels")
st.caption("Phase 3 — auto-generates eval case labels from clustered candidates via an LLM judge.")

try:
    summary = api_get("/eval-cases/summary")
    st.write("Current eval case set:", summary)
except Exception:
    st.caption("No existing eval case set found yet, or API unreachable.")

st.warning("This calls the Groq API and may take a while depending on batch size and quota.")

col1, col2, col3 = st.columns(3)
with col1:
    limit = st.number_input("Limit (optional)", min_value=0, value=0, help="0 = no limit")
with col2:
    confidence_threshold = st.number_input(
        "Confidence threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.05
    )
with col3:
    dedup_threshold = st.number_input(
        "Dedup threshold", min_value=0.0, max_value=1.0, value=0.92, step=0.01
    )

if st.button("Generate labels", type="primary"):
    payload = {
        "limit": int(limit) if limit else None,
        "confidence_threshold": confidence_threshold,
        "dedup_threshold": dedup_threshold,
    }
    job = run_pipeline_job_with_live_log(
        start_path="/pipeline/generate-labels",
        status_path_template="/pipeline/jobs/{job_id}",
        payload=payload,
    )
    if job["status"] == "completed":
        st.success("Label generation completed.")
    else:
        st.error(f"Label generation failed (exit code {job.get('returncode')}). See log above.")
