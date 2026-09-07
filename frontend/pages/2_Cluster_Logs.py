import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from api_client import run_pipeline_job_with_live_log

st.title("2. Sample & Cluster Logs")
st.caption("Phase 2 — samples and clusters the current log set, ranking eval-case candidates.")

st.info("Uses the existing log set from Phase 1 — no new logs are generated here.")

if st.button("Run clustering", type="primary"):
    job = run_pipeline_job_with_live_log(
        start_path="/pipeline/cluster",
        status_path_template="/pipeline/jobs/{job_id}",
    )
    if job["status"] == "completed":
        st.success("Clustering completed.")
    else:
        st.error(f"Clustering failed (exit code {job.get('returncode')}). See log above.")
