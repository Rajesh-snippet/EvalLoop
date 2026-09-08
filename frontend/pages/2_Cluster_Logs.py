import streamlit as st
from api_client import run_pipeline_job_with_live_log
from ui import configure_page, page_header, section, sidebar
configure_page("Cluster Logs"); sidebar(); page_header("Sample & Cluster Logs", "Phase 2: discover interaction groups and prioritize high-value evaluation candidates.")
st.markdown('<div class="el-callout">This stage works on the existing Phase 1 log corpus. It does not generate new logs.</div>', unsafe_allow_html=True)
section("Run clustering")
if st.button("Run clustering", type="primary"):
    job = run_pipeline_job_with_live_log("/pipeline/cluster", "/pipeline/jobs/{job_id}")
    if job["status"] == "completed": st.success("Clustering completed successfully.")
    else: st.error(f"Clustering failed. Exit code: {job.get('returncode', 'unknown')}")
with st.expander("What this stage does"): st.markdown("EvalLoop combines sampling strategies with embedding-based clustering and candidate scoring to surface representative, unusual, risky, or failure-heavy interactions.")
