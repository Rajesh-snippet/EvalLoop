import streamlit as st
from api_client import api_get, run_pipeline_job_with_live_log
from ui import configure_page, metric_card, page_header, section, sidebar
configure_page("Generate Logs"); sidebar(); page_header("Generate Synthetic Logs", "Phase 1: create production-like LLM interactions for the evaluation pipeline.")
try: summary = api_get("/logs/summary")
except Exception: summary = None
section("Current log set")
if summary:
    cols = st.columns(4)
    vals = [("Total logs", summary.get("total_logs", summary.get("count", "—")), "Current DuckDB corpus"), ("Errors", summary.get("error_count", summary.get("errors", "—")), "Recorded error signals"), ("Retries", summary.get("retry_count", summary.get("retries", "—")), "Interactions with retries"), ("Safety cases", summary.get("safety_count", summary.get("safety", "—")), "Safety edge cases")]
    for col, (a,b,c) in zip(cols, vals):
        with col: metric_card(a,b,c)
else: st.info("No current log summary is available. Make sure the API is running and Phase 1 data exists.")
section("Generate")
st.warning("Log generation uses the configured LLM provider and may consume API quota.")
if st.button("Generate logs", type="primary"):
    job = run_pipeline_job_with_live_log("/pipeline/generate-logs", "/pipeline/jobs/{job_id}")
    if job["status"] == "completed": st.success("Log generation completed successfully.")
    else: st.error(f"Log generation failed. Exit code: {job.get('returncode', 'unknown')}")
with st.expander("What this stage does"): st.markdown("The generator creates structured interactions containing prompts, responses, feature metadata, feedback/failure signals, retry information, and redaction metadata.")
