import streamlit as st
from api_client import api_get, run_pipeline_job_with_live_log
from ui import configure_page, metric_card, page_header, section, sidebar
configure_page("Generate Eval Cases"); sidebar(); page_header("Generate Eval Cases", "Phase 3: convert prioritized candidates into structured, reviewable evaluation cases.")
try: summary = api_get("/eval-cases/summary")
except Exception: summary = None
section("Current evaluation set")
if summary:
    cols = st.columns(4)
    vals = [("Total cases", summary.get("total", summary.get("total_cases", "—")), "All stored cases"), ("Approved", summary.get("approved", "—"), "Trusted cases"), ("Draft", summary.get("draft", "—"), "Awaiting human review"), ("Rejected", summary.get("rejected", "—"), "Excluded from dataset")]
    for col, (a,b,c) in zip(cols, vals):
        with col: metric_card(a,b,c)
else: st.info("No evaluation case summary is available yet.")
section("Generation settings")
c1,c2,c3 = st.columns(3)
with c1: limit = st.number_input("Candidate limit", min_value=0, value=0, step=1, help="0 means all candidates.")
with c2: confidence_threshold = st.number_input("Auto-approval threshold", min_value=0.0, max_value=1.0, value=.75, step=.05, format="%.2f")
with c3: dedup_threshold = st.number_input("Deduplication threshold", min_value=0.0, max_value=1.0, value=.92, step=.01, format="%.2f")
st.warning("This stage calls the configured LLM provider for label generation and judging.")
if st.button("Generate evaluation cases", type="primary"):
    payload = {"limit": int(limit) if limit else None, "confidence_threshold": float(confidence_threshold), "dedup_threshold": float(dedup_threshold)}
    job = run_pipeline_job_with_live_log("/pipeline/generate-labels", "/pipeline/jobs/{job_id}", payload)
    if job["status"] == "completed": st.success("Evaluation case generation completed successfully.")
    else: st.error(f"Evaluation case generation failed. Exit code: {job.get('returncode', 'unknown')}")
with st.expander("What this stage produces"): st.markdown("Each candidate becomes an evaluation case with evaluation type, expected behavior or rubric, difficulty, tags, confidence, and lifecycle status.")
