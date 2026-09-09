import streamlit as st
from ui import configure_page, page_header, section, sidebar
configure_page("Pipeline Control")
sidebar()
page_header("EvalLoop Pipeline", "Run the EvalLoop data-to-evaluation workflow from a single workspace.")
st.markdown('<div class="el-callout">EvalLoop converts production-like LLM interactions into reusable evaluation cases, routes uncertain cases through human review, and evaluates model performance against the approved dataset.</div>', unsafe_allow_html=True)
section("Workflow")
cols = st.columns(5)
stages = [("01", "Generate Logs", "Create synthetic production-like interactions."), ("02", "Cluster Logs", "Discover interaction groups and rank candidates."), ("03", "Generate Eval Cases", "Turn candidates into structured evaluation cases."), ("04", "Human Review", "Validate and approve generated evaluation cases."), ("05", "Export & Run Eval", "Export approved cases and measure model performance.")]
for col, (num, title, desc) in zip(cols, stages):
    with col:
        st.markdown(f'<div class="el-card"><div class="el-card-label">{num}</div><div style="font-weight:650;margin-top:.35rem">{title}</div><div class="el-card-help">{desc}</div></div>', unsafe_allow_html=True)
section("Getting started")
st.markdown("1. Start the FastAPI service: `uvicorn src.api.main:app --reload`\n2. Start this frontend: `streamlit run frontend/Home.py`\n3. Use the sidebar to execute each pipeline stage.\n4. Use the existing Phase 4 review workspace to review draft cases.")
st.caption("Local/demo application. Authentication and multi-user controls are not included.")
