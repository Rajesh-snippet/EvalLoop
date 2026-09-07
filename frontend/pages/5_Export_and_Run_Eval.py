import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from api_client import api_get, api_post

st.title("5. Export & Run Eval")
st.caption("Phase 5 — export approved cases to JSONL, run them against the target model, and view results.")

# --- Export -----------------------------------------------------------
st.subheader("Export approved cases")
if st.button("Export to JSONL"):
    try:
        result = api_post("/export")
        st.success(f"Exported v{result['version']} -> {result['path']} ({result['total_cases']} cases)")
    except Exception as exc:
        st.error(f"Export failed: {exc}")

st.markdown("---")

# --- Run eval -----------------------------------------------------------
st.subheader("Run eval against target model")
st.warning("This calls Groq for every approved case (target model + judge model) — can take several minutes.")

limit = st.number_input("Limit (optional, 0 = all approved cases)", min_value=0, value=0)

if st.button("Start eval run", type="primary"):
    try:
        start_result = api_post("/eval-runs", {"limit": int(limit) if limit else None})
    except Exception as exc:
        st.error(f"Could not start run: {exc}")
        st.stop()

    run_id = start_result["run_id"]
    st.write(f"Run started: `{run_id}`")
    status_placeholder = st.empty()

    while True:
        job = api_get(f"/eval-runs/{run_id}/status")
        status_placeholder.write(f"**Status:** {job['status']}")
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(3)

    if job["status"] == "completed":
        result = job["result"]
        st.success(f"Run complete: {result['passed']}/{result['total']} passed")
        if result.get("errors"):
            st.warning(f"{result['errors']} case(s) hit a runner error — see eval_runs.duckdb judge_reasoning.")
    else:
        st.error(f"Run failed: {job.get('error')}")

st.markdown("---")

# --- Run history + regression -------------------------------------------
st.subheader("Run history")
try:
    runs = api_get("/eval-runs")
except Exception as exc:
    runs = []
    st.caption(f"Could not load run history: {exc}")

if runs:
    st.dataframe(runs, use_container_width=True)

    if len(runs) >= 2:
        st.subheader("Regression diff (two most recent runs)")
        current_run_id = runs[0]["run_id"]
        previous_run_id = runs[1]["run_id"]
        diff = api_get(
            "/eval-runs/regression",
            params={"previous_run_id": previous_run_id, "current_run_id": current_run_id},
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Regressions ({len(diff['regressions'])})**")
            st.write(diff["regressions"] or "None")
        with col2:
            st.markdown(f"**Improvements ({len(diff['improvements'])})**")
            st.write(diff["improvements"] or "None")
    else:
        st.caption("Run eval at least twice to see a regression diff here.")
else:
    st.caption("No runs yet.")
