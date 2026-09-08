import time
import streamlit as st
from api_client import api_get, api_post
from ui import configure_page, metric_card, page_header, section, sidebar
configure_page("Export and Run Evaluation"); sidebar(); page_header("Export & Run Evaluation", "Phase 5: export approved cases, execute evaluations, and compare model runs.")
section("1. Export approved dataset")
st.markdown("Create a versioned JSONL dataset containing evaluation cases that have reached the approved state.")
if st.button("Export approved cases", type="primary"):
    try:
        r = api_post("/export")
        st.success(f"Exported dataset v{r['version']} with {r['total_cases']} approved cases.")
        st.code(r['path'], language="text")
    except Exception as exc: st.error(f"Export failed: {exc}")
section("2. Run evaluation")
st.warning("An evaluation run can make LLM calls for approved cases and may take several minutes.")
limit = st.number_input("Evaluation case limit", min_value=0, value=0, step=1, help="0 means all approved cases.")
if st.button("Start evaluation run", type="primary"):
    try: start = api_post("/eval-runs", {"limit": int(limit) if limit else None})
    except Exception as exc: st.error(f"Could not start evaluation run: {exc}"); st.stop()
    run_id = start["run_id"]; box = st.empty()
    while True:
        job = api_get(f"/eval-runs/{run_id}/status"); box.info(f"Run status: {job.get('status','unknown')}")
        if job.get("status") in {"completed", "failed"}: break
        time.sleep(3)
    if job["status"] == "completed":
        r = job.get("result", {}); cols = st.columns(3)
        for col,label,key,help_text in zip(cols,["Passed","Total","Runner errors"],["passed","total","errors"],["Cases passing evaluation","Cases evaluated","Cases with execution errors"]):
            with col: metric_card(label, r.get(key,0), help_text)
        st.success("Evaluation run completed successfully.")
    else: st.error(f"Evaluation run failed: {job.get('error','Unknown error')}")
section("3. Run history")
try: runs = api_get("/eval-runs")
except Exception as exc: runs=[]; st.error(f"Could not load evaluation history: {exc}")
if runs:
    st.dataframe(runs, use_container_width=True, hide_index=True)
    if len(runs) >= 2:
        section("Regression comparison")
        cur,prev = runs[0]["run_id"],runs[1]["run_id"]
        diff = api_get("/eval-runs/regression", params={"previous_run_id":prev,"current_run_id":cur})
        c1,c2 = st.columns(2)
        with c1: metric_card("Regressions", len(diff.get("regressions",[])), "Cases that passed before and fail now")
        with c2: metric_card("Improvements", len(diff.get("improvements",[])), "Cases that failed before and pass now")
        with st.expander("Regression details"): st.write(diff.get("regressions") or "No regressions detected.")
        with st.expander("Improvement details"): st.write(diff.get("improvements") or "No improvements detected.")
    else: st.info("Run the evaluation at least twice to compare two runs.")
else: st.info("No evaluation runs are available yet.")
with st.expander("Phase 5 scope"): st.markdown("Phase 5 turns approved evaluation cases into a versioned JSONL asset, executes model evaluations, stores run history, and compares runs for regressions.")
