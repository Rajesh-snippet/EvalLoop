"""
EvalLoop — Flywheel Demo.

The plan's explicit Phase 6 ask: "show the flywheel — logs entering,
candidates being selected, labels being generated, humans approving, and
the eval dataset growing." This page traces ONE log through every stage
it actually passed through, using real data already in the databases —
no synthetic narrative, no mocked steps.

Merged into the main EvalLoop frontend as a native page.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root, for src.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # frontend/, for ui.py

import duckdb
import streamlit as st  # noqa: E402

from src.eval_builder.eval_case_db import DEFAULT_EVAL_DB_PATH, load_all_eval_cases  # noqa: E402
from src.eval_runner.metrics import DEFAULT_RUNS_DB_PATH  # noqa: E402
from src.review.review_edit_db import load_edits_for_case  # noqa: E402
from src.utils.db import DEFAULT_DB_PATH, load_all_logs  # noqa: E402
from ui import badge, callout, configure_page, page_header, section, sidebar, status_badge  # noqa: E402

configure_page("Flywheel Demo")
sidebar(active=None)
page_header(
    "Flywheel Demo",
    "Trace one log's real journey through every stage of the pipeline — sampled, "
    "labeled, reviewed, approved, evaluated.",
)


@st.cache_data(show_spinner="Loading logs...")
def _load_logs():
    return load_all_logs(DEFAULT_DB_PATH)


@st.cache_data(show_spinner="Loading eval cases...")
def _load_cases():
    return load_all_eval_cases(DEFAULT_EVAL_DB_PATH)


def _latest_run_result(case_id: str):
    """Direct query against eval_runs.duckdb — schema owned by this
    project's own runner.py, already verified."""
    if not Path(DEFAULT_RUNS_DB_PATH).exists():
        return None
    con = duckdb.connect(DEFAULT_RUNS_DB_PATH, read_only=True)
    try:
        row = con.execute(
            """
            SELECT passed, judge_reasoning, timestamp
            FROM eval_runs
            WHERE case_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            [case_id],
        ).fetchone()
    except duckdb.CatalogException:
        row = None
    finally:
        con.close()
    return row


logs = _load_logs()
cases = _load_cases()

if not logs:
    callout("No logs found yet — generate logs in Phase 1 first.", "warning")
    st.stop()

cases_by_log_id = {}
for case in cases:
    cases_by_log_id.setdefault(case.source_log_id, []).append(case)

# Prioritize logs that actually became eval cases in the dropdown ordering,
# so a first-time visitor doesn't have to hunt for an interesting example.
logs_with_cases = [log for log in logs if log.log_id in cases_by_log_id]
logs_without_cases = [log for log in logs if log.log_id not in cases_by_log_id]
ordered_logs = logs_with_cases + logs_without_cases

section("Pick a log to trace")
options = {
    f"{log.log_id}  —  {log.feature_name}"
    + ("  (became an eval case)" if log.log_id in cases_by_log_id else "  (not selected as a candidate)"): log
    for log in ordered_logs
}
choice = st.selectbox("Log", list(options.keys()))
log = options[choice]

st.markdown("---")

# --- Stage 1: Log ingested ------------------------------------------------
section("Stage 1 — Log ingested")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"**Feature**\n\n{log.feature_name}")
with col2:
    st.markdown(f"**Feedback**\n\n{log.user_feedback or 'none'}")
with col3:
    st.markdown(f"**Error status**\n\n{log.error_status or 'none'}")
with col4:
    st.markdown(f"**Retries**\n\n{log.retry_count}")

with st.expander("Prompt / response"):
    st.markdown("**Prompt**")
    st.code(log.prompt, language=None)
    st.markdown("**Original response**")
    st.code(log.response, language=None)

matching_cases = cases_by_log_id.get(log.log_id, [])

# --- Stage 2: Sampled + clustered + labeled ---------------------------------
st.markdown("---")
section("Stage 2 — Sampled, clustered, and labeled")

if not matching_cases:
    callout(
        "This log was **not** selected as an eval-case candidate — it either wasn't "
        "sampled by any strategy (random / failure-biased / diversity), or was "
        "sampled but rejected as a near-duplicate of an existing case. This is "
        "expected and healthy: not every log should become an eval case.",
        "neutral",
    )
    st.stop()

for case in matching_cases:
    st.markdown(
        f"**Cluster:** `{case.source_cluster_id}` &nbsp;&nbsp; "
        f"**Eval type:** `{case.eval_type}` &nbsp;&nbsp; "
        f"**Difficulty:** `{case.difficulty}` &nbsp;&nbsp; "
        f"**Confidence:** `{case.confidence_score:.2f}`",
    )
    st.markdown(
        f"**Label source:** {badge(case.label_source, 'info' if case.label_source == 'auto' else 'warning')}",
        unsafe_allow_html=True,
    )

    with st.expander("Generated label", expanded=True):
        if case.expected_behavior:
            st.markdown("**Expected behavior**")
            st.write(case.expected_behavior)
        if case.rubric:
            st.markdown("**Rubric**")
            for line in case.rubric:
                st.write(f"- {line}")
        if case.forbidden_assertions:
            st.markdown("**Forbidden assertions**")
            for line in case.forbidden_assertions:
                st.write(f"- {line}")

    # --- Stage 3: Human review ---------------------------------------------
    st.markdown("---")
    section("Stage 3 — Human review")

    edits = load_edits_for_case(case.case_id, DEFAULT_EVAL_DB_PATH)
    if not edits:
        callout(
            f"No review edits recorded yet for this case. Current status: "
            f"{status_badge(case.status)}",
            "neutral",
        )
    else:
        st.markdown(f"**{len(edits)} correction(s) made during review:**")
        for edit in edits:
            with st.container(border=True):
                st.markdown(f"**Field:** `{edit.field_changed}`")
                bc, ac = st.columns(2)
                with bc:
                    st.caption("Before")
                    st.write((edit.before_value or "")[:300])
                with ac:
                    st.caption("After")
                    st.write((edit.after_value or "")[:300])
                if edit.reason:
                    st.caption(f"Reason: {edit.reason}")

    st.markdown(f"**Current status:** {status_badge(case.status)}", unsafe_allow_html=True)

    # --- Stage 4: Approved + exported ----------------------------------------
    st.markdown("---")
    section("Stage 4 — Approved dataset")
    if case.status == "approved":
        callout(
            f"Approved on {case.approved_at}. This case is part of the approved "
            "dataset and included in the next JSONL export.",
            "success",
        )
    else:
        callout(f"Not yet approved (status: {case.status}) — not included in dataset exports.", "neutral")

    # --- Stage 5: Evaluated ---------------------------------------------------
    st.markdown("---")
    section("Stage 5 — Evaluated against the target model")
    run_result = _latest_run_result(case.case_id)
    if run_result is None:
        callout("Not yet included in an eval run.", "neutral")
    else:
        passed, reasoning, timestamp = run_result
        st.markdown(
            f"**Latest run ({timestamp}):** {status_badge('pass' if passed else 'fail')}",
            unsafe_allow_html=True,
        )
        st.caption(reasoning)
