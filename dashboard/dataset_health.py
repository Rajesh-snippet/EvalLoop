"""
EvalLoop Phase 5 — Dataset Health Dashboard.

Read-only view over eval_cases.duckdb + eval_runs.duckdb: composition of
the approved eval set, review/label provenance, and pass-rate trend
across historical eval runs.

Run:
    streamlit run dashboard/dataset_health.py
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st  # noqa: E402

from src.eval_builder.eval_case_db import (  # noqa: E402
    DEFAULT_EVAL_DB_PATH,
    load_all_eval_cases,
)
from src.eval_runner.metrics import (  # noqa: E402
    DEFAULT_RUNS_DB_PATH,
    list_runs,
    regression_diff,
)

st.set_page_config(
    page_title="EvalLoop Dataset Health",
    layout="wide",
)

st.title("EvalLoop — Dataset Health")
st.caption("Composition, provenance, and regression tracking for the approved eval set.")


@st.cache_data(show_spinner="Loading eval cases...")
def _load_cases():
    return load_all_eval_cases(DEFAULT_EVAL_DB_PATH)


@st.cache_data(show_spinner="Loading eval runs...")
def _load_runs():
    return list_runs(DEFAULT_RUNS_DB_PATH)


cases = _load_cases()
runs = _load_runs()

if not cases:
    st.info("No eval cases found yet. Run Phases 1-4 first.")
    st.stop()

approved = [c for c in cases if c.status == "approved"]
draft = [c for c in cases if c.status == "draft"]
rejected = [c for c in cases if c.status == "rejected"]

# --- Top-level counts -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total cases", len(cases))
col2.metric("Approved", len(approved))
col3.metric("Draft (pending review)", len(draft))
col4.metric("Rejected", len(rejected))

st.markdown("---")

# --- Composition: category/cluster, difficulty -------------------------
left, right = st.columns(2)

with left:
    st.subheader("Approved cases by cluster")
    if approved:
        cluster_counts = Counter(c.source_cluster_id for c in approved)
        df = pd.DataFrame(
            sorted(cluster_counts.items(), key=lambda kv: -kv[1]),
            columns=["cluster", "count"],
        ).set_index("cluster")
        st.bar_chart(df)
    else:
        st.caption("No approved cases yet.")

with right:
    st.subheader("Approved cases by difficulty")
    if approved:
        difficulty_counts = Counter(c.difficulty for c in approved)
        df = pd.DataFrame(
            [{"difficulty": d, "count": difficulty_counts.get(d, 0)} for d in ["easy", "medium", "hard"]]
        ).set_index("difficulty")
        st.bar_chart(df)
    else:
        st.caption("No approved cases yet.")

# --- Provenance: auto-labeled vs human-edited --------------------------
st.subheader("Label provenance")
if approved:
    provenance_counts = Counter(getattr(c, "label_source", "auto") for c in approved)
    auto = provenance_counts.get("auto", 0)
    edited = provenance_counts.get("human_edited", 0)
    total_approved = len(approved)
    p1, p2 = st.columns(2)
    p1.metric("Auto-labeled, approved as-is", f"{auto} ({auto / total_approved:.0%})")
    p2.metric("Human-edited before approval", f"{edited} ({edited / total_approved:.0%})")
else:
    st.caption("No approved cases yet.")

# --- Freshness: days since last case added per cluster ------------------
st.subheader("Freshness by cluster")


def _to_naive_utc(dt):
    """Normalize a datetime to naive-UTC so aware/naive values can be subtracted safely."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


if approved:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    freshness_rows = []
    by_cluster: dict[str, list] = {}
    for c in approved:
        by_cluster.setdefault(c.source_cluster_id, []).append(c)

    for cluster_id, cluster_cases in by_cluster.items():
        latest = max(
            (
                normalized
                for c in cluster_cases
                if (normalized := _to_naive_utc(getattr(c, "approved_at", None))) is not None
            ),
            default=None,
        )
        days_since = (now - latest).days if latest else None
        freshness_rows.append(
            {
                "cluster": cluster_id,
                "cases": len(cluster_cases),
                "days_since_last_added": days_since if days_since is not None else "unknown",
            }
        )
    st.dataframe(pd.DataFrame(freshness_rows).sort_values("cluster"), use_container_width=True)
else:
    st.caption("No approved cases yet.")

st.markdown("---")

# --- Pass rate trend + regression diff ----------------------------------
st.subheader("Eval run history")

if not runs:
    st.info("No eval runs yet. Run `python -m src.eval_runner.runner` after approving cases.")
else:
    trend_df = pd.DataFrame(
        [{"run": r["timestamp"], "pass_rate": r["pass_rate"]} for r in reversed(runs)]
    ).set_index("run")
    st.line_chart(trend_df)

    latest_run = runs[0]
    st.metric(
        "Current pass rate",
        f"{latest_run['pass_rate']:.0%}",
        help=f"Run {latest_run['run_id']} — {latest_run['passed']}/{latest_run['total']} cases",
    )

    if len(runs) >= 2:
        st.subheader("Regressions vs previous run")
        diff = regression_diff(runs[1]["run_id"], runs[0]["run_id"], DEFAULT_RUNS_DB_PATH)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"**Regressions ({len(diff['regressions'])})** — passed before, failing now")
            if diff["regressions"]:
                st.dataframe(pd.DataFrame({"case_id": diff["regressions"]}), use_container_width=True)
            else:
                st.caption("None. 🎉")
        with r2:
            st.markdown(f"**Improvements ({len(diff['improvements'])})** — failed before, passing now")
            if diff["improvements"]:
                st.dataframe(pd.DataFrame({"case_id": diff["improvements"]}), use_container_width=True)
            else:
                st.caption("None this run.")
    else:
        st.caption("Need at least 2 runs to compute a regression diff.")
