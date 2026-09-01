"""
Streamlit review UI (Phase 4) — thin rendering layer over queue.py. All
actual logic (queue ordering, similarity lookup, diffing, state transitions)
lives in queue.py and is independently testable without this UI; this file
should stay close to pure rendering + button wiring.

Run with:
    streamlit run src/review/streamlit_app.py

Layout matches the build plan's Phase 4 spec:
    Left panel   -> original LogEntry (prompt, response, feedback, error status)
    Center panel -> proposed EvalCase fields, editable
    Right panel  -> similar existing (approved) cases, for reviewer context
    Buttons      -> Approve / Edit+Approve / Reject
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from src.eval_builder.eval_case_db import DEFAULT_EVAL_DB_PATH  # noqa: E402
from src.logs.schema import LogEntry  # noqa: E402
from src.review.queue import apply_review_action, find_similar_approved_cases, get_review_queue  # noqa: E402
from src.utils.db import DEFAULT_DB_PATH, load_all_logs  # noqa: E402

st.set_page_config(page_title="EvalLoop Review Queue", layout="wide")

REVIEWER_ID = "rajesh"  # single-reviewer setup for this project; would become
                         # an auth-derived value in a multi-reviewer deployment


@st.cache_data(show_spinner="Loading logs...")
def _load_logs_by_id() -> dict[str, LogEntry]:
    logs = load_all_logs(DEFAULT_DB_PATH)
    return {l.log_id: l for l in logs}


def _refresh_queue():
    # Streamlit's cache_data on get_review_queue itself would go stale after
    # an approve/reject action within the same session, so the queue is
    # deliberately NOT cached — it's cheap (a DuckDB query over a few dozen
    # rows), and correctness after every action matters more than shaving a
    # fraction of a second off a re-render.
    return get_review_queue(DEFAULT_EVAL_DB_PATH)


def main():
    st.title("EvalLoop — Human Review Queue")

    logs_by_id = _load_logs_by_id()
    queue = _refresh_queue()

    if not queue:
        st.success("Review queue is empty — no draft cases waiting.")
        return

    st.caption(f"{len(queue)} case(s) awaiting review, ordered by ascending confidence.")

    # session state tracks position in the queue so approving/rejecting one
    # case doesn't reset the reviewer back to the top every rerun
    if "queue_position" not in st.session_state:
        st.session_state.queue_position = 0
    pos = min(st.session_state.queue_position, len(queue) - 1)
    case = queue[pos]

    st.progress((pos + 1) / len(queue), text=f"Case {pos + 1} of {len(queue)}")

    left, center, right = st.columns([1, 1.3, 1])

    with left:
        st.subheader("Original interaction")
        log = logs_by_id.get(case.source_log_id)
        if log is None:
            st.warning(f"Source log {case.source_log_id} not found in raw_logs.duckdb")
        else:
            st.markdown(f"**Feature:** {log.feature_name}")
            st.markdown(f"**User feedback:** {log.user_feedback} &nbsp;|&nbsp; "
                        f"**Error status:** {log.error_status or 'none'} &nbsp;|&nbsp; "
                        f"**Retries:** {log.retry_count}")
            st.text_area("Prompt", log.prompt, height=120, disabled=True, key="log_prompt")
            st.text_area("Response", log.response, height=150, disabled=True, key="log_response")

        st.divider()
        st.markdown(f"**Eval type:** {case.eval_type} &nbsp;|&nbsp; "
                     f"**Difficulty:** {case.difficulty} &nbsp;|&nbsp; "
                     f"**Confidence:** {case.confidence_score:.2f}")
        st.markdown(f"**Cluster:** {case.source_cluster_id}")

    with center:
        st.subheader("Proposed eval case (editable)")

        edited_expected_behavior = None
        edited_rubric = None
        edited_forbidden = None

        if case.eval_type in ("golden_answer", "expected_refusal"):
            edited_expected_behavior = st.text_area(
                "Expected behavior", case.expected_behavior or "", height=200, key="edit_expected"
            )
        else:
            rubric_text = "\n".join(case.rubric or [])
            edited_rubric_text = st.text_area(
                "Rubric (one assertion per line)", rubric_text, height=150, key="edit_rubric"
            )
            edited_rubric = [line.strip() for line in edited_rubric_text.split("\n") if line.strip()]

            forbidden_text = "\n".join(case.forbidden_assertions or [])
            edited_forbidden_text = st.text_area(
                "Forbidden assertions (one per line)", forbidden_text, height=80, key="edit_forbidden"
            )
            edited_forbidden = [line.strip() for line in edited_forbidden_text.split("\n") if line.strip()]

        edited_difficulty = st.selectbox(
            "Difficulty", ["easy", "medium", "hard"],
            index=["easy", "medium", "hard"].index(case.difficulty), key="edit_difficulty"
        )
        edited_tags_text = st.text_input("Tags (comma-separated)", ", ".join(case.tags), key="edit_tags")
        edited_tags = [t.strip() for t in edited_tags_text.split(",") if t.strip()]

        st.divider()
        col_a, col_b, col_c = st.columns(3)

        def _build_edited_fields() -> dict:
            fields = {"difficulty": edited_difficulty, "tags": edited_tags}
            if edited_expected_behavior is not None:
                fields["expected_behavior"] = edited_expected_behavior
            if edited_rubric is not None:
                fields["rubric"] = edited_rubric
                fields["forbidden_assertions"] = edited_forbidden
            return fields

        if col_a.button("✅ Approve as-is", use_container_width=True):
            apply_review_action(case, "approve", reviewer_id=REVIEWER_ID, eval_db_path=DEFAULT_EVAL_DB_PATH)
            st.session_state.queue_position = pos  # stays; queue shrinks, next case slides into this index
            st.rerun()

        if col_b.button("✏️ Save edits + Approve", use_container_width=True):
            apply_review_action(
                case, "edit_approve", reviewer_id=REVIEWER_ID,
                edited_fields=_build_edited_fields(), eval_db_path=DEFAULT_EVAL_DB_PATH,
            )
            st.rerun()

        with col_c.popover("❌ Reject", use_container_width=True):
            reason = st.text_area("Rejection reason (required)", key="reject_reason")
            if st.button("Confirm reject", key="confirm_reject"):
                if not reason.strip():
                    st.error("A rejection reason is required.")
                else:
                    apply_review_action(
                        case, "reject", reviewer_id=REVIEWER_ID,
                        reject_reason=reason, eval_db_path=DEFAULT_EVAL_DB_PATH,
                    )
                    st.rerun()

        nav_a, nav_b = st.columns(2)
        if nav_a.button("⬅ Skip back", disabled=(pos == 0)):
            st.session_state.queue_position = max(0, pos - 1)
            st.rerun()
        if nav_b.button("Skip forward ➡", disabled=(pos >= len(queue) - 1)):
            st.session_state.queue_position = pos + 1
            st.rerun()

    with right:
        st.subheader("Similar approved cases")
        similar = find_similar_approved_cases(case.input.prompt, DEFAULT_EVAL_DB_PATH, k=3)
        if not similar:
            st.caption("No approved cases yet to compare against.")
        for sim_case, score in similar:
            with st.container(border=True):
                st.markdown(f"**Similarity: {score:.2f}** &nbsp;|&nbsp; {sim_case.eval_type}")
                st.caption(sim_case.input.prompt[:150])
                if sim_case.expected_behavior:
                    st.text(sim_case.expected_behavior[:200])
                elif sim_case.rubric:
                    st.text("\n".join(f"• {r}" for r in sim_case.rubric[:3]))


if __name__ == "__main__":
    main()
