"""
EvalLoop Phase 4 — Human Review Workspace.

Thin UI layer over review/queue.py. Review state transitions, queue ordering,
similarity lookup, and persistence remain in queue.py.

Run:
    streamlit run src/review/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from src.eval_builder.eval_case_db import DEFAULT_EVAL_DB_PATH  # noqa: E402
from src.logs.schema import LogEntry  # noqa: E402
from src.review.queue import (  # noqa: E402
    apply_review_action,
    find_similar_approved_cases,
    get_review_queue,
)
from src.utils.db import DEFAULT_DB_PATH, load_all_logs  # noqa: E402


st.set_page_config(
    page_title="EvalLoop Review Workspace",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

REVIEWER_ID = "rajesh"

DIFFICULTY_OPTIONS = ["easy", "medium", "hard"]

# Small amount of CSS is intentional: the application should still look like
# a native Streamlit tool, while giving the reviewer a clearer hierarchy.
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .review-header {
            border-bottom: 1px solid rgba(128,128,128,.25);
            padding-bottom: 1rem;
            margin-bottom: 1.25rem;
        }

        .section-label {
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.65;
            margin-bottom: 0.35rem;
        }

        .case-title {
            font-size: 1.45rem;
            font-weight: 700;
            margin: 0;
        }

        .case-subtitle {
            opacity: 0.7;
            margin-top: 0.2rem;
        }

        .meta-card {
            border: 1px solid rgba(128,128,128,.25);
            border-radius: 0.55rem;
            padding: 0.75rem 0.9rem;
            min-height: 76px;
        }

        .meta-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            opacity: 0.6;
        }

        .meta-value {
            font-size: 1rem;
            font-weight: 650;
            margin-top: 0.2rem;
            word-break: break-word;
        }

        .review-note {
            border-left: 3px solid currentColor;
            padding: 0.65rem 0.85rem;
            background: rgba(128,128,128,.07);
            border-radius: 0 0.4rem 0.4rem 0;
            margin: 0.5rem 0 1rem;
        }

        .similar-score {
            font-size: 0.75rem;
            font-weight: 700;
            opacity: 0.7;
        }

        .similar-type {
            font-size: 0.75rem;
            opacity: 0.65;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 0.55rem;
        }

        .action-bar {
            padding-top: 0.4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading source logs...")
def _load_logs_by_id() -> dict[str, LogEntry]:
    logs = load_all_logs(DEFAULT_DB_PATH)
    return {log.log_id: log for log in logs}


def _refresh_queue():
    """Always query the queue so actions are reflected immediately."""
    return get_review_queue(DEFAULT_EVAL_DB_PATH)


def _confidence_bucket(score: float) -> tuple[str, str]:
    if score < 0.65:
        return "High review priority", "Low confidence — inspect carefully."
    if score < 0.75:
        return "Review required", "Moderate confidence — human validation is required."
    return "Higher confidence", "This case is closer to the approval threshold."


def _meta_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="meta-card">
            <div class="meta-label">{label}</div>
            <div class="meta-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_source_log(log: LogEntry | None, source_log_id: str) -> None:
    st.markdown('<div class="section-label">Source evidence</div>', unsafe_allow_html=True)
    st.markdown("### Original interaction")

    if log is None:
        st.error(
            f"The source log `{source_log_id}` could not be found in "
            "`raw_logs.duckdb`. The evaluation case cannot be fully verified."
        )
        return

    meta_1, meta_2, meta_3, meta_4 = st.columns(4)
    with meta_1:
        _meta_card("Feature", log.feature_name)
    with meta_2:
        _meta_card("Feedback", str(log.user_feedback or "none"))
    with meta_3:
        _meta_card("Error", str(log.error_status or "none"))
    with meta_4:
        _meta_card("Retries", str(log.retry_count))

    st.markdown("#### User prompt")
    st.code(log.prompt, language=None)

    st.markdown("#### Original model response")
    st.code(log.response, language=None)

    with st.expander("Additional source metadata"):
        metadata = {
            "log_id": log.log_id,
            "timestamp": str(log.timestamp),
            "model": log.model,
            "latency_ms": log.latency_ms,
            "prompt_tokens": log.prompt_tokens,
            "completion_tokens": log.completion_tokens,
            "redacted": log.redacted,
            "redaction_method": log.redaction_method,
            "safety_edge_case": log.is_safety_edge_case,
        }
        st.json(metadata)


def _render_proposed_case(case) -> dict:
    st.markdown('<div class="section-label">Evaluation definition</div>', unsafe_allow_html=True)
    st.markdown("### Proposed evaluation case")
    st.caption(
        "Review the generated definition against the original interaction. "
        "Edit only when the proposed evaluation is inaccurate, ambiguous, "
        "or incomplete."
    )

    edited_expected_behavior = None
    edited_rubric = None
    edited_forbidden = None

    if case.eval_type in ("golden_answer", "expected_refusal"):
        edited_expected_behavior = st.text_area(
            "Expected behavior",
            case.expected_behavior or "",
            height=190,
            key=f"expected_{case.case_id}",
            help="What a correct model response should do. Keep this observable and testable.",
        )
    else:
        rubric_text = "\n".join(case.rubric or [])
        edited_rubric_text = st.text_area(
            "Evaluation rubric",
            rubric_text,
            height=180,
            key=f"rubric_{case.case_id}",
            help="Use one concrete, independently checkable assertion per line.",
        )
        edited_rubric = [
            line.strip()
            for line in edited_rubric_text.splitlines()
            if line.strip()
        ]

        forbidden_text = "\n".join(case.forbidden_assertions or [])
        edited_forbidden_text = st.text_area(
            "Forbidden assertions",
            forbidden_text,
            height=120,
            key=f"forbidden_{case.case_id}",
            help="Behaviors that should count as a failure.",
        )
        edited_forbidden = [
            line.strip()
            for line in edited_forbidden_text.splitlines()
            if line.strip()
        ]

    col_1, col_2 = st.columns(2)
    with col_1:
        edited_difficulty = st.selectbox(
            "Difficulty",
            DIFFICULTY_OPTIONS,
            index=DIFFICULTY_OPTIONS.index(case.difficulty),
            key=f"difficulty_{case.case_id}",
        )
    with col_2:
        edited_tags_text = st.text_input(
            "Tags",
            ", ".join(case.tags),
            key=f"tags_{case.case_id}",
            help="Comma-separated labels useful for later dataset filtering.",
        )

    edited_tags = [tag.strip() for tag in edited_tags_text.split(",") if tag.strip()]

    with st.expander("Reviewer checklist", expanded=True):
        st.markdown(
            """
            Before approving, verify:

            - **Task alignment:** the evaluation actually tests the behavior in the source prompt.
            - **Correctness:** expected behavior or rubric is factually correct.
            - **Testability:** another reviewer could judge the model consistently.
            - **No evaluation leakage:** the case does not tell the evaluated model that it is being tested.
            - **No unnecessary assumptions:** criteria do not require information absent from the task.
            - **Appropriate difficulty:** the selected difficulty matches the reasoning required.
            """
        )

    fields = {
        "difficulty": edited_difficulty,
        "tags": edited_tags,
    }

    if edited_expected_behavior is not None:
        fields["expected_behavior"] = edited_expected_behavior
    if edited_rubric is not None:
        fields["rubric"] = edited_rubric
        fields["forbidden_assertions"] = edited_forbidden

    return fields


def _render_similar_cases(case) -> None:
    st.markdown('<div class="section-label">Reviewer context</div>', unsafe_allow_html=True)
    st.markdown("### Similar approved cases")
    st.caption(
        "These are existing approved evaluations with similar input semantics. "
        "Use them as consistency references, not as automatic answers."
    )

    similar = find_similar_approved_cases(
        case.input.prompt,
        DEFAULT_EVAL_DB_PATH,
        k=3,
    )

    if not similar:
        st.info("No approved cases are available for comparison yet.")
        return

    for index, (similar_case, score) in enumerate(similar, start=1):
        with st.container(border=True):
            st.markdown(
                f"**Reference {index}**  "
                f'<span class="similar-score">similarity {score:.2f}</span>  '
                f'<span class="similar-type">{similar_case.eval_type}</span>',
                unsafe_allow_html=True,
            )
            st.caption(similar_case.input.prompt[:240])

            if similar_case.expected_behavior:
                st.write(similar_case.expected_behavior[:350])
            elif similar_case.rubric:
                for assertion in similar_case.rubric[:3]:
                    st.write(f"- {assertion}")


def _build_header(case, position: int, total: int) -> None:
    priority, explanation = _confidence_bucket(case.confidence_score)

    st.markdown(
        f"""
        <div class="review-header">
            <div class="section-label">Human evaluation</div>
            <div class="case-title">Review case {position + 1} of {total}</div>
            <div class="case-subtitle">
                Validate the generated evaluation before it enters the approved dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(
        (position + 1) / total,
        text=f"Queue progress: {position + 1} / {total}",
    )

    meta = st.columns(6)
    values = [
        ("Case ID", str(case.case_id)),
        ("Evaluation type", str(case.eval_type)),
        ("Difficulty", str(case.difficulty)),
        ("Confidence", f"{case.confidence_score:.2f}"),
        ("Priority", priority),
        ("Cluster", str(case.source_cluster_id)),
    ]

    for column, (label, value) in zip(meta, values):
        with column:
            _meta_card(label, value)

    st.markdown(
        f'<div class="review-note">{explanation}</div>',
        unsafe_allow_html=True,
    )


def _render_actions(case, position: int) -> None:
    st.markdown("---")
    st.markdown('<div class="section-label">Decision</div>', unsafe_allow_html=True)
    st.markdown("### Reviewer decision")
    st.caption(
        "Approve when the proposed evaluation is correct as written. "
        "Use edit-and-approve when the concept is valid but needs correction. "
        "Reject when the case should not enter the evaluation dataset."
    )

    edited_fields = st.session_state.get(
        f"edited_fields_{case.case_id}",
        None,
    )

    approve_col, edit_col, reject_col = st.columns([1, 1.2, 1])

    with approve_col:
        if st.button(
            "Approve as-is",
            use_container_width=True,
            type="primary",
            key=f"approve_{case.case_id}",
        ):
            apply_review_action(
                case,
                "approve",
                reviewer_id=REVIEWER_ID,
                eval_db_path=DEFAULT_EVAL_DB_PATH,
            )
            st.rerun()

    with edit_col:
        if st.button(
            "Save edits and approve",
            use_container_width=True,
            key=f"edit_approve_{case.case_id}",
        ):
            if edited_fields is None:
                st.warning("No edited fields are available. Review the editable fields above first.")
            else:
                apply_review_action(
                    case,
                    "edit_approve",
                    reviewer_id=REVIEWER_ID,
                    edited_fields=edited_fields,
                    eval_db_path=DEFAULT_EVAL_DB_PATH,
                )
                st.rerun()

    with reject_col:
        with st.popover(
            "Reject case",
            use_container_width=True,
        ):
            reason = st.text_area(
                "Why should this case be rejected?",
                key=f"reject_reason_{case.case_id}",
                placeholder="Explain the substantive reason this case should not enter the dataset.",
                height=110,
            )
            if st.button(
                "Confirm rejection",
                use_container_width=True,
                key=f"confirm_reject_{case.case_id}",
            ):
                if not reason.strip():
                    st.error("A rejection reason is required.")
                else:
                    apply_review_action(
                        case,
                        "reject",
                        reviewer_id=REVIEWER_ID,
                        reject_reason=reason.strip(),
                        eval_db_path=DEFAULT_EVAL_DB_PATH,
                    )
                    st.rerun()


def _render_navigation(position: int, total: int) -> None:
    previous_col, spacer_col, next_col = st.columns([1, 3, 1])

    with previous_col:
        if st.button(
            "Previous case",
            disabled=position == 0,
            use_container_width=True,
        ):
            st.session_state.queue_position = max(0, position - 1)
            st.rerun()

    with next_col:
        if st.button(
            "Skip to next",
            disabled=position >= total - 1,
            use_container_width=True,
        ):
            st.session_state.queue_position = position + 1
            st.rerun()


def main() -> None:
    logs_by_id = _load_logs_by_id()
    queue = _refresh_queue()

    if "queue_position" not in st.session_state:
        st.session_state.queue_position = 0

    st.title("EvalLoop Review Workspace")
    st.caption(
        "Human validation of LLM-generated evaluation cases. "
        "The queue prioritizes lower-confidence cases first."
    )

    if not queue:
        st.success("Review queue is empty.")
        st.markdown(
            """
            ### All draft cases have been reviewed

            Approved cases are now available to the downstream dataset pipeline.
            If new draft cases are generated later, they will appear here
            automatically.
            """
        )
        return

    position = min(st.session_state.queue_position, len(queue) - 1)
    case = queue[position]

    _build_header(case, position, len(queue))

    left, center, right = st.columns([1.15, 1.35, 0.95], gap="large")

    with left:
        log = logs_by_id.get(case.source_log_id)
        _render_source_log(log, case.source_log_id)

    with center:
        edited_fields = _render_proposed_case(case)
        st.session_state[f"edited_fields_{case.case_id}"] = edited_fields

    with right:
        _render_similar_cases(case)

    _render_actions(case, position)
    _render_navigation(position, len(queue))


if __name__ == "__main__":
    main()
