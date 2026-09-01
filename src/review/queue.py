"""
Review queue logic (Phase 4) — non-UI layer, so it's testable without
Streamlit. streamlit_app.py is a thin rendering layer over these functions.

Responsibilities per the build plan's Phase 4 spec:
  - fetch the queue of low-confidence (draft) cases
  - surface similar EXISTING (approved) cases as reviewer context, reusing
    Phase 3's dedup embedding-similarity machinery rather than reimplementing it
  - apply approve / edit+approve / reject actions through the already-built
    EvalCase status state machine, writing a ReviewEdit record for every
    field that actually changed
"""

from __future__ import annotations

import json

from src.eval_builder.eval_case_db import load_eval_cases_by_status, save_eval_case
from src.eval_builder.eval_case_schema import EvalCase, transition_status
from src.review.review_edit_schema import ReviewEdit
from src.review.review_edit_db import save_review_edits

# Fields a reviewer is allowed to edit before approving/rejecting — kept as
# an explicit whitelist so a future field added to EvalCase doesn't silently
# become editable (or worse, diffable-and-logged) without a deliberate
# decision to include it here.
EDITABLE_FIELDS = ["expected_behavior", "rubric", "forbidden_assertions", "tags", "difficulty"]


def get_review_queue(eval_db_path: str) -> list[EvalCase]:
    """Draft cases, lowest confidence first — the ones needing the most
    scrutiny surface at the top of the queue."""
    drafts = load_eval_cases_by_status("draft", db_path=eval_db_path)
    return sorted(drafts, key=lambda c: c.confidence_score)


def find_similar_approved_cases(
    candidate_prompt: str,
    eval_db_path: str,
    k: int = 3,
) -> list[tuple[EvalCase, float]]:
    """Top-k most similar APPROVED cases to the case under review, for the
    reviewer-context panel the build plan's UI spec calls for. Returns
    (case, similarity) pairs, most similar first. Reuses embed_texts rather
    than a separate embedding path, so similarity numbers here are on the
    same scale as Phase 3's dedup threshold."""
    import numpy as np

    from src.clustering.embed import embed_texts

    approved = load_eval_cases_by_status("approved", db_path=eval_db_path)
    if not approved:
        return []

    all_prompts = [candidate_prompt] + [c.input.prompt for c in approved]
    embeddings = embed_texts(all_prompts)
    candidate_emb = embeddings[0]
    approved_embs = embeddings[1:]

    similarities = approved_embs @ candidate_emb  # cosine sim, embeddings are L2-normalized
    order = np.argsort(similarities)[::-1][:k]
    return [(approved[i], float(similarities[i])) for i in order]


def _field_to_str(value) -> str:
    """Consistent stringification for diffing/logging list vs scalar fields."""
    if isinstance(value, list):
        return json.dumps(value)
    return str(value) if value is not None else ""


def apply_review_action(
    case: EvalCase,
    action: str,  # "approve" | "edit_approve" | "reject"
    reviewer_id: str,
    edited_fields: dict | None = None,
    reject_reason: str | None = None,
    eval_db_path: str = None,
) -> EvalCase:
    """Applies a reviewer's decision: diffs any edited_fields against the
    case's current values, writes a ReviewEdit for each field that actually
    changed (skips fields passed but unchanged — an edit that reverts to the
    same value isn't a correction worth logging), transitions status via the
    existing state machine, and persists both the updated case and any
    ReviewEdit records.

    action='approve' with edited_fields=None is a plain approval, no edits.
    action='edit_approve' requires edited_fields (at least one real change).
    action='reject' requires reject_reason; edited_fields is ignored (per
    the build plan, a rejected case's original content stays as historical
    record — you don't "fix" something you're rejecting).
    """
    if action not in ("approve", "edit_approve", "reject"):
        raise ValueError(f"Unknown action: {action}")
    if action == "reject" and not reject_reason:
        raise ValueError("reject_reason is required when action='reject'")
    if action == "edit_approve" and not edited_fields:
        raise ValueError("edit_approve requires at least one field in edited_fields")

    edits_to_log: list[ReviewEdit] = []
    updated_case = case

    if edited_fields and action in ("approve", "edit_approve"):
        unknown = set(edited_fields) - set(EDITABLE_FIELDS)
        if unknown:
            raise ValueError(f"Fields not editable via review: {unknown}")

        update_dict = {}
        for field_name, new_value in edited_fields.items():
            old_value = getattr(case, field_name)
            if _field_to_str(old_value) == _field_to_str(new_value):
                continue  # no real change, don't log a no-op edit
            edits_to_log.append(
                ReviewEdit(
                    case_id=case.case_id,
                    field_changed=field_name,
                    before_value=_field_to_str(old_value),
                    after_value=_field_to_str(new_value),
                    reviewer_id=reviewer_id,
                )
            )
            update_dict[field_name] = new_value

        if update_dict:
            update_dict["label_source"] = "human_edited"
            updated_case = case.model_copy(update=update_dict)

    new_status = "rejected" if action == "reject" else "approved"
    updated_case = transition_status(
        updated_case, new_status, reviewer_id=reviewer_id,
        review_notes=reject_reason,
    )

    if eval_db_path:
        save_eval_case(updated_case, db_path=eval_db_path)
        if edits_to_log:
            save_review_edits(edits_to_log, db_path=eval_db_path)

    return updated_case
