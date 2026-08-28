"""
EvalCase schema (Phase 3) — the record produced for every log that gets
turned into an eval case. Mirrors the schema laid out in the build plan,
Section 3.2.

Status lifecycle (enforced by transition_status() below, not ad-hoc updates
scattered across the codebase — per the build plan's Phase 4 guidance):

    draft --> approved
    draft --> rejected
    approved --> deprecated

No other transitions are valid — e.g. rejected cases don't get silently
re-approved, and approved cases don't go back to draft. If a rejected case
turns out to be worth reconsidering, that's a new case, not a status flip,
so the rejection reason stays a truthful historical record.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

EvalType = Literal["golden_answer", "rubric", "expected_refusal"]
Difficulty = Literal["easy", "medium", "hard"]
CaseStatus = Literal["draft", "approved", "rejected", "deprecated"]
LabelSource = Literal["auto", "human_edited"]

_VALID_TRANSITIONS: dict[CaseStatus, set[CaseStatus]] = {
    "draft": {"approved", "rejected"},
    "approved": {"deprecated"},
    "rejected": set(),
    "deprecated": set(),
}


def new_case_id() -> str:
    return f"case_{uuid.uuid4().hex[:12]}"


class EvalCaseInput(BaseModel):
    system_prompt: str
    prompt: str


class EvalCase(BaseModel):
    case_id: str = Field(default_factory=new_case_id)
    source_log_id: str
    source_cluster_id: int  # -1 for noise-sourced candidates, per Phase 2 convention

    input: EvalCaseInput
    eval_type: EvalType

    expected_behavior: Optional[str] = None       # required for golden_answer / expected_refusal
    rubric: Optional[list[str]] = None             # required for rubric type
    forbidden_assertions: Optional[list[str]] = None

    difficulty: Difficulty = "medium"
    tags: list[str] = Field(default_factory=list)

    confidence_score: float = Field(ge=0.0, le=1.0)
    label_source: LabelSource = "auto"

    status: CaseStatus = "draft"
    reviewer_id: Optional[str] = None
    review_notes: Optional[str] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    approved_at: Optional[datetime] = None

    @model_validator(mode="after")
    def _eval_type_fields_consistency(self) -> "EvalCase":
        if self.eval_type in ("golden_answer", "expected_refusal") and not (
            self.expected_behavior and self.expected_behavior.strip()
        ):
            raise ValueError(f"eval_type='{self.eval_type}' requires a non-empty expected_behavior")
        if self.eval_type == "rubric" and not self.rubric:
            raise ValueError("eval_type='rubric' requires a non-empty rubric list")
        return self

    @model_validator(mode="after")
    def _approved_at_consistency(self) -> "EvalCase":
        if self.status == "approved" and self.approved_at is None:
            raise ValueError("status='approved' requires approved_at to be set")
        if self.status != "approved" and self.approved_at is not None:
            raise ValueError(f"approved_at should only be set when status='approved', got status='{self.status}'")
        return self


def transition_status(
    case: EvalCase,
    new_status: CaseStatus,
    reviewer_id: Optional[str] = None,
    review_notes: Optional[str] = None,
) -> EvalCase:
    """Returns a NEW EvalCase with the status transition applied (EvalCase is
    treated as immutable here — callers persist the returned copy rather than
    mutating in place, so a bug can't silently leave a half-updated record).

    Raises ValueError on any transition not in _VALID_TRANSITIONS.
    """
    allowed = _VALID_TRANSITIONS.get(case.status, set())
    if new_status not in allowed:
        raise ValueError(
            f"Invalid status transition: '{case.status}' -> '{new_status}'. "
            f"Allowed from '{case.status}': {sorted(allowed) or '(none, terminal state)'}"
        )

    update = {
        "status": new_status,
        "reviewer_id": reviewer_id if reviewer_id is not None else case.reviewer_id,
        "review_notes": review_notes if review_notes is not None else case.review_notes,
    }
    if new_status == "approved":
        update["approved_at"] = datetime.now(timezone.utc)

    return case.model_copy(update=update)
