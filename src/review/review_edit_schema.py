"""
ReviewEdit schema (Phase 4) — one record per field a human reviewer changes
on a draft EvalCase during review. Per the build plan: "Track what changed
and why. Use this to improve labeling prompts and measure auto-label
quality" — e.g. if 'rubric' is the most-corrected field across many edits,
that's a concrete signal label_generator.py's rubric prompt needs work.

One EvalCase review action can produce MULTIPLE ReviewEdit records (one per
field actually changed) — approving a case with only the difficulty bumped
from 'medium' to 'hard' produces one record, not one per untouched field.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def new_edit_id() -> str:
    return f"edit_{uuid.uuid4().hex[:12]}"


class ReviewEdit(BaseModel):
    edit_id: str = Field(default_factory=new_edit_id)
    case_id: str
    field_changed: str  # e.g. "rubric", "expected_behavior", "difficulty", "tags"
    before_value: str   # stored as string even for list fields (json-dumped by the caller)
    after_value: str
    reason: str | None = None
    reviewer_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
