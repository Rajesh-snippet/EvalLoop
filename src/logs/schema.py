"""
LogEntry schema — the unified format every synthetic (and, later, real) production
log gets normalized into before it enters the rest of the EvalLoop pipeline.

Design notes:
- error_status is a free-form string bucket rather than an enum so the synthetic
  generator (and any future real log adapter) can introduce new error types
  without a schema migration. Known values used by this project are listed in
  ERROR_STATUS_VALUES for reference/validation in tests.
- redacted / redaction_method are tracked separately from the redaction itself
  so we always know *how* a log was cleaned, not just *whether* it was.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

FeedbackType = Literal["positive", "negative", "none"]
RedactionMethod = Literal["regex_v1", "none"]

# Known error buckets used by the synthetic generator. Not enforced by the
# schema (kept as a plain string) so future real-log ingestion isn't blocked
# by an enum mismatch — validated against this list in tests instead.
ERROR_STATUS_VALUES = [
    "timeout",
    "malformed_output",
    "refusal_incorrect",
    "context_overflow",
    "rate_limited",
]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class LogEntry(BaseModel):
    log_id: str = Field(default_factory=lambda: new_id("log"))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    feature_name: str
    model: str

    system_prompt: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)

    latency_ms: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)

    user_feedback: FeedbackType = "none"
    retry_count: int = Field(default=0, ge=0)
    error_status: Optional[str] = None

    redacted: bool = False
    redaction_method: Optional[RedactionMethod] = None

    is_safety_edge_case: bool = False

    @model_validator(mode="after")
    def _redaction_method_consistency(self) -> "LogEntry":
        if self.redacted and self.redaction_method in (None, "none"):
            raise ValueError("redacted=True requires a real redaction_method (e.g. 'regex_v1')")
        if not self.redacted and self.redaction_method not in (None, "none"):
            raise ValueError("redaction_method should be None/'none' when redacted=False")
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "log_id": "log_a1b2c3d4e5f6",
                "timestamp": "2026-08-10T14:22:00Z",
                "feature_name": "customer_support_bot",
                "model": "llama-3.3-70b-versatile",
                "system_prompt": "You are a helpful customer support assistant.",
                "prompt": "My order #4521 hasn't arrived, what do I do?",
                "response": "I'm sorry to hear that! Let me help you track order #4521...",
                "latency_ms": 842,
                "prompt_tokens": 34,
                "completion_tokens": 61,
                "user_feedback": "positive",
                "retry_count": 0,
                "error_status": None,
                "redacted": False,
                "redaction_method": None,
                "is_safety_edge_case": False,
            }
        }
