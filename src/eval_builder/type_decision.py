"""
Eval type decision (Phase 3.1) — deterministic, rule-based, NOT an LLM call.
Kept explainable and cheap: given a LogEntry + its cluster context, decide
whether it needs a golden_answer, a rubric, or an expected_refusal.

Rules, in priority order (first match wins):
  1. is_safety_edge_case=True                          -> expected_refusal
  2. feature has a single "correct" style of answer     -> golden_answer
     (structured/deterministic output: SQL generation, code review with a
     specific bug to flag)
  3. everything else (open-ended, subjective quality)    -> rubric

FEATURE_EVAL_TYPE_HINTS is the only place that encodes domain knowledge
about which features are "structured" vs "open-ended" — deliberately a
plain dict at module level, not buried in conditionals, so it's the one
place to look/edit if a feature's classification is wrong.
"""

from __future__ import annotations

from src.eval_builder.eval_case_schema import EvalType
from src.logs.schema import LogEntry

# Features whose correct answer is structured/deterministic enough to expect
# a golden answer rather than a rubric. Anything not listed here defaults to
# "rubric" (the safe default for open-ended, subjective-quality domains).
FEATURE_EVAL_TYPE_HINTS: dict[str, EvalType] = {
    "sql_query_generator": "golden_answer",
    "code_review_assistant": "golden_answer",
}


def decide_eval_type(log: LogEntry) -> EvalType:
    if log.is_safety_edge_case:
        return "expected_refusal"
    return FEATURE_EVAL_TYPE_HINTS.get(log.feature_name, "rubric")


def decide_difficulty(log: LogEntry) -> str:
    """Simple, explainable difficulty heuristic — not ML-derived, just a few
    signals that plausibly correlate with how hard a case is to get right.
    Deliberately conservative: defaults to 'medium' unless there's a clear
    signal pushing it either way."""
    if log.is_safety_edge_case:
        return "hard"
    if log.error_status is not None or log.retry_count > 0:
        return "hard"
    if log.user_feedback == "negative":
        return "medium"
    return "easy"
