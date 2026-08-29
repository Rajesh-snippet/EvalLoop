"""
LLM-as-judge (Phase 3.3) — reviews a GENERATED LABEL for quality (not the
original log/response), and produces a confidence score in [0, 1].

Important distinction from label_generator.py: the judge never sees the
original AI response that produced this log. It only judges whether the
proposed rubric/expected_behavior itself is well-formed, specific, and
actually checkable — e.g. "the response should be helpful" is a bad rubric
line (vague, not gradable); "the response must mention the 3-week PTO
advance-notice requirement" is a good one.

Final confidence combines the judge's own quality score with
label_generator's multi-pass agreement signal:

    confidence = 0.5 * judge_score + 0.5 * pass_agreement_score

pass_agreement_score is 1.0 if passes agreed (or only 1 pass was run) and
0.3 if they disagreed — not 0.0, since disagreement doesn't necessarily mean
the final pass is wrong, just less certain; 0.3 still meaningfully pulls
overall confidence down toward the human-review threshold rather than
zeroing it out entirely.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv  # noqa: E402
from groq import Groq  # noqa: E402

from src.eval_builder.label_generator import GeneratedLabel  # noqa: E402
from src.utils.constants import GROQ_GENERATION_MODEL, REASONING_MODELS  # noqa: E402
from src.utils.retry import retry_with_backoff  # noqa: E402

load_dotenv()

JUDGE_SYSTEM_PROMPT = (
    "You are reviewing the QUALITY of a proposed evaluation label (not the "
    "original AI response it came from). Judge whether the label is "
    "well-formed, specific, and gradable by another AI judge later: are the "
    "assertions concrete and checkable, or vague and subjective? Is the "
    "expected_behavior specific enough that a grader could actually compare "
    "a candidate answer against it? "
    'Respond ONLY with JSON: {"quality_score": <float 0.0-1.0>, "reasoning": "..."}. '
    "quality_score should be low (below 0.5) if the label is vague, "
    "self-contradictory, too short to be useful, or unclear about what "
    "counts as passing."
)

DISAGREEMENT_PASS_SCORE = 0.3
AGREEMENT_PASS_SCORE = 1.0


@dataclass
class JudgeResult:
    quality_score: float
    reasoning: str
    pass_agreement_score: float
    confidence: float


def _client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set (checked environment and .env file)")
    return Groq(api_key=api_key)


def _strip_json_fence(text: str) -> str:
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


def _label_as_text(label: GeneratedLabel) -> str:
    parts = []
    if label.expected_behavior:
        parts.append(f"expected_behavior: {label.expected_behavior}")
    if label.rubric:
        parts.append("rubric:\n" + "\n".join(f"  - {r}" for r in label.rubric))
    if label.forbidden_assertions:
        parts.append("forbidden_assertions:\n" + "\n".join(f"  - {f}" for f in label.forbidden_assertions))
    return "\n".join(parts)


@retry_with_backoff(max_retries=3, base_delay=1.0)
def _judge_quality(client: Groq, label: GeneratedLabel) -> tuple[float, str]:
    user_msg = f"eval_type: {label.eval_type}\n\n{_label_as_text(label)}"
    resp = client.chat.completions.create(
        model=GROQ_GENERATION_MODEL,
        messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
        temperature=0.2,
        max_tokens=300,
        **({"reasoning_effort": "low"} if GROQ_GENERATION_MODEL in REASONING_MODELS else {}),
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Groq returned empty content during judging")
    text = _strip_json_fence(text)
    parsed = json.loads(text)

    score = float(parsed["quality_score"])
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"quality_score out of range [0,1]: {score}")
    return score, parsed.get("reasoning", "")


def judge_label(label: GeneratedLabel) -> JudgeResult:
    client = _client()
    quality_score, reasoning = _judge_quality(client, label)

    pass_agreement_score = AGREEMENT_PASS_SCORE if label.passes_agreed else DISAGREEMENT_PASS_SCORE
    confidence = 0.5 * quality_score + 0.5 * pass_agreement_score

    return JudgeResult(
        quality_score=quality_score,
        reasoning=reasoning,
        pass_agreement_score=pass_agreement_score,
        confidence=confidence,
    )
