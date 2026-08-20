"""
Synthetic log generator (Phase 1).

Uses Groq to generate realistic, varied prompts per feature, then generates
responses matching a chosen category (good / bad / malformed / retry_sequence
/ safety_edge_case). This is deliberately two separate LLM calls per category
rather than one — asking a single call to "write a prompt AND a bad response
to it" tends to produce responses that are obviously, artificially bad. Two
calls with distinct instructions produce more realistic failure modes.

Requires GROQ_API_KEY in the environment. Run this locally:

    export GROQ_API_KEY=your_key_here      # macOS/Linux
    $env:GROQ_API_KEY = "your_key_here"    # Windows PowerShell

    python -m src.logs.synthetic_generator --count 1000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from groq import Groq  # noqa: E402

from src.logs.redaction import redact_log_fields  # noqa: E402
from src.logs.schema import ERROR_STATUS_VALUES, LogEntry  # noqa: E402
from src.utils.constants import (  # noqa: E402
    FEATURES,
    GROQ_GENERATION_MODEL,
    RESPONSE_CATEGORY_WEIGHTS,
    SIMULATED_MODELS,
)
from src.utils.retry import retry_with_backoff  # noqa: E402

# A handful of raw PII-shaped snippets we deliberately inject into ~15% of
# generated prompts, so the redaction module (Phase 1.2) and later dataset
# stats have real signal to work with. These are synthetic/fake values only.
PII_INJECTION_SNIPPETS = [
    "you can reach me at jane.doe{n}@gmail.com",
    "call me back at 555-{n:03d}-4821",
    "my name is Alex Johnson, order #{n}",
    "here's my API key for reference: sk-fake{n}abcdefghijklmnopqr",
]


def _client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Export it before running the generator "
            "(see module docstring for the exact command on your OS)."
        )
    return Groq(api_key=api_key)


@retry_with_backoff(max_retries=4, base_delay=1.5)
def _chat(client: Groq, system: str, user: str, temperature: float = 0.9, max_tokens: int = 300) -> str:
    resp = client.chat.completions.create(
        model=GROQ_GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def generate_prompt(client: Groq, feature: dict, batch_seed: int) -> str:
    """Generate one realistic user prompt for a given feature."""
    system = (
        "You generate a single realistic user message for a production LLM "
        "application, for testing purposes. Output ONLY the user message text, "
        "nothing else — no quotes, no preamble, no explanation."
    )
    user = (
        f"Application domain: {feature['topic_hint']}\n"
        f"Write one realistic, natural user message a real person might send. "
        f"Vary sentence length, tone, and phrasing style from typical examples "
        f"(sometimes terse, sometimes rambling, sometimes with typos). "
        f"Variation seed: {batch_seed}."
    )
    return _chat(client, system, user, temperature=1.0, max_tokens=120)


def generate_response(client: Groq, feature: dict, prompt: str, category: str) -> tuple[str, dict]:
    """Generate a response matching the target category. Returns
    (response_text, extra_log_fields) where extra_log_fields carries
    category-specific metadata (error_status, is_safety_edge_case, etc.)."""

    extra: dict = {"error_status": None, "is_safety_edge_case": False, "user_feedback": "none", "retry_count": 0}
