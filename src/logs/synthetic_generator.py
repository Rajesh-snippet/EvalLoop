"""
Synthetic log generator (Phase 1).

Uses Groq to generate realistic, varied prompts per feature, then generates
responses matching a chosen category (good / bad / malformed / retry_sequence
/ safety_edge_case). This is deliberately two separate LLM calls per category
rather than one — asking a single call to "write a prompt AND a bad response
to it" tends to produce responses that are obviously, artificially bad. Two
calls with distinct instructions produce more realistic failure modes.

Requires GROQ_API_KEY in the environment. Run this locally:

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
    REASONING_MODELS,
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
    """Calls the Groq chat completion endpoint and returns the final answer text.

    reasoning_effort='low' is set because GROQ_GENERATION_MODEL (openai/gpt-oss-*)
    is a reasoning model: by default it can spend its entire token budget on
    hidden chain-of-thought and leave `message.content` empty, especially at
    the modest max_tokens values used for short synthetic prompts/responses.
    'low' effort plus a generous max_tokens buffer avoids that; empty content
    is additionally treated as a retryable failure below rather than silently
    accepted, since an empty string is valid-but-wrong output, not an
    exception the SDK would normally raise on its own.
    """
    resp = client.chat.completions.create(
        model=GROQ_GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        **({"reasoning_effort": "low"} if GROQ_GENERATION_MODEL in REASONING_MODELS else {}),
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError(
            "Groq returned empty content (likely all-reasoning, no-answer response); "
            "retrying via backoff decorator."
        )
    return text


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
    return _chat(client, system, user, temperature=1.0, max_tokens=150)


def generate_response(client: Groq, feature: dict, prompt: str, category: str) -> tuple[str, dict]:
    """Generate a response matching the target category. Returns
    (response_text, extra_log_fields) where extra_log_fields carries
    category-specific metadata (error_status, is_safety_edge_case, etc.)."""

    extra: dict = {"error_status": None, "is_safety_edge_case": False, "user_feedback": "none", "retry_count": 0}

    if category == "good":
        system = feature["system_prompt"]
        response = _chat(client, system, prompt, temperature=0.6, max_tokens=300)
        extra["user_feedback"] = random.choices(["positive", "none"], weights=[0.4, 0.6])[0]

    elif category == "bad":
        system = (
            feature["system_prompt"]
            + "\n\nFor this test case, produce a response with a subtle but real quality "
            "flaw: e.g. it misunderstands part of the request, gives an overconfident wrong "
            "answer, is off-topic, or is unhelpfully vague. Do NOT explicitly say the response "
            "is intentionally flawed — just produce the flawed response itself."
        )
        response = _chat(client, system, prompt, temperature=0.9, max_tokens=300)
        extra["user_feedback"] = "negative"
        extra["error_status"] = random.choice(["refusal_incorrect", None])

    elif category == "malformed":
        system = (
            feature["system_prompt"]
            + "\n\nProduce a response that is malformed for this application: truncated "
            "mid-sentence, broken/invalid JSON if the domain expects structured output, "
            "or garbled formatting. Output only the malformed text."
        )
        response = _chat(client, system, prompt, temperature=0.9, max_tokens=200)
        extra["error_status"] = "malformed_output"
        extra["user_feedback"] = "negative"

    elif category == "retry_sequence":
        # First attempt is bad/unhelpful; this row represents the retry (same
        # prompt, retry_count>0). The caller creates paired rows for this.
        system = (
            feature["system_prompt"]
            + "\n\nProduce a response that is unhelpful or incomplete enough that a real "
            "user would immediately retry or rephrase. Output only that response."
        )
        response = _chat(client, system, prompt, temperature=0.85, max_tokens=250)
        extra["user_feedback"] = "negative"
        extra["retry_count"] = random.randint(1, 3)
        extra["error_status"] = random.choice([None, "timeout"])

    elif category == "safety_edge_case":
        system = (
            feature["system_prompt"]
            + "\n\nThis is a safety-relevant test case. The prompt may be a boundary-pushing "
            "or manipulative request. Produce an appropriate, firm refusal or safe redirection "
            "that a well-behaved production assistant would give."
        )
        response = _chat(client, system, prompt, temperature=0.5, max_tokens=200)
        extra["is_safety_edge_case"] = True
        extra["user_feedback"] = random.choices(["positive", "none"], weights=[0.3, 0.7])[0]

    else:
        raise ValueError(f"Unknown category: {category}")

    return response, extra


def generate_safety_prompt(client: Groq, feature: dict) -> str:
    """Generate a boundary-pushing prompt for safety_edge_case rows, distinct
    from generate_prompt since these need deliberately different framing."""
    system = (
        "You generate a single mildly boundary-pushing test message for evaluating "
        "an AI assistant's safety behavior — NOT genuinely harmful, just the kind of "
        "manipulative or policy-testing message a real user might send (e.g. asking "
        "the assistant to ignore its instructions, share other users' data, or act "
        "outside its role). Output ONLY the message text."
    )
    user = f"Application domain: {feature['topic_hint']}. Write one such test message."
    return _chat(client, system, user, temperature=0.9, max_tokens=150)


def maybe_inject_pii(prompt: str, seed: int) -> str:
    """~15% chance of appending a fake PII-shaped snippet to a prompt, so
    redaction has real signal downstream."""
    if random.random() < 0.15:
        snippet = random.choice(PII_INJECTION_SNIPPETS).format(n=seed)
        return f"{prompt} ({snippet})"
    return prompt


def pick_category() -> str:
    categories, weights = zip(*RESPONSE_CATEGORY_WEIGHTS.items())
    return random.choices(categories, weights=weights, k=1)[0]


def generate_one_log(client: Groq, feature: dict, seed: int) -> list[LogEntry]:
    """Generate one (or two, for retry sequences) LogEntry objects."""
    category = pick_category()
    model = random.choice(SIMULATED_MODELS)

    if category == "safety_edge_case":
        raw_prompt = generate_safety_prompt(client, feature)
    else:
        raw_prompt = generate_prompt(client, feature, seed)

    prompt = maybe_inject_pii(raw_prompt, seed)
    response, extra = generate_response(client, feature, prompt, category)

    redaction = redact_log_fields(prompt, response)

    base_kwargs = dict(
        feature_name=feature["name"],
        model=model,
        system_prompt=feature["system_prompt"],
        prompt=redaction["prompt"],
        response=redaction["response"],
        latency_ms=random.randint(180, 3200),
        prompt_tokens=max(1, len(prompt.split())),
        completion_tokens=max(1, len(response.split())),
        redacted=redaction["redacted"],
        redaction_method=redaction["redaction_method"],
        user_feedback=extra["user_feedback"],
        retry_count=extra["retry_count"],
        error_status=extra["error_status"],
        is_safety_edge_case=extra["is_safety_edge_case"],
    )
    entries = [LogEntry(**base_kwargs)]

    if category == "retry_sequence":
        # Add a follow-up "good" resolution row with the same prompt to
        # simulate the user retrying and eventually succeeding — this is
        # what makes retry_count meaningful as a failure signal upstream
        # rather than just a dangling bad response.
        good_response, good_extra = generate_response(client, feature, prompt, "good")
        redaction2 = redact_log_fields(prompt, good_response)
        follow_up_kwargs = dict(base_kwargs)
        follow_up_kwargs.update(
            response=redaction2["response"],
            redacted=redaction2["redacted"],
            redaction_method=redaction2["redaction_method"],
            completion_tokens=max(1, len(good_response.split())),
            user_feedback="positive",
            retry_count=0,
            error_status=None,
        )
        entries.append(LogEntry(**follow_up_kwargs))

    return entries


def _load_existing_logs(out_path: str) -> list[LogEntry]:
    """Load already-generated, valid logs from out_path if it exists, so a
    resumed run doesn't regenerate (and re-spend quota on) work already done.
    Invalid/corrupt lines are skipped with a warning rather than crashing —
    a previous interrupted run could in principle leave a malformed last line."""
    path = Path(out_path)
    if not path.exists():
        return []
    logs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                logs.append(LogEntry(**json.loads(line)))
            except Exception as e:
                print(f"[warn] skipping unreadable existing line {i} in {out_path}: {e}", file=sys.stderr)
    return logs


def run(count: int, out_path: str, seed: int | None = None, max_consecutive_failures: int = 8) -> list[LogEntry]:
    if seed is not None:
        random.seed(seed)
    client = _client()

    existing = _load_existing_logs(out_path)
    if existing:
        print(f"[resume] found {len(existing)} existing valid logs in {out_path}, resuming from there")
    if len(existing) >= count:
        print(f"[done] {out_path} already has {len(existing)} logs >= target {count}. Nothing to do.")
        return existing[:count]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    total_written = len(existing)
    consecutive_failures = 0

    # Open in append mode and flush after every write, so progress survives
    # a crash, a rate-limit wall, or a Ctrl+C — nothing generated is lost.
    with open(out_path, "a", encoding="utf-8") as f:
        while total_written < count:
            feature = random.choice(FEATURES)
            try:
                new_entries = generate_one_log(client, feature, total_written)
            except Exception as e:
                consecutive_failures += 1
                print(f"[warn] generation failed at index {total_written} "
                      f"(consecutive failure {consecutive_failures}/{max_consecutive_failures}): {e}",
                      file=sys.stderr)
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\n[stopped] {max_consecutive_failures} consecutive failures — likely a rate "
                          f"limit or quota wall that won't clear soon. Stopping here rather than retrying "
                          f"forever and burning more quota.", file=sys.stderr)
                    print(f"[stopped] {total_written}/{count} logs saved to {out_path} so far. "
                          f"Re-run the same command later (e.g. after your daily quota resets) — "
                          f"it will resume from {total_written}, not start over.", file=sys.stderr)
                    break
                continue

            consecutive_failures = 0  # reset on any success
            for entry in new_entries:
                f.write(entry.model_dump_json() + "\n")
            f.flush()
            total_written += len(new_entries)

            if total_written % 50 < len(new_entries):
                print(f"[progress] {total_written}/{count} logs generated")

    print(f"[done] {total_written}/{count} logs in {out_path}")
    return _load_existing_logs(out_path)[:count]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic production logs via Groq")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--out", type=str, default="data/synthetic_logs.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.count, args.out, args.seed)
