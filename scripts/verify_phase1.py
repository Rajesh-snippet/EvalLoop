"""
Phase 1 verification script — run this after generating synthetic logs to
sanity-check the output before moving on to Phase 2.

Usage:
    python scripts/verify_phase1.py --jsonl data/synthetic_logs.jsonl

Checks performed:
  1. Every line parses as valid JSON and validates against LogEntry schema
  2. Category/feature distribution looks reasonable (not all one feature)
  3. Redaction actually fired on at least some logs (not 0%, not 100%)
  4. Retry sequences exist and are paired correctly (retry_count>0 rows exist)
  5. Safety edge cases exist
  6. No obviously broken fields (empty prompts/responses, negative tokens, etc.)
  7. Prints sample rows from each category so you can eyeball quality
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logs.schema import LogEntry  # noqa: E402


def load_and_validate(jsonl_path: str) -> list[LogEntry]:
    logs = []
    errors = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                logs.append(LogEntry(**json.loads(line)))
            except Exception as e:
                errors.append((i, str(e)))

    print(f"[1] Schema validation: {len(logs)} valid / {len(errors)} invalid lines")
    if errors:
        print("    First few errors:")
        for i, err in errors[:5]:
            print(f"      line {i}: {err[:150]}")
        print("    ⚠ Fix the generator or investigate before trusting this dataset.")
    else:
        print("    ✅ All lines parsed and validated cleanly.")
    return logs


def check_distribution(logs: list[LogEntry]) -> None:
    print("\n[2] Feature distribution:")
    by_feature = Counter(l.feature_name for l in logs)
    for feat, count in by_feature.most_common():
        pct = count / len(logs) * 100
        bar = "█" * int(pct / 2)
        print(f"    {feat:24} {count:5} ({pct:5.1f}%) {bar}")
    if len(by_feature) < 3:
        print("    ⚠ Fewer than 3 distinct features present — expected up to 6.")
    max_share = max(by_feature.values()) / len(logs)
    if max_share > 0.5:
        print(f"    ⚠ One feature dominates ({max_share:.0%} of logs) — check random.choice(FEATURES) is working.")
    else:
        print("    ✅ Reasonably spread across features.")


def check_redaction(logs: list[LogEntry]) -> None:
    print("\n[3] Redaction:")
    redacted = [l for l in logs if l.redacted]
    rate = len(redacted) / len(logs)
    print(f"    {len(redacted)}/{len(logs)} logs redacted ({rate:.1%})")
    # Expect roughly ~15% given the PII injection rate in the generator, but
    # allow a wide band since real Groq-generated prompts may organically
    # contain phone/email-shaped text too.
    if rate == 0:
        print("    ⚠ Zero redaction fired — PII injection or redaction rules may be broken.")
    elif rate > 0.6:
        print("    ⚠ Redaction rate unexpectedly high — check for a false-positive rule firing too often.")
    else:
        print("    ✅ Redaction rate looks plausible.")
    if redacted:
        sample = redacted[0]
        print(f"    Sample redacted prompt: {sample.prompt[:100]}")


def check_failure_signals(logs: list[LogEntry]) -> None:
    print("\n[4] Failure signals (retries, errors, safety cases):")
    retries = [l for l in logs if l.retry_count > 0]
    errors = [l for l in logs if l.error_status is not None]
    safety = [l for l in logs if l.is_safety_edge_case]
    negative_fb = [l for l in logs if l.user_feedback == "negative"]

    print(f"    retry_count > 0:       {len(retries)} ({len(retries)/len(logs):.1%})")
    print(f"    error_status set:      {len(errors)} ({len(errors)/len(logs):.1%})")
    print(f"    is_safety_edge_case:   {len(safety)} ({len(safety)/len(logs):.1%})")
    print(f"    negative feedback:     {len(negative_fb)} ({len(negative_fb)/len(logs):.1%})")

    problems = []
    if not retries:
        problems.append("no retry-sequence rows found")
    if not errors:
        problems.append("no error_status rows found")
    if not safety:
        problems.append("no safety edge cases found")
    if problems:
        print(f"    ⚠ Missing signal: {', '.join(problems)} — failure-biased sampling in Phase 2 will have nothing to bias toward.")
    else:
        print("    ✅ All failure signal types present — good for Phase 2 failure-biased sampling.")


def check_field_sanity(logs: list[LogEntry]) -> None:
    print("\n[5] Field sanity checks:")
    empty_prompts = [l for l in logs if not l.prompt.strip()]
    empty_responses = [l for l in logs if not l.response.strip()]
    zero_tokens = [l for l in logs if l.prompt_tokens == 0 or l.completion_tokens == 0]
    duplicate_ids = len(logs) - len({l.log_id for l in logs})

    issues = []
    if empty_prompts:
        issues.append(f"{len(empty_prompts)} empty prompts")
    if empty_responses:
        issues.append(f"{len(empty_responses)} empty responses")
    if zero_tokens:
        issues.append(f"{len(zero_tokens)} rows with zero token counts")
    if duplicate_ids:
        issues.append(f"{duplicate_ids} duplicate log_ids")

    empty_rate = (len(empty_prompts) + len(empty_responses)) / (2 * len(logs))
    if empty_rate > 0.1:
        print(f"    🛑 FAIL: {empty_rate:.0%} of prompt/response fields are empty — dataset is unusable as-is.")
        print("       Likely cause: a reasoning model (gpt-oss) returned empty content because")
        print("       its token budget was consumed by hidden reasoning. Re-generate after fixing")
        print("       max_tokens / reasoning_effort in synthetic_generator.py, don't proceed to Phase 2.")
    elif issues:
        print(f"    ⚠ Found: {', '.join(issues)}")
    else:
        print("    ✅ No empty fields, zero-token rows, or duplicate IDs.")


def print_samples(logs: list[LogEntry]) -> None:
    print("\n[6] Sample rows for manual quality check:")
    seen_categories = set()
    for l in logs:
        key = (
            "safety" if l.is_safety_edge_case else
            "malformed" if l.error_status == "malformed_output" else
            "retry" if l.retry_count > 0 else
            "bad" if l.user_feedback == "negative" else
            "good"
        )
        if key in seen_categories:
            continue
        seen_categories.add(key)
        print(f"\n    --- category: {key} | feature: {l.feature_name} ---")
        print(f"    prompt:   {l.prompt[:150]}")
        print(f"    response: {l.response[:150]}")
        if len(seen_categories) == 5:
            break

    missing = {"safety", "malformed", "retry", "bad", "good"} - seen_categories
    if missing:
        print(f"\n    ⚠ Could not find sample rows for: {missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="data/synthetic_logs.jsonl")
    args = parser.parse_args()

    if not Path(args.jsonl).exists():
        print(f"File not found: {args.jsonl}")
        print("Run the generator first: python -m src.logs.synthetic_generator --count 50")
        sys.exit(1)

    print(f"Verifying: {args.jsonl}\n{'='*60}")
    logs = load_and_validate(args.jsonl)
    if not logs:
        print("\nNo valid logs to check further. Stopping.")
        sys.exit(1)

    check_distribution(logs)
    check_redaction(logs)
    check_failure_signals(logs)
    check_field_sanity(logs)
    print_samples(logs)

    print(f"\n{'='*60}")
    print(f"Done. {len(logs)} logs checked. Review any ⚠ warnings above before moving to Phase 2.")


if __name__ == "__main__":
    main()
