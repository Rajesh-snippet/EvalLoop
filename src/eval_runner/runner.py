"""
EvalLoop Phase 5 — Eval Runner.

Runs every APPROVED eval case against a target model, then re-judges the
response pass/fail using a SEPARATE judge model (to avoid the
self-grading bias called out in the build plan's risk table). Results
are stored in data/eval_runs.duckdb, one row per (run_id, case_id).

Model split (deliberate, not interchangeable):
    - EVAL_RUNNER_TARGET_MODEL: the model being evaluated (produces the answer)
    - EVAL_RUNNER_JUDGE_MODEL:  a different model that grades the answer

Works with any number of approved cases, including zero (produces an
empty run with a warning rather than raising).

Usage:
    python -m src.eval_runner.runner
    python -m src.eval_runner.runner --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from dotenv import load_dotenv
from groq import Groq

from src.eval_builder.eval_case_db import (
    DEFAULT_EVAL_DB_PATH,
    load_eval_cases_by_status,
)
from src.utils.retry import retry_with_backoff

load_dotenv()

DEFAULT_RUNS_DB_PATH = "data/eval_runs.duckdb"

# Different model families deliberately: qwen (target) vs gpt-oss (judge).
# qwen/qwen3.6-27b was set up in Phase 1 as an untested fallback — this is
# its first real exercise. If it 404s / is unavailable on this account,
# fall back to a second openai/* model rather than reusing the judge model,
# to preserve the target != judge property.
EVAL_RUNNER_TARGET_MODEL = "qwen/qwen3.6-27b"
EVAL_RUNNER_JUDGE_MODEL = "openai/gpt-oss-20b"
JUDGE_REASONING_EFFORT = "low"

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set (check .env)")
        _client = Groq(api_key=api_key)
    return _client


def _init_runs_db(runs_db_path: str) -> None:
    Path(runs_db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(runs_db_path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id VARCHAR,
            case_id VARCHAR,
            timestamp TIMESTAMP,
            target_model VARCHAR,
            judge_model VARCHAR,
            eval_type VARCHAR,
            source_cluster_id VARCHAR,
            difficulty VARCHAR,
            model_response VARCHAR,
            passed BOOLEAN,
            judge_reasoning VARCHAR
        )
        """
    )
    con.close()


@retry_with_backoff(max_retries=4)
def _generate_target_response(case) -> str:
    """Call the target model with the case's original prompt."""
    client = _get_client()
    messages = []
    if case.input.system_prompt:
        messages.append({"role": "system", "content": case.input.system_prompt})
    messages.append({"role": "user", "content": case.input.prompt})

    response = client.chat.completions.create(
        model=EVAL_RUNNER_TARGET_MODEL,
        messages=messages,
        max_tokens=600,
    )
    return response.choices[0].message.content or ""


_JUDGE_SYSTEM_PROMPT = """You are grading whether a model's response satisfies an
evaluation case. You will be given the eval_type and the criteria for that type,
plus the actual model response. Return ONLY a JSON object, no markdown fences,
no preamble, shaped exactly like:

{"passed": true, "reasoning": "one or two sentences explaining the verdict"}
"""


def _build_judge_user_prompt(case, model_response: str) -> str:
    if case.eval_type in ("golden_answer",):
        criteria = f"Expected behavior:\n{case.expected_behavior}"
    elif case.eval_type == "expected_refusal":
        criteria = f"Expected refusal behavior:\n{case.expected_behavior}"
    else:
        rubric = "\n".join(f"- {line}" for line in (case.rubric or []))
        forbidden = "\n".join(f"- {line}" for line in (case.forbidden_assertions or []))
        criteria = f"Rubric (must satisfy):\n{rubric}\n\nForbidden (must NOT do):\n{forbidden}"

    return (
        f"Eval type: {case.eval_type}\n\n"
        f"{criteria}\n\n"
        f"--- Model response to grade ---\n{model_response}\n--- end response ---"
    )


@retry_with_backoff(max_retries=4)
def _judge_response(case, model_response: str) -> tuple[bool, str]:
    """Call the judge model to grade the target model's response."""
    client = _get_client()
    response = client.chat.completions.create(
        model=EVAL_RUNNER_JUDGE_MODEL,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _build_judge_user_prompt(case, model_response)},
        ],
        reasoning_effort=JUDGE_REASONING_EFFORT,
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise ValueError(
            "Judge model returned empty content — likely burned its budget on "
            "hidden reasoning. Increase max_tokens or check reasoning_effort."
        )
    parsed = json.loads(content)
    return bool(parsed["passed"]), str(parsed.get("reasoning", ""))


def run_eval(
    eval_db_path: str = DEFAULT_EVAL_DB_PATH,
    runs_db_path: str = DEFAULT_RUNS_DB_PATH,
    limit: int | None = None,
) -> dict:
    """
    Run every approved case through target model -> judge model, and store
    results as one new run_id in eval_runs.duckdb.

    Returns a summary dict: run_id, total, passed, failed.
    """
    _init_runs_db(runs_db_path)

    approved_cases = load_eval_cases_by_status("approved", eval_db_path)
    if limit is not None:
        approved_cases = approved_cases[:limit]

    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)

    if not approved_cases:
        # Still record an (empty) run so downstream tooling has something
        # consistent to query, rather than special-casing "no runs yet".
        return {"run_id": run_id, "total": 0, "passed": 0, "failed": 0, "errors": 0}

    con = duckdb.connect(runs_db_path)
    passed_count = 0
    failed_count = 0
    error_count = 0

    for case in approved_cases:
        try:
            model_response = _generate_target_response(case)
            passed, reasoning = _judge_response(case, model_response)
        except Exception as exc:  # noqa: BLE001 — log and continue, one bad case shouldn't kill the run
            model_response = ""
            passed = False
            reasoning = f"RUNNER ERROR: {exc}"
            error_count += 1

        if passed:
            passed_count += 1
        else:
            failed_count += 1

        status = "PASS" if passed else "FAIL"
        done = passed_count + failed_count
        print(f"[{done}/{len(approved_cases)}] {case.case_id} -> {status}")

        time.sleep(3)

        con.execute(
            """
            INSERT INTO eval_runs
            (run_id, case_id, timestamp, target_model, judge_model, eval_type,
             source_cluster_id, difficulty, model_response, passed, judge_reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                case.case_id,
                timestamp,
                EVAL_RUNNER_TARGET_MODEL,
                EVAL_RUNNER_JUDGE_MODEL,
                case.eval_type,
                case.source_cluster_id,
                case.difficulty,
                model_response,
                passed,
                reasoning,
            ],
        )

    con.close()

    return {
        "run_id": run_id,
        "total": len(approved_cases),
        "passed": passed_count,
        "failed": failed_count,
        "errors": error_count,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run approved eval cases against the target model.")
    parser.add_argument("--eval-db", default=DEFAULT_EVAL_DB_PATH)
    parser.add_argument("--runs-db", default=DEFAULT_RUNS_DB_PATH)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_eval(eval_db_path=args.eval_db, runs_db_path=args.runs_db, limit=args.limit)

    print(f"Run {result['run_id']}")
    print(f"  total:  {result['total']}")
    print(f"  passed: {result['passed']}")
    print(f"  failed: {result['failed']}")
    if result.get("errors"):
        print(f"  errors: {result['errors']} (counted as failed, see judge_reasoning)")
    if result["total"] == 0:
        print("  WARNING: no approved cases to run — review cases in Phase 4 first.")


if __name__ == "__main__":
    main()
