"""
EvalLoop Phase 6 — API layer.

Thin FastAPI wrapper over the existing pipeline modules. No new business
logic lives here — every endpoint calls straight into the already-tested
functions from Phases 1-5. Local/demo use only: no auth.

Run:
    uvicorn src.api.main:app --reload

Docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.eval_builder.eval_case_db import (
    DEFAULT_EVAL_DB_PATH,
    eval_case_summary_stats,
    load_eval_cases_by_status,
)
from src.eval_runner.metrics import (
    DEFAULT_RUNS_DB_PATH,
    list_runs,
    pass_rate,
    regression_diff,
)
from src.eval_runner.runner import run_eval
from src.export.jsonl_exporter import export_approved_cases
from src.review.queue import apply_review_action, find_similar_approved_cases, get_review_queue
from src.utils.db import DEFAULT_DB_PATH, summary_stats

app = FastAPI(
    title="EvalLoop API",
    description="Production log-to-eval dataset builder — pipeline control surface.",
    version="1.0.0",
)

# --- In-memory job registry for background eval runs -------------------
# Local/demo scope: a single-process dict is fine here. A restart loses
# in-flight job status, but eval_runs.duckdb itself is untouched, so no
# data is lost — only the "is it still running?" flag.
_JOBS: dict[str, dict] = {}


# --- Schemas -------------------------------------------------------------

class ReviewActionRequest(BaseModel):
    case_id: str
    action: Literal["approve", "edit_approve", "reject"]
    reviewer_id: str
    edits: Optional[dict] = None
    reason: Optional[str] = None


class EvalRunRequest(BaseModel):
    limit: Optional[int] = None


# --- Health ---------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


# --- Logs (Phase 1) --------------------------------------------------------

@app.get("/logs/summary")
def logs_summary():
    """Summary stats over the raw synthetic log set."""
    try:
        return summary_stats(DEFAULT_DB_PATH)
    except FileNotFoundError:
        raise HTTPException(404, "raw_logs.duckdb not found — run Phase 1 first.")


# --- Eval cases (Phase 3) ---------------------------------------------------

@app.get("/eval-cases/summary")
def eval_cases_summary():
    """Draft/approved/rejected counts and related stats."""
    return eval_case_summary_stats(DEFAULT_EVAL_DB_PATH)


@app.get("/eval-cases")
def eval_cases(status: Literal["draft", "approved", "rejected", "deprecated"] = Query(...)):
    """List eval cases by status."""
    cases = load_eval_cases_by_status(status, DEFAULT_EVAL_DB_PATH)
    return [case.model_dump() for case in cases]


# --- Review queue (Phase 4) --------------------------------------------------

@app.get("/review/queue")
def review_queue(limit: Optional[int] = None):
    """Draft cases ordered lowest-confidence-first, ready for human review."""
    queue = get_review_queue(DEFAULT_EVAL_DB_PATH)
    if limit is not None:
        queue = queue[:limit]
    return [case.model_dump() for case in queue]


@app.get("/review/similar/{case_id}")
def review_similar(case_id: str, top_k: int = 3):
    """Approved cases most similar to the given draft case, for reviewer context."""
    try:
        return find_similar_approved_cases(case_id, DEFAULT_EVAL_DB_PATH, top_k=top_k)
    except KeyError:
        raise HTTPException(404, f"case_id '{case_id}' not found.")


@app.post("/review/action")
def review_action(request: ReviewActionRequest):
    """Apply approve / edit_approve / reject to a draft case."""
    try:
        result = apply_review_action(
            case_id=request.case_id,
            action=request.action,
            reviewer_id=request.reviewer_id,
            edits=request.edits,
            reason=request.reason,
            eval_db_path=DEFAULT_EVAL_DB_PATH,
        )
        return result.model_dump() if hasattr(result, "model_dump") else result
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except KeyError:
        raise HTTPException(404, f"case_id '{request.case_id}' not found.")


# --- Export (Phase 5) -------------------------------------------------------

@app.post("/export")
def export():
    """Export all approved cases to a new versioned JSONL file. Fast, no API calls."""
    return export_approved_cases(eval_db_path=DEFAULT_EVAL_DB_PATH, data_dir="data")


# --- Eval runs (Phase 5) -----------------------------------------------------

def _execute_run(run_id: str, limit: Optional[int]) -> None:
    _JOBS[run_id] = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        result = run_eval(runs_db_path=DEFAULT_RUNS_DB_PATH, limit=limit, run_id=run_id)
        _JOBS[run_id] = {
            **_JOBS[run_id],
            "status": "completed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "result": result,
        }
    except Exception as exc:  # noqa: BLE001 — surface to the polling client, don't crash the thread silently
        _JOBS[run_id] = {
            **_JOBS[run_id],
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        }


@app.post("/eval-runs")
def start_eval_run(request: EvalRunRequest):
    """
    Kick off an eval run in the background (calls Groq — can take minutes)
    and return immediately with a run_id to poll.
    """
    run_id = str(uuid.uuid4())
    _JOBS[run_id] = {"status": "queued"}
    thread = threading.Thread(target=_execute_run, args=(run_id, request.limit), daemon=True)
    thread.start()
    return {"run_id": run_id, "status": "queued"}


@app.get("/eval-runs/{run_id}/status")
def eval_run_status(run_id: str):
    """
    Poll job status. While running, also reports live progress by counting
    rows already written to eval_runs.duckdb for this run_id.
    """
    job = _JOBS.get(run_id)
    if job is None:
        raise HTTPException(404, f"No job found for run_id '{run_id}' (server may have restarted).")

    response = dict(job)
    if job["status"] in ("running", "queued"):
        current_rate = pass_rate(run_id, DEFAULT_RUNS_DB_PATH)
        response["cases_completed_so_far"] = (
            None if current_rate is None else "in progress — see /eval-runs for row counts"
        )
    return response


@app.get("/eval-runs")
def eval_runs():
    """All completed eval runs, most recent first."""
    return list_runs(DEFAULT_RUNS_DB_PATH)


@app.get("/eval-runs/regression")
def eval_runs_regression(previous_run_id: str, current_run_id: str):
    """Case-by-case pass/fail diff between two runs."""
    return regression_diff(previous_run_id, current_run_id, DEFAULT_RUNS_DB_PATH)
