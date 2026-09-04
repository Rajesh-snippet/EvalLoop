"""
EvalLoop Phase 5 — Metrics.

Pass-rate computation and regression diffing across eval_runs.duckdb runs.
A "regression" is a case that passed in an earlier run and fails in a
later one; an "improvement" is the reverse. Surfaced as a table of
case_ids, not just an aggregate number — an aggregate pass rate can stay
flat while individual cases silently flip, which is exactly the failure
mode this system exists to catch.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from src.eval_runner.runner import DEFAULT_RUNS_DB_PATH


def _connect(runs_db_path: str) -> duckdb.DuckDBPyConnection | None:
    if not Path(runs_db_path).exists():
        return None
    return duckdb.connect(runs_db_path, read_only=True)


def list_runs(runs_db_path: str = DEFAULT_RUNS_DB_PATH) -> list[dict]:
    """Return all distinct runs, most recent first."""
    con = _connect(runs_db_path)
    if con is None:
        return []

    rows = con.execute(
        """
        SELECT run_id, MIN(timestamp) AS timestamp, COUNT(*) AS total,
               SUM(CASE WHEN passed THEN 1 ELSE 0 END) AS passed
        FROM eval_runs
        GROUP BY run_id
        ORDER BY timestamp DESC
        """
    ).fetchall()
    con.close()

    return [
        {
            "run_id": row[0],
            "timestamp": row[1],
            "total": row[2],
            "passed": row[3],
            "pass_rate": (row[3] / row[2]) if row[2] else 0.0,
        }
        for row in rows
    ]


def pass_rate(run_id: str, runs_db_path: str = DEFAULT_RUNS_DB_PATH) -> float | None:
    """Pass rate (0-1) for a single run. None if the run has no rows."""
    con = _connect(runs_db_path)
    if con is None:
        return None

    row = con.execute(
        "SELECT COUNT(*), SUM(CASE WHEN passed THEN 1 ELSE 0 END) FROM eval_runs WHERE run_id = ?",
        [run_id],
    ).fetchone()
    con.close()

    total, passed = row
    if not total:
        return None
    return passed / total


def _run_results(con: duckdb.DuckDBPyConnection, run_id: str) -> dict[str, bool]:
    rows = con.execute(
        "SELECT case_id, passed FROM eval_runs WHERE run_id = ?",
        [run_id],
    ).fetchall()
    return {case_id: bool(passed) for case_id, passed in rows}


def regression_diff(
    previous_run_id: str,
    current_run_id: str,
    runs_db_path: str = DEFAULT_RUNS_DB_PATH,
) -> dict:
    """
    Compare two runs case-by-case.

    Returns a dict with:
        regressions: case_ids that passed previously, fail now
        improvements: case_ids that failed previously, pass now
        unchanged_pass / unchanged_fail: case_ids stable in either direction
        only_in_previous / only_in_current: case_ids present in one run but not the other
          (e.g. new cases approved since the previous run, or cases since deprecated)
    """
    con = _connect(runs_db_path)
    if con is None:
        return {
            "regressions": [], "improvements": [],
            "unchanged_pass": [], "unchanged_fail": [],
            "only_in_previous": [], "only_in_current": [],
        }

    previous = _run_results(con, previous_run_id)
    current = _run_results(con, current_run_id)
    con.close()

    shared = previous.keys() & current.keys()

    regressions = sorted(cid for cid in shared if previous[cid] and not current[cid])
    improvements = sorted(cid for cid in shared if not previous[cid] and current[cid])
    unchanged_pass = sorted(cid for cid in shared if previous[cid] and current[cid])
    unchanged_fail = sorted(cid for cid in shared if not previous[cid] and not current[cid])
    only_in_previous = sorted(previous.keys() - current.keys())
    only_in_current = sorted(current.keys() - previous.keys())

    return {
        "regressions": regressions,
        "improvements": improvements,
        "unchanged_pass": unchanged_pass,
        "unchanged_fail": unchanged_fail,
        "only_in_previous": only_in_previous,
        "only_in_current": only_in_current,
    }


def latest_two_run_ids(runs_db_path: str = DEFAULT_RUNS_DB_PATH) -> tuple[str | None, str | None]:
    """Convenience helper: (previous_run_id, current_run_id) or (None, ...) if <2 runs exist."""
    runs = list_runs(runs_db_path)
    if len(runs) < 2:
        return None, (runs[0]["run_id"] if runs else None)
    return runs[1]["run_id"], runs[0]["run_id"]
