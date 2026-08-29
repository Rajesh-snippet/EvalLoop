"""
DuckDB storage layer for EvalCase records (Phase 3.4) — separate database
(data/eval_cases.duckdb) from the raw logs DB, per the build plan's file
structure. Kept as its own module rather than added to src/utils/db.py so
this file is self-contained and doesn't require re-touching Phase 1/2 code.

Also stores the dedup rejection log (DedupLogEntry) in its own table, so
"why was this candidate rejected" stays queryable/auditable rather than
silently discarded — per the build plan: "Track why each candidate was
accepted or rejected."
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from src.eval_builder.eval_case_schema import EvalCase, EvalCaseInput

DEFAULT_EVAL_DB_PATH = "data/eval_cases.duckdb"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS eval_cases (
    case_id VARCHAR PRIMARY KEY,
    source_log_id VARCHAR,
    source_cluster_id INTEGER,
    input_system_prompt VARCHAR,
    input_prompt VARCHAR,
    eval_type VARCHAR,
    expected_behavior VARCHAR,
    rubric VARCHAR,              -- JSON-encoded list, or NULL
    forbidden_assertions VARCHAR, -- JSON-encoded list, or NULL
    difficulty VARCHAR,
    tags VARCHAR,                 -- JSON-encoded list
    confidence_score DOUBLE,
    label_source VARCHAR,
    status VARCHAR,
    reviewer_id VARCHAR,
    review_notes VARCHAR,
    created_at TIMESTAMP,
    approved_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dedup_rejections (
    rejection_id VARCHAR PRIMARY KEY,
    source_log_id VARCHAR,
    rejected_at TIMESTAMP,
    most_similar_case_id VARCHAR,
    similarity DOUBLE,
    reason VARCHAR
);
"""


def init_eval_db(db_path: str = DEFAULT_EVAL_DB_PATH) -> duckdb.DuckDBPyConnection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute(SCHEMA_SQL)
    return con


def _case_to_row(c: EvalCase) -> tuple:
    return (
        c.case_id, c.source_log_id, c.source_cluster_id,
        c.input.system_prompt, c.input.prompt,
        c.eval_type, c.expected_behavior,
        json.dumps(c.rubric) if c.rubric is not None else None,
        json.dumps(c.forbidden_assertions) if c.forbidden_assertions is not None else None,
        c.difficulty, json.dumps(c.tags),
        c.confidence_score, c.label_source, c.status,
        c.reviewer_id, c.review_notes, c.created_at, c.approved_at,
    )


def save_eval_case(case: EvalCase, db_path: str = DEFAULT_EVAL_DB_PATH) -> None:
    con = init_eval_db(db_path)
    con.execute(
        """INSERT OR REPLACE INTO eval_cases VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        _case_to_row(case),
    )
    con.close()


def save_eval_cases(cases: list[EvalCase], db_path: str = DEFAULT_EVAL_DB_PATH) -> int:
    if not cases:
        return 0
    con = init_eval_db(db_path)
    con.executemany(
        """INSERT OR REPLACE INTO eval_cases VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [_case_to_row(c) for c in cases],
    )
    con.close()
    return len(cases)


def _row_to_case(row: tuple, columns: list[str]) -> EvalCase:
    d = dict(zip(columns, row))
    return EvalCase(
        case_id=d["case_id"],
        source_log_id=d["source_log_id"],
        source_cluster_id=d["source_cluster_id"],
        input=EvalCaseInput(system_prompt=d["input_system_prompt"], prompt=d["input_prompt"]),
        eval_type=d["eval_type"],
        expected_behavior=d["expected_behavior"],
        rubric=json.loads(d["rubric"]) if d["rubric"] else None,
        forbidden_assertions=json.loads(d["forbidden_assertions"]) if d["forbidden_assertions"] else None,
        difficulty=d["difficulty"],
        tags=json.loads(d["tags"]) if d["tags"] else [],
        confidence_score=d["confidence_score"],
        label_source=d["label_source"],
        status=d["status"],
        reviewer_id=d["reviewer_id"],
        review_notes=d["review_notes"],
        created_at=d["created_at"],
        approved_at=d["approved_at"],
    )


def load_all_eval_cases(db_path: str = DEFAULT_EVAL_DB_PATH) -> list[EvalCase]:
    if not Path(db_path).exists():
        return []
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute("SELECT * FROM eval_cases ORDER BY case_id").fetchall()
    columns = [d[0] for d in con.description]
    con.close()
    return [_row_to_case(row, columns) for row in rows]


def load_eval_cases_by_status(status: str, db_path: str = DEFAULT_EVAL_DB_PATH) -> list[EvalCase]:
    if not Path(db_path).exists():
        return []
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute(
        "SELECT * FROM eval_cases WHERE status = ? ORDER BY case_id", [status]
    ).fetchall()
    columns = [d[0] for d in con.description]
    con.close()
    return [_row_to_case(row, columns) for row in rows]


def log_dedup_rejection(
    rejection_id: str,
    source_log_id: str,
    rejected_at,
    most_similar_case_id: str,
    similarity: float,
    reason: str,
    db_path: str = DEFAULT_EVAL_DB_PATH,
) -> None:
    con = init_eval_db(db_path)
    con.execute(
        "INSERT OR REPLACE INTO dedup_rejections VALUES (?, ?, ?, ?, ?, ?)",
        [rejection_id, source_log_id, rejected_at, most_similar_case_id, similarity, reason],
    )
    con.close()


def eval_case_summary_stats(db_path: str = DEFAULT_EVAL_DB_PATH) -> dict:
    if not Path(db_path).exists():
        return {"total_cases": 0, "by_status": {}, "by_eval_type": {}, "by_difficulty": {},
                "avg_confidence": None, "dedup_rejections": 0}
    con = duckdb.connect(db_path, read_only=True)
    total = con.execute("SELECT COUNT(*) FROM eval_cases").fetchone()[0]
    by_status = dict(con.execute("SELECT status, COUNT(*) FROM eval_cases GROUP BY status").fetchall())
    by_type = dict(con.execute("SELECT eval_type, COUNT(*) FROM eval_cases GROUP BY eval_type").fetchall())
    by_difficulty = dict(con.execute("SELECT difficulty, COUNT(*) FROM eval_cases GROUP BY difficulty").fetchall())
    avg_confidence = con.execute("SELECT AVG(confidence_score) FROM eval_cases").fetchone()[0]
    n_rejections = con.execute("SELECT COUNT(*) FROM dedup_rejections").fetchone()[0]
    con.close()
    return {
        "total_cases": total,
        "by_status": by_status,
        "by_eval_type": by_type,
        "by_difficulty": by_difficulty,
        "avg_confidence": avg_confidence,
        "dedup_rejections": n_rejections,
    }
