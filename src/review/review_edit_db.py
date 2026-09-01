"""
DuckDB storage for ReviewEdit records (Phase 4). Uses the SAME database
file as eval_cases.duckdb (eval_case_db.DEFAULT_EVAL_DB_PATH) — review edits
are part of the same case lifecycle, not a separate concern, so they share
storage with eval_cases and dedup_rejections rather than living in a fourth
database file.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from src.eval_builder.eval_case_db import DEFAULT_EVAL_DB_PATH
from src.review.review_edit_schema import ReviewEdit

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS review_edits (
    edit_id VARCHAR PRIMARY KEY,
    case_id VARCHAR,
    field_changed VARCHAR,
    before_value VARCHAR,
    after_value VARCHAR,
    reason VARCHAR,
    reviewer_id VARCHAR,
    timestamp TIMESTAMP
);
"""


def init_review_db(db_path: str = DEFAULT_EVAL_DB_PATH) -> duckdb.DuckDBPyConnection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute(SCHEMA_SQL)
    return con


def save_review_edit(edit: ReviewEdit, db_path: str = DEFAULT_EVAL_DB_PATH) -> None:
    con = init_review_db(db_path)
    con.execute(
        "INSERT OR REPLACE INTO review_edits VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [edit.edit_id, edit.case_id, edit.field_changed, edit.before_value,
         edit.after_value, edit.reason, edit.reviewer_id, edit.timestamp],
    )
    con.close()


def save_review_edits(edits: list[ReviewEdit], db_path: str = DEFAULT_EVAL_DB_PATH) -> int:
    if not edits:
        return 0
    con = init_review_db(db_path)
    con.executemany(
        "INSERT OR REPLACE INTO review_edits VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [[e.edit_id, e.case_id, e.field_changed, e.before_value,
          e.after_value, e.reason, e.reviewer_id, e.timestamp] for e in edits],
    )
    con.close()
    return len(edits)


def _row_to_edit(row: tuple, columns: list[str]) -> ReviewEdit:
    d = dict(zip(columns, row))
    return ReviewEdit(**d)


def load_edits_for_case(case_id: str, db_path: str = DEFAULT_EVAL_DB_PATH) -> list[ReviewEdit]:
    if not Path(db_path).exists():
        return []
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            "SELECT * FROM review_edits WHERE case_id = ? ORDER BY timestamp", [case_id]
        ).fetchall()
        columns = [d[0] for d in con.description]
    except duckdb.CatalogException:
        return []  # review_edits table doesn't exist yet — no edits have ever been saved
    finally:
        con.close()
    return [_row_to_edit(r, columns) for r in rows]


def load_all_review_edits(db_path: str = DEFAULT_EVAL_DB_PATH) -> list[ReviewEdit]:
    if not Path(db_path).exists():
        return []
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute("SELECT * FROM review_edits ORDER BY timestamp").fetchall()
        columns = [d[0] for d in con.description]
    except duckdb.CatalogException:
        return []
    finally:
        con.close()
    return [_row_to_edit(r, columns) for r in rows]


def edits_by_field_summary(db_path: str = DEFAULT_EVAL_DB_PATH) -> dict[str, int]:
    """Answers the build plan's explicit question: which field gets
    corrected most often across all reviews? Directly usable as a signal for
    which label_generator.py prompt needs improvement."""
    if not Path(db_path).exists():
        return {}
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            "SELECT field_changed, COUNT(*) FROM review_edits GROUP BY field_changed ORDER BY 2 DESC"
        ).fetchall()
    except duckdb.CatalogException:
        return {}
    finally:
        con.close()
    return dict(rows)
