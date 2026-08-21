"""
DuckDB storage layer for LogEntry records.

Kept as a thin adapter (init_db / insert_logs / query helpers) so swapping
to Postgres later (mentioned as a documented tradeoff in the build plan) only
requires changing this file, not any pipeline code that calls it.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from src.logs.schema import LogEntry

DEFAULT_DB_PATH = "data/raw_logs.duckdb"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS logs (
    log_id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP,
    feature_name VARCHAR,
    model VARCHAR,
    system_prompt VARCHAR,
    prompt VARCHAR,
    response VARCHAR,
    latency_ms INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    user_feedback VARCHAR,
    retry_count INTEGER,
    error_status VARCHAR,
    redacted BOOLEAN,
    redaction_method VARCHAR,
    is_safety_edge_case BOOLEAN
);
"""


def init_db(db_path: str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute(SCHEMA_SQL)
    return con


def insert_logs(logs: list[LogEntry], db_path: str = DEFAULT_DB_PATH) -> int:
    con = init_db(db_path)
    rows = [
        (
            l.log_id, l.timestamp, l.feature_name, l.model, l.system_prompt,
            l.prompt, l.response, l.latency_ms, l.prompt_tokens, l.completion_tokens,
            l.user_feedback, l.retry_count, l.error_status, l.redacted,
            l.redaction_method, l.is_safety_edge_case,
        )
        for l in logs
    ]
    con.executemany(
        """INSERT OR REPLACE INTO logs VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    con.close()
    return len(rows)


def load_jsonl_into_db(jsonl_path: str, db_path: str = DEFAULT_DB_PATH) -> int:
    logs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            logs.append(LogEntry(**json.loads(line)))
    return insert_logs(logs, db_path)


def summary_stats(db_path: str = DEFAULT_DB_PATH) -> dict:
    con = duckdb.connect(db_path, read_only=True)
    total = con.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
    by_feature = con.execute(
        "SELECT feature_name, COUNT(*) FROM logs GROUP BY feature_name ORDER BY 2 DESC"
    ).fetchall()
    by_feedback = con.execute(
        "SELECT user_feedback, COUNT(*) FROM logs GROUP BY user_feedback"
    ).fetchall()
    error_rate = con.execute(
        "SELECT COUNT(*) FROM logs WHERE error_status IS NOT NULL"
    ).fetchone()[0]
    redacted_count = con.execute("SELECT COUNT(*) FROM logs WHERE redacted").fetchone()[0]
    safety_count = con.execute(
        "SELECT COUNT(*) FROM logs WHERE is_safety_edge_case"
    ).fetchone()[0]
    retry_count = con.execute("SELECT COUNT(*) FROM logs WHERE retry_count > 0").fetchone()[0]
    con.close()
    return {
        "total_logs": total,
        "by_feature": dict(by_feature),
        "by_feedback": dict(by_feedback),
        "error_rate": error_rate / total if total else 0,
        "redacted_rate": redacted_count / total if total else 0,
        "safety_edge_case_rate": safety_count / total if total else 0,
        "retry_rate": retry_count / total if total else 0,
    }
