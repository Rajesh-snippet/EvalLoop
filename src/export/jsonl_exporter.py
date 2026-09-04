"""
EvalLoop Phase 5 — JSONL Exporter.

Exports all APPROVED eval cases into a versioned JSONL file under
data/exports/, and appends a human-readable changelog entry describing
what changed relative to the previous export (added / removed case_ids).

Usage:
    python -m src.export.jsonl_exporter
    python -m src.export.jsonl_exporter --data-dir data
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from src.eval_builder.eval_case_db import (
    DEFAULT_EVAL_DB_PATH,
    load_eval_cases_by_status,
)

DEFAULT_EXPORT_DIR = "data/exports"
CHANGELOG_FILENAME = "CHANGELOG.md"
VERSION_PATTERN = re.compile(r"eval_dataset_v(\d+)\.jsonl$")


def _next_version(export_dir: Path) -> int:
    """Find the highest existing eval_dataset_v{n}.jsonl and return n+1."""
    if not export_dir.exists():
        return 1

    versions = [
        int(match.group(1))
        for path in export_dir.glob("eval_dataset_v*.jsonl")
        if (match := VERSION_PATTERN.search(path.name))
    ]
    return max(versions, default=0) + 1


def _load_previous_case_ids(export_dir: Path, version: int) -> set[str]:
    """Read case_ids from the immediately preceding export, if any."""
    if version <= 1:
        return set()

    previous_path = export_dir / f"eval_dataset_v{version - 1}.jsonl"
    if not previous_path.exists():
        return set()

    case_ids: set[str] = set()
    with previous_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cid = record.get("case_id")
            if cid:
                case_ids.add(cid)
    return case_ids


def case_to_export_record(case) -> dict:
    """Map an EvalCase to the export JSON shape defined in the build plan."""
    return {
        "case_id": case.case_id,
        "input": {
            "system_prompt": case.input.system_prompt,
            "prompt": case.input.prompt,
        },
        "eval_type": case.eval_type,
        "expected_behavior": case.expected_behavior,
        "rubric": case.rubric,
        "forbidden_assertions": case.forbidden_assertions,
        "tags": case.tags,
        "difficulty": case.difficulty,
        "source_cluster": case.source_cluster_id,
        "date_added": (
            case.approved_at.isoformat()
            if getattr(case, "approved_at", None)
            else datetime.now(timezone.utc).isoformat()
        ),
    }


def export_approved_cases(
    eval_db_path: str = DEFAULT_EVAL_DB_PATH,
    data_dir: str = "data",
) -> dict:
    """
    Export all approved cases to a new versioned JSONL file.

    Works regardless of how many approved cases exist — zero produces an
    empty (but valid) JSONL file plus a changelog note, rather than
    raising. Returns a summary dict describing what was written.
    """
    export_dir = Path(data_dir) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    approved_cases = load_eval_cases_by_status("approved", eval_db_path)

    version = _next_version(export_dir)
    previous_ids = _load_previous_case_ids(export_dir, version)
    current_ids = {case.case_id for case in approved_cases}

    added = sorted(current_ids - previous_ids)
    removed = sorted(previous_ids - current_ids)

    export_path = export_dir / f"eval_dataset_v{version}.jsonl"
    with export_path.open("w", encoding="utf-8") as f:
        for case in approved_cases:
            f.write(json.dumps(case_to_export_record(case)) + "\n")

    _append_changelog(
        export_dir=export_dir,
        version=version,
        total=len(approved_cases),
        added=added,
        removed=removed,
    )

    return {
        "version": version,
        "path": str(export_path),
        "total_cases": len(approved_cases),
        "added": added,
        "removed": removed,
    }


def _append_changelog(
    export_dir: Path,
    version: int,
    total: int,
    added: list[str],
    removed: list[str],
) -> None:
    changelog_path = export_dir / CHANGELOG_FILENAME
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines = [
        f"## v{version} — {timestamp}",
        f"- total approved cases: {total}",
    ]

    if not added and not removed and version > 1:
        lines.append("- no change from previous version")
    else:
        if added:
            lines.append(f"- added ({len(added)}): {', '.join(added)}")
        if removed:
            lines.append(f"- removed ({len(removed)}): {', '.join(removed)}")
        if version == 1:
            lines.append("- initial export")

    entry = "\n".join(lines) + "\n\n"

    if changelog_path.exists():
        existing = changelog_path.read_text(encoding="utf-8")
    else:
        existing = "# EvalLoop Export Changelog\n\n"

    changelog_path.write_text(existing + entry, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export approved eval cases to JSONL.")
    parser.add_argument("--eval-db", default=DEFAULT_EVAL_DB_PATH)
    parser.add_argument("--data-dir", default="data")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = export_approved_cases(eval_db_path=args.eval_db, data_dir=args.data_dir)

    print(f"Exported v{result['version']} -> {result['path']}")
    print(f"  total approved cases: {result['total_cases']}")
    if result["total_cases"] == 0:
        print("  WARNING: no approved cases yet — export is empty. "
              "Review some cases in Phase 4 before running the eval runner.")
    if result["added"]:
        print(f"  added: {len(result['added'])}")
    if result["removed"]:
        print(f"  removed: {len(result['removed'])}")


if __name__ == "__main__":
    main()
