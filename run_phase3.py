"""
Phase 3 orchestration script — turns Phase 2's top candidates into actual
EvalCase records:

    load top candidates (phase2_results.json) -> load full logs from DuckDB
    -> for each candidate: decide eval_type + difficulty -> generate label
    (multi-pass if hard) -> judge confidence -> dedup check -> save (approved
    if confidence >= threshold, else draft for human review in Phase 4)

Usage:
    python scripts/run_phase3.py
    python scripts/run_phase3.py --confidence-threshold 0.8 --limit 20

Note on quota: each candidate costs 1 label-generation call (3 if hard
difficulty) + 1 judge call. For the default top-30 candidates from Phase 2,
budget roughly 30-60 calls depending on how many are hard difficulty —
modest compared to the log generator's cost, but not free. --limit lets you
do a small test run first.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clustering.embed import embed_texts  # noqa: E402
from src.eval_builder.dedup import DEFAULT_SIMILARITY_THRESHOLD, check_duplicate  # noqa: E402
from src.eval_builder.eval_case_db import (  # noqa: E402
    load_all_eval_cases,
    log_dedup_rejection,
    save_eval_case,
)
from src.eval_builder.eval_case_schema import EvalCase, EvalCaseInput, transition_status  # noqa: E402
from src.eval_builder.judge import judge_label  # noqa: E402
from src.eval_builder.label_generator import generate_label  # noqa: E402
from src.eval_builder.type_decision import decide_difficulty, decide_eval_type  # noqa: E402
from src.utils.db import load_all_logs  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-results", default="data/phase2_results.json")
    parser.add_argument("--logs-db", default="data/raw_logs.duckdb")
    parser.add_argument("--eval-db", default="data/eval_cases.duckdb")
    parser.add_argument("--confidence-threshold", type=float, default=0.75,
                         help="cases scoring >= this are auto-approved; below go to draft for human review")
    parser.add_argument("--dedup-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N candidates (for testing)")
    args = parser.parse_args()

    print(f"[1/5] Loading top candidates from {args.phase2_results} ...")
    with open(args.phase2_results, "r", encoding="utf-8") as f:
        phase2 = json.load(f)
    candidates = phase2["top_candidates"]
    if args.limit:
        candidates = candidates[: args.limit]
    print(f"      {len(candidates)} candidates to process")

    print(f"[2/5] Loading full log records from {args.logs_db} ...")
    all_logs = load_all_logs(args.logs_db)
    logs_by_id = {l.log_id: l for l in all_logs}

    print(f"[3/5] Loading existing eval cases from {args.eval_db} (for dedup) ...")
    existing_cases = load_all_eval_cases(args.eval_db)
    existing_embeddings = {}
    if existing_cases:
        prompts = [c.input.prompt for c in existing_cases]
        embs = embed_texts(prompts)
        existing_embeddings = {c.case_id: embs[i] for i, c in enumerate(existing_cases)}
    print(f"      {len(existing_cases)} existing cases loaded")

    print(f"[4/5] Generating eval cases (confidence threshold={args.confidence_threshold}, "
          f"dedup threshold={args.dedup_threshold}) ...")

    n_created, n_approved, n_draft, n_deduped, n_failed = 0, 0, 0, 0, 0

    for i, cand in enumerate(candidates):
        log_id = cand["log_id"]
        log = logs_by_id.get(log_id)
        if log is None:
            print(f"  [{i+1}/{len(candidates)}] ⚠ log {log_id} not found in DB, skipping")
            n_failed += 1
            continue

        eval_type = decide_eval_type(log)
        difficulty = decide_difficulty(log)

        # dedup check happens BEFORE spending Groq calls on labeling —
        # cheaper to reject early than to generate a label for a duplicate
        cand_embedding = embed_texts([log.prompt])[0]
        dedup_result = check_duplicate(cand_embedding, existing_embeddings, threshold=args.dedup_threshold)
        if dedup_result.is_duplicate:
            log_dedup_rejection(
                rejection_id=str(uuid.uuid4()), source_log_id=log_id,
                rejected_at=datetime.now(timezone.utc),
                most_similar_case_id=dedup_result.most_similar_case_id,
                similarity=dedup_result.similarity, reason=dedup_result.reason,
                db_path=args.eval_db,
            )
            print(f"  [{i+1}/{len(candidates)}] {log_id}: DEDUPED ({dedup_result.reason})")
            n_deduped += 1
            continue

        try:
            label = generate_label(log, eval_type, difficulty)
            judge_result = judge_label(label)
        except Exception as e:
            print(f"  [{i+1}/{len(candidates)}] {log_id}: ⚠ generation/judging failed: {e}", file=sys.stderr)
            n_failed += 1
            continue

        case = EvalCase(
            source_log_id=log_id,
            source_cluster_id=cand["cluster_id"],
            input=EvalCaseInput(system_prompt=log.system_prompt, prompt=log.prompt),
            eval_type=eval_type,
            expected_behavior=label.expected_behavior,
            rubric=label.rubric,
            forbidden_assertions=label.forbidden_assertions,
            difficulty=difficulty,
            tags=label.tags,
            confidence_score=judge_result.confidence,
            label_source="auto",
        )

        if judge_result.confidence >= args.confidence_threshold:
            case = transition_status(case, "approved", reviewer_id="auto")
            n_approved += 1
            status_note = "AUTO-APPROVED"
        else:
            n_draft += 1
            status_note = "DRAFT (needs human review)"

        save_eval_case(case, db_path=args.eval_db)
        existing_embeddings[case.case_id] = cand_embedding  # so later candidates in this same run dedup against it too
        n_created += 1

        print(f"  [{i+1}/{len(candidates)}] {log_id}: {eval_type}/{difficulty}, "
              f"confidence={judge_result.confidence:.2f} -> {status_note}")

    print(f"\n[5/5] Done.")
    print(f"Summary: {n_created} cases created ({n_approved} auto-approved, {n_draft} draft), "
          f"{n_deduped} deduped, {n_failed} failed")
    print(f"Saved to {args.eval_db}")


if __name__ == "__main__":
    main()
