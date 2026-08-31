"""
Phase 2 orchestration script — runs the full pipeline against your real
logs in data/raw_logs.duckdb:

    load logs -> embed prompts -> cluster -> label clusters (Groq) ->
    score candidates -> save results + print a summary

Usage:
    python scripts/run_phase2.py
    python scripts/run_phase2.py --min-cluster-size 6 --top-n 30

Output: data/phase2_results.json — cluster labels/sizes, top-N scored
candidates overall and per-cluster, plus example outputs from all 3
samplers, so you (and later, Phase 3) have something concrete to look at.

Cluster labeling calls Groq (~1 short call per cluster). If GROQ_API_KEY
isn't set or labeling fails, clustering/scoring still complete — clusters
just get a placeholder name instead of crashing the whole run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clustering.cluster import representative_indices, run_clustering  # noqa: E402
from src.clustering.embed import embed_texts  # noqa: E402
from src.eval_builder.eval_case_db import load_eval_cases_by_status  # noqa: E402
from src.sampling.diversity_sampler import sample_diverse  # noqa: E402
from src.sampling.failure_biased_sampler import sample_failure_biased  # noqa: E402
from src.sampling.random_sampler import sample_random  # noqa: E402
from src.scoring.candidate_scorer import score_candidates  # noqa: E402
from src.utils.db import load_all_logs  # noqa: E402

import numpy as np  # noqa: E402


def compute_cluster_approved_counts(
    embeddings: np.ndarray,
    cluster_result,
    eval_db_path: str,
) -> dict[int, int]:
    """Closes the flywheel loop described in the build plan: count how many
    APPROVED eval cases already exist per cluster, so score_candidates can
    push already-covered clusters down and under-covered ones up.

    IMPORTANT: HDBSCAN cluster IDs are arbitrary per-run, not stable labels —
    a cluster numbered 4 in an old run has no guaranteed relationship to
    cluster 4 in a fresh clustering, even over the same underlying logs
    (different log counts, different min_cluster_size, or just re-running
    can reshuffle IDs). Trusting a stored source_cluster_id from an old
    EvalCase against a NEW cluster_result would silently misattribute
    coverage to the wrong cluster. Instead, every approved case's prompt is
    re-embedded and assigned to whichever CURRENT cluster centroid it's
    nearest to — correct regardless of how cluster numbering shifted between
    runs, at the cost of one extra embedding pass over the approved cases."""
    approved = load_eval_cases_by_status("approved", db_path=eval_db_path)
    if not approved:
        return {}

    real_cluster_ids = sorted(set(cluster_result.labels.tolist()) - {-1})
    if not real_cluster_ids:
        return {}

    centroids = {
        cid: embeddings[np.where(cluster_result.labels == cid)[0]].mean(axis=0)
        for cid in real_cluster_ids
    }
    centroid_ids = list(centroids.keys())
    centroid_arr = np.stack([centroids[c] for c in centroid_ids])

    case_embeddings = embed_texts([c.input.prompt for c in approved])

    counts = {cid: 0 for cid in real_cluster_ids}
    for emb in case_embeddings:
        dists = np.linalg.norm(centroid_arr - emb, axis=1)
        nearest = centroid_ids[int(np.argmin(dists))]
        counts[nearest] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/raw_logs.duckdb")
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=30, help="top-N candidates to save/print")
    parser.add_argument("--sampler-preview-n", type=int, default=10)
    parser.add_argument("--out", default="data/phase2_results.json")
    parser.add_argument("--eval-db", default="data/eval_cases.duckdb",
                         help="used to read back APPROVED case counts per cluster for coverage-gap scoring")
    parser.add_argument("--skip-labeling", action="store_true",
                         help="skip Groq cluster labeling (e.g. to conserve quota during testing)")
    args = parser.parse_args()

    print(f"[1/6] Loading logs from {args.db} ...")
    logs = load_all_logs(args.db)
    print(f"      {len(logs)} logs loaded")
    if len(logs) < 20:
        print("      ⚠ Very few logs — clustering/sampling results may be degenerate.", file=sys.stderr)

    print(f"[2/6] Embedding {len(logs)} prompts (all-MiniLM-L6-v2, local, first run downloads ~80MB) ...")
    embeddings = embed_texts([l.prompt for l in logs])
    print(f"      embeddings shape: {embeddings.shape}")

    print(f"[3/6] Clustering (min_cluster_size={args.min_cluster_size}) ...")
    cluster_result = run_clustering(embeddings, min_cluster_size=args.min_cluster_size)
    print(f"      {cluster_result.n_clusters} clusters, {cluster_result.n_noise} noise points")

    cluster_info = {}
    if args.skip_labeling:
        print("[4/6] Skipping cluster labeling (--skip-labeling set)")
        for cid in sorted(set(cluster_result.labels.tolist()) - {-1}):
            size = int((cluster_result.labels == cid).sum())
            cluster_info[cid] = {"label": f"cluster_{cid}", "description": "(labeling skipped)", "size": size}
    else:
        print(f"[4/6] Labeling clusters via Groq ({GROQ_MODEL_NOTE}) ...")
        try:
            from src.clustering.cluster_labeler import label_clusters

            def rep_fn(cid, k):
                return representative_indices(embeddings, cluster_result, cid, k)

            labels = label_clusters(logs, cluster_result.labels, rep_fn, k_examples=5)
            for cid, info in labels.items():
                cluster_info[cid] = {"label": info.label, "description": info.description, "size": info.size}
                print(f"      cluster {cid} ({info.size} logs): {info.label}")
        except Exception as e:
            print(f"      ⚠ Cluster labeling failed entirely ({e}); continuing with placeholder labels.", file=sys.stderr)
            for cid in sorted(set(cluster_result.labels.tolist()) - {-1}):
                size = int((cluster_result.labels == cid).sum())
                cluster_info[cid] = {"label": f"cluster_{cid}", "description": "(labeling failed)", "size": size}

    print(f"[5/6] Scoring candidates ...")
    cluster_approved_counts = compute_cluster_approved_counts(embeddings, cluster_result, args.eval_db)
    if cluster_approved_counts:
        total_mapped = sum(cluster_approved_counts.values())
        print(f"      mapped {total_mapped} previously-approved eval case(s) onto current clusters for coverage-gap scoring")
        for cid, count in sorted(cluster_approved_counts.items()):
            if count > 0:
                print(f"        cluster {cid}: {count} already approved")
    else:
        print(f"      no approved eval cases found yet (or {args.eval_db} doesn't exist) — coverage-gap has no effect this run")
    scored = score_candidates(logs, embeddings, cluster_result.labels, cluster_approved_counts=cluster_approved_counts)
    top_candidates = [
        {
            "log_id": s.log.log_id,
            "feature_name": s.log.feature_name,
            "prompt_preview": s.log.prompt[:150],
            "cluster_id": s.cluster_id,
            "cluster_label": cluster_info.get(s.cluster_id, {}).get("label", "noise"),
            "value_score": round(s.value_score, 3),
            "failure_signal": round(s.failure_signal, 3),
            "novelty": round(s.novelty, 3),
        }
        for s in scored[: args.top_n]
    ]
    print(f"      top {args.top_n} candidates selected (scores {scored[0].value_score:.2f} down to {scored[args.top_n-1].value_score:.2f})")

    print(f"[6/6] Sampler previews (n={args.sampler_preview_n} each) ...")
    seed = 42
    previews = {
        "random": [l.log_id for l in sample_random(logs, min(args.sampler_preview_n, len(logs)), seed=seed)],
        "failure_biased": [l.log_id for l in sample_failure_biased(logs, min(args.sampler_preview_n, len(logs)), seed=seed)],
        "diversity": [l.log_id for l in sample_diverse(logs, embeddings, min(args.sampler_preview_n, len(logs)), seed=seed)],
    }

    results = {
        "n_logs": len(logs),
        "n_clusters": cluster_result.n_clusters,
        "n_noise": cluster_result.n_noise,
        "clusters": {str(k): v for k, v in cluster_info.items()},
        "top_candidates": top_candidates,
        "sampler_previews": previews,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Results saved to {args.out}")
    print(f"Summary: {len(logs)} logs -> {cluster_result.n_clusters} clusters ({cluster_result.n_noise} noise) -> top {args.top_n} candidates ranked")


GROQ_MODEL_NOTE = "~1 short call per cluster, small quota footprint"

if __name__ == "__main__":
    main()
