"""
Phase 2 verification script — run this after run_phase2.py to sanity-check
the clustering/scoring output before moving to Phase 3.

Usage:
    python scripts/verify_phase2.py --results data/phase2_results.json

Checks performed:
  1. JSON structure is valid and has the expected keys
  2. Cluster count and noise ratio are in a sane range
  3. Cluster sizes are reasonably balanced (no single cluster dominating)
  4. Every cluster has a real label (flags "_unlabeled" placeholders from
     failed Groq calls, e.g. a quota wall mid-run)
  5. Top candidates have genuinely differentiated scores (flags suspicious
     ties, e.g. the noise-novelty bug from earlier in this project)
  6. Sampler previews exist and are non-empty
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_results(path: str) -> dict:
    if not Path(path).exists():
        print(f"File not found: {path}")
        print("Run scripts/run_phase2.py first.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_structure(d: dict) -> bool:
    required = {"n_logs", "n_clusters", "n_noise", "clusters", "top_candidates", "sampler_previews"}
    missing = required - set(d.keys())
    if missing:
        print(f"[1] Structure: 🛑 FAIL — missing keys: {missing}")
        return False
    print(f"[1] Structure: ✅ all expected keys present")
    return True


def check_cluster_health(d: dict) -> None:
    n_logs = d["n_logs"]
    n_clusters = d["n_clusters"]
    n_noise = d["n_noise"]
    noise_ratio = n_noise / n_logs if n_logs else 0

    print(f"\n[2] Cluster health:")
    print(f"    {n_logs} logs -> {n_clusters} clusters, {n_noise} noise points ({noise_ratio:.1%})")

    if n_clusters == 0:
        print(f"    🛑 FAIL — zero clusters found. min_cluster_size is likely too high for this dataset size.")
    elif n_clusters == 1:
        print(f"    ⚠ only 1 cluster — everything got lumped together. Consider lowering min_cluster_size.")
    elif noise_ratio > 0.30:
        print(f"    ⚠ noise ratio is high ({noise_ratio:.1%}) — a lot of logs didn't fit any cluster.")
    else:
        print(f"    ✅ cluster count and noise ratio look reasonable.")

    sizes = {int(cid): info["size"] for cid, info in d["clusters"].items()}
    if sizes:
        total_clustered = sum(sizes.values())
        max_share = max(sizes.values()) / total_clustered
        print(f"\n    Cluster sizes:")
        for cid in sorted(sizes):
            size = sizes[cid]
            pct = size / total_clustered * 100
            bar = "█" * int(pct / 3)
            print(f"      cluster {cid:2} {size:4} logs ({pct:5.1f}%) {bar}")
        if max_share > 0.45:
            print(f"    ⚠ one cluster holds {max_share:.0%} of all clustered logs — may be a catch-all bucket worth splitting.")
        else:
            print(f"    ✅ cluster sizes are reasonably balanced.")


def check_labels(d: dict) -> None:
    print(f"\n[3] Cluster labels:")
    unlabeled = [cid for cid, info in d["clusters"].items() if "_unlabeled" in info.get("label", "")]
    failed_desc = [cid for cid, info in d["clusters"].items() if "labeling failed" in info.get("description", "")]
    total = len(d["clusters"])

    if unlabeled or failed_desc:
        bad = sorted(set(unlabeled) | set(failed_desc), key=int)
        print(f"    ⚠ {len(bad)}/{total} clusters have placeholder labels (Groq call failed, likely a quota wall): {bad}")
        print(f"    This does NOT block Phase 3 — cluster_id is what matters functionally, labels are cosmetic.")
        print(f"    Worth re-labeling later (cheap, ~1 call/cluster) before writing up results for a portfolio.")
    else:
        print(f"    ✅ all {total} clusters have real labels.")
        for cid, info in d["clusters"].items():
            print(f"      cluster {cid}: {info['label']}")


def check_candidate_scores(d: dict) -> None:
    print(f"\n[4] Top candidate scores:")
    candidates = d["top_candidates"]
    scores = [c["value_score"] for c in candidates]
    unique_scores = len(set(scores))

    print(f"    {len(candidates)} candidates, {unique_scores} unique scores")
    if unique_scores < len(candidates) * 0.7:
        print(f"    ⚠ many tied scores — check candidate_scorer.py's novelty computation for noise points.")
    else:
        print(f"    ✅ scores are well differentiated.")

    # flag if noise-sourced candidates dominate or are entirely absent from the top list —
    # either extreme suggests something worth a second look, not necessarily wrong
    noise_in_top = sum(1 for c in candidates if c["cluster_id"] == -1)
    if noise_in_top == 0:
        print(f"    note: no noise-cluster candidates in top {len(candidates)} — fine, just worth knowing.")
    elif noise_in_top == len(candidates):
        print(f"    ⚠ ALL top candidates are noise-cluster — real clusters are being scored too low relative to noise.")
    else:
        print(f"    {noise_in_top}/{len(candidates)} top candidates come from noise (unclustered) logs.")

    print(f"\n    Top 5 candidates:")
    for c in candidates[:5]:
        print(f"      score={c['value_score']:.3f} cluster={c['cluster_id']:3} "
              f"({c['cluster_label']}) feature={c['feature_name']}")
        print(f"        {c['prompt_preview'][:100]}")


def check_sampler_previews(d: dict) -> None:
    print(f"\n[5] Sampler previews:")
    previews = d.get("sampler_previews", {})
    expected = {"random", "failure_biased", "diversity"}
    missing = expected - set(previews.keys())
    if missing:
        print(f"    ⚠ missing sampler preview(s): {missing}")
        return
    for name, ids in previews.items():
        print(f"    {name:16} {len(ids)} log_ids")
    empty = [name for name, ids in previews.items() if not ids]
    if empty:
        print(f"    ⚠ empty preview list(s): {empty}")
    else:
        print(f"    ✅ all 3 sampler previews present and non-empty.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/phase2_results.json")
    args = parser.parse_args()

    print(f"Verifying: {args.results}\n{'='*60}")
    d = load_results(args.results)
    if not check_structure(d):
        sys.exit(1)

    check_cluster_health(d)
    check_labels(d)
    check_candidate_scores(d)
    check_sampler_previews(d)

    print(f"\n{'='*60}")
    print(f"Done. Review any ⚠/🛑 warnings above before moving to Phase 3.")


if __name__ == "__main__":
    main()
