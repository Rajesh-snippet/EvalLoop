"""
Candidate value scorer (Phase 2.3) — combines everything upstream into one
ranked score per log, used to decide which logs are worth turning into eval
cases in Phase 3. Not every sampled log becomes an eval case; this score is
what prioritizes among them.

    value_score = w1 * failure_signal
                + w2 * novelty
                + w3 * feature_impact_weight
                + w4 * cluster_coverage_gap

- failure_signal: reuses the failure_weight() formula from the failure-biased
  sampler, normalized to [0, 1] so it's comparable in scale to the other terms.
- novelty: how far this log's prompt is from its own cluster's centroid
  (HDBSCAN's outlier_scores_, or distance-based fallback) — unusual prompts
  within a category are more likely to surface real edge-case behavior.
- feature_impact_weight: a per-feature multiplier reflecting how much a
  regression in that feature would matter. Defaults to 1.0 for every feature
  (no built-in opinion about which of your 6 synthetic features is "more
  important" — that's a business judgment, not something to infer from the
  data). Override via the feature_weights argument when you have that
  context (e.g. a payments feature should outweigh a meal-planner feature).
- cluster_coverage_gap: inverse of how many APPROVED eval cases already exist
  for this log's cluster. This is what closes the flywheel loop described in
  the build plan — clusters that already have good eval coverage stop
  attracting new candidates, clusters with none get prioritized. Requires
  passing in existing per-cluster approved counts; defaults to "no coverage
  yet" (gap=1.0 for every cluster) until Phase 3+ produces real counts.

All four weights (w1..w4) default to 1.0 (equal-weighted). These are
explicitly tunable, not fixed — documented as such because "we tuned these
empirically" is a legitimate, expected answer to give in an interview, not
a weakness to hide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.logs.schema import LogEntry
from src.sampling.failure_biased_sampler import failure_weight

MAX_POSSIBLE_FAILURE_WEIGHT = 1.0 + 3.0 + 2.0 * 3 + 4.0 + 5.0  # = 19.0, see failure_weight()


@dataclass
class ScoringWeights:
    failure: float = 1.0
    novelty: float = 1.0
    feature_impact: float = 1.0
    coverage_gap: float = 1.0


@dataclass
class ScoredCandidate:
    log: LogEntry
    cluster_id: int
    failure_signal: float
    novelty: float
    feature_impact_weight: float
    coverage_gap: float
    value_score: float


def compute_novelty(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
) -> np.ndarray:
    """Per-point novelty in [0, 1]: normalized distance from own cluster's
    centroid. Noise points (label -1) are scored by distance to the NEAREST
    cluster centroid instead of a flat constant — a noise point sitting just
    outside a dense cluster and a noise point sitting far from everything are
    both "unfit" by HDBSCAN's criteria, but the second is more genuinely
    novel, and previously both got an identical score (1.0), which meant
    noise-point rankings were arbitrary ties broken only by input order.
    Noise novelty is scaled into [0.9, 1.0] so noise points still generally
    outrank in-cluster points (preserving the original "noise is maximally
    novel" intent) while remaining differentiable from each other. If every
    point is noise (no real clusters found at all), there's nothing to
    measure distance to, so all points fall back to a flat 1.0."""
    n = len(embeddings)
    novelty = np.zeros(n, dtype=np.float32)

    real_cluster_ids = sorted(set(cluster_labels.tolist()) - {-1})
    centroids = {
        cid: embeddings[np.where(cluster_labels == cid)[0]].mean(axis=0)
        for cid in real_cluster_ids
    }

    for cid in set(cluster_labels.tolist()):
        idx = np.where(cluster_labels == cid)[0]
        if cid == -1:
            if not centroids:
                novelty[idx] = 1.0
                continue
            centroid_arr = np.stack(list(centroids.values()))  # (n_clusters, dim)
            # distance from each noise point to every cluster centroid, take the min
            dists_to_each = np.linalg.norm(
                embeddings[idx][:, None, :] - centroid_arr[None, :, :], axis=2
            )
            nearest_dist = dists_to_each.min(axis=1)
            max_d = nearest_dist.max()
            scaled = nearest_dist / max_d if max_d > 0 else np.zeros_like(nearest_dist)
            novelty[idx] = 0.9 + 0.1 * scaled
            continue
        if len(idx) == 1:
            novelty[idx] = 1.0
            continue
        centroid = centroids[cid]
        dists = np.linalg.norm(embeddings[idx] - centroid, axis=1)
        max_d = dists.max()
        novelty[idx] = dists / max_d if max_d > 0 else 0.0

    return novelty


def score_candidates(
    logs: list[LogEntry],
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    feature_weights: dict[str, float] | None = None,
    cluster_approved_counts: dict[int, int] | None = None,
    weights: ScoringWeights | None = None,
) -> list[ScoredCandidate]:
    """logs, embeddings, cluster_labels must all be the same length/order
    (i.e. the full corpus, not a pre-filtered sample — scoring needs the
    full cluster context to compute novelty and coverage gap correctly)."""
    if not (len(logs) == len(embeddings) == len(cluster_labels)):
        raise ValueError(
            f"length mismatch: logs={len(logs)}, embeddings={len(embeddings)}, "
            f"cluster_labels={len(cluster_labels)}"
        )
    weights = weights or ScoringWeights()
    feature_weights = feature_weights or {}
    cluster_approved_counts = cluster_approved_counts or {}

    novelty_arr = compute_novelty(embeddings, cluster_labels)

    max_approved = max(cluster_approved_counts.values(), default=0)

    results = []
    for i, log in enumerate(logs):
        cid = int(cluster_labels[i])

        fw = failure_weight(log) / MAX_POSSIBLE_FAILURE_WEIGHT
        nov = float(novelty_arr[i])
        feat_w = feature_weights.get(log.feature_name, 1.0)

        approved_here = cluster_approved_counts.get(cid, 0)
        # gap=1.0 if this cluster has zero approved cases yet, shrinking
        # toward 0 as it accumulates coverage relative to the best-covered
        # cluster seen so far. If nothing has been approved anywhere yet,
        # every cluster gets gap=1.0 (no coverage info to differentiate on).
        coverage_gap = 1.0 - (approved_here / max_approved) if max_approved > 0 else 1.0

        value = (
            weights.failure * fw
            + weights.novelty * nov
            + weights.feature_impact * feat_w
            + weights.coverage_gap * coverage_gap
        )

        results.append(
            ScoredCandidate(
                log=log,
                cluster_id=cid,
                failure_signal=fw,
                novelty=nov,
                feature_impact_weight=feat_w,
                coverage_gap=coverage_gap,
                value_score=value,
            )
        )

    results.sort(key=lambda c: c.value_score, reverse=True)
    return results
