"""
Clustering (Phase 2) — HDBSCAN over prompt embeddings.

Deliberately takes a plain (n, dim) numpy array rather than calling embed.py
internally — keeps this module testable with synthetic embeddings (no model
download needed) and reusable if the embedding backend ever changes.

HDBSCAN over KMeans because:
  - Don't need to guess cluster count in advance
  - Naturally labels outliers as noise (-1) instead of forcing every point
    into a cluster — those outliers are exactly the "unusual prompt" signal
    the candidate scorer (Phase 2.3) wants to weight up.
  - Handles clusters of very different densities/sizes, which is realistic
    for a corpus that mixes six very different feature domains.
"""

from __future__ import annotations

from dataclasses import dataclass

import hdbscan
import numpy as np


@dataclass
class ClusterResult:
    labels: np.ndarray          # shape (n,), -1 = noise/outlier
    probabilities: np.ndarray   # shape (n,), HDBSCAN's confidence per point
    n_clusters: int
    n_noise: int


def run_clustering(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
) -> ClusterResult:
    """Runs HDBSCAN over pre-computed embeddings.

    min_cluster_size=5 is a starting point for a few-hundred-log corpus —
    small enough to find real sub-categories, large enough to not just
    fragment into singleton "clusters". Worth revisiting once you see actual
    cluster counts on your real data; if everything collapses into one giant
    cluster or fragments into 50 tiny ones, this is the knob to turn.
    """
    if len(embeddings) < min_cluster_size:
        raise ValueError(
            f"Only {len(embeddings)} embeddings but min_cluster_size={min_cluster_size} "
            "— reduce min_cluster_size or provide more data."
        )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",  # embeddings are L2-normalized, so euclidean
                              # distance is monotonic with cosine distance
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings)
    probabilities = clusterer.probabilities_

    unique_labels = set(labels.tolist())
    n_noise = int((labels == -1).sum())
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    return ClusterResult(
        labels=labels,
        probabilities=probabilities,
        n_clusters=n_clusters,
        n_noise=n_noise,
    )


def representative_indices(
    embeddings: np.ndarray,
    cluster_result: ClusterResult,
    cluster_id: int,
    k: int = 5,
) -> list[int]:
    """Returns indices of the k points closest to the centroid of the given
    cluster — these are what get shown to the LLM for cluster labeling
    (Phase 2.2), since a handful of representative examples produces a much
    more accurate label than the whole cluster dumped into a prompt."""
    member_idx = np.where(cluster_result.labels == cluster_id)[0]
    if len(member_idx) == 0:
        return []
    centroid = embeddings[member_idx].mean(axis=0)
    dists = np.linalg.norm(embeddings[member_idx] - centroid, axis=1)
    order = np.argsort(dists)[:k]
    return member_idx[order].tolist()
