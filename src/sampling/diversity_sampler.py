"""
Diversity sampler (Phase 2).

Picks n logs that maximize pairwise spread in embedding space, using greedy
farthest-point sampling: start from one point, repeatedly add whichever
remaining point is farthest (by minimum distance) from everything already
picked. This is the standard greedy approximation to the (NP-hard) exact
max-min-distance selection problem — good enough here, no need for exact.

Why this matters for the eval set: without it, sampling could easily pick
30 near-duplicate "reset my password" logs and call it diverse. Farthest-
point selection actively spreads picks across the embedding space, so the
sample covers more of the actual variety in the traffic.
"""

from __future__ import annotations

import numpy as np

from src.logs.schema import LogEntry


def sample_diverse(
    logs: list[LogEntry],
    embeddings: np.ndarray,
    n: int,
    seed: int | None = None,
) -> list[LogEntry]:
    """Greedy farthest-point sampling over embeddings (assumed L2-normalized,
    so squared-euclidean distance is used as a cheap proxy for cosine
    distance — monotonic for normalized vectors, avoids sqrt in the hot loop).

    logs and embeddings must be the same length and in the same order.
    """
    if len(logs) != len(embeddings):
        raise ValueError(f"logs ({len(logs)}) and embeddings ({len(embeddings)}) length mismatch")
    if n > len(logs):
        raise ValueError(f"Requested sample of {n} but only {len(logs)} logs available")
    if n <= 0:
        return []

    rng = np.random.default_rng(seed)
    n_points = len(logs)

    selected_idx = [int(rng.integers(0, n_points))]
    # min_dist_sq[i] = squared distance from point i to its NEAREST selected point so far
    min_dist_sq = np.sum((embeddings - embeddings[selected_idx[0]]) ** 2, axis=1)

    while len(selected_idx) < n:
        next_idx = int(np.argmax(min_dist_sq))
        selected_idx.append(next_idx)
        new_dist_sq = np.sum((embeddings - embeddings[next_idx]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, new_dist_sq)
        # already-selected points get distance -inf so they're never re-picked
        min_dist_sq[next_idx] = -np.inf

    return [logs[i] for i in selected_idx]
