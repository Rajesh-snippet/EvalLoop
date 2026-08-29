"""
Dedup (Phase 3.3) — before inserting a new EvalCase, check its prompt
against existing cases' prompts via embedding cosine similarity. Above
threshold -> reject as duplicate, with an auditable reason logged (per the
build plan: "Track why each candidate was accepted or rejected") rather than
silently dropped.

Uses the same embedding backend as Phase 2 clustering (src.clustering.embed)
so distances are on a consistent scale with the rest of the pipeline.

DEFAULT_SIMILARITY_THRESHOLD=0.92 is a starting point, not empirically tuned
against a labeled sample of known-duplicate/non-duplicate pairs (the build
plan flags this as something worth doing before treating the threshold as
final — noted here, not yet done, since it needs real judgment calls on
actual generated pairs, not something to fabricate).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_SIMILARITY_THRESHOLD = 0.92


@dataclass
class DedupResult:
    is_duplicate: bool
    most_similar_case_id: str | None
    similarity: float | None
    reason: str


def check_duplicate(
    candidate_embedding: np.ndarray,
    existing_embeddings: dict[str, np.ndarray],  # case_id -> embedding
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> DedupResult:
    """embeddings are assumed L2-normalized (as src.clustering.embed produces),
    so cosine similarity reduces to a dot product. Returns the single most
    similar existing case regardless of outcome, so a near-miss (similar but
    below threshold) is still visible to a human reviewer as useful context —
    this is exactly the "similar existing cases" panel the build plan's
    Phase 4 review UI calls for."""
    if not existing_embeddings:
        return DedupResult(
            is_duplicate=False, most_similar_case_id=None, similarity=None,
            reason="no existing cases to compare against",
        )

    case_ids = list(existing_embeddings.keys())
    matrix = np.stack([existing_embeddings[cid] for cid in case_ids])
    similarities = matrix @ candidate_embedding  # dot product == cosine sim for normalized vectors

    best_idx = int(np.argmax(similarities))
    best_sim = float(similarities[best_idx])
    best_id = case_ids[best_idx]

    if best_sim >= threshold:
        return DedupResult(
            is_duplicate=True, most_similar_case_id=best_id, similarity=best_sim,
            reason=f"duplicate of {best_id} (similarity {best_sim:.3f} >= threshold {threshold})",
        )
    return DedupResult(
        is_duplicate=False, most_similar_case_id=best_id, similarity=best_sim,
        reason=f"not a duplicate; closest existing case is {best_id} (similarity {best_sim:.3f} < threshold {threshold})",
    )
