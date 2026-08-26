"""
Embedding wrapper (Phase 2) — sentence-transformers, local, free, no API quota.

Uses all-MiniLM-L6-v2: small (~80MB), fast on CPU, good enough for clustering
short prompts. Not meant to be state-of-the-art semantic search quality —
just enough signal for HDBSCAN to find real structure in the prompt corpus.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model_cache: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(_MODEL_NAME)
    return _model_cache


def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns an (n_texts, embedding_dim) float32 array, L2-normalized so
    cosine similarity reduces to a dot product downstream (used by both
    clustering and the diversity sampler)."""
    model = get_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)
