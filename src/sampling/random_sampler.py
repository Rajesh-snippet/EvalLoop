"""
Random sampler (Phase 2) — uniform sampling, the baseline control group.

Every other sampler in this package is judged against what random sampling
would have picked, so this is deliberately the simplest file in /sampling.
"""

from __future__ import annotations

import random

from src.logs.schema import LogEntry


def sample_random(logs: list[LogEntry], n: int, seed: int | None = None) -> list[LogEntry]:
    """Uniform random sample of n logs, without replacement.

    Raises ValueError rather than silently truncating if n > len(logs) —
    callers should notice and decide (e.g. reduce n, or just take all logs)
    rather than getting a smaller-than-requested sample without knowing why.
    """
    if n > len(logs):
        raise ValueError(f"Requested sample of {n} but only {len(logs)} logs available")
    rng = random.Random(seed)
    return rng.sample(logs, n)
