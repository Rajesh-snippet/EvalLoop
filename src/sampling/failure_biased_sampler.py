"""
Failure-biased sampler (Phase 2).

Design principle (the interview talking point from the build plan): a good
eval set should NOT mirror production traffic distribution. It should
over-represent risk, failures, and edge cases, because that's where
regressions actually hurt users. This sampler is the mechanism for that —
it weights sampling probability toward logs with negative feedback, retries,
errors, and safety edge cases, so those over-represented categories get
picked far more often than their raw frequency in the logs would suggest.

Weight formula (documented here since it's the one part of this file most
worth defending in an interview — the exact constants are a starting point,
tuned to be directionally sensible, not derived from any ground truth):

    weight = 1
            + 3 * (user_feedback == 'negative')
            + 2 * min(retry_count, 3)
            + 4 * (error_status is not None)
            + 5 * is_safety_edge_case

A "perfect" log (positive/none feedback, no retries, no error, not a safety
case) gets weight 1 — it can still be sampled, just rarely, so the eval set
isn't ONLY failures (a dataset of pure failures would miss regressions in
things that currently work). Safety edge cases get the heaviest single boost
(+5) since those are the highest-cost failure mode to miss in production.
"""

from __future__ import annotations

import random

from src.logs.schema import LogEntry


def failure_weight(log: LogEntry) -> float:
    weight = 1.0
    if log.user_feedback == "negative":
        weight += 3.0
    weight += 2.0 * min(log.retry_count, 3)
    if log.error_status is not None:
        weight += 4.0
    if log.is_safety_edge_case:
        weight += 5.0
    return weight


def sample_failure_biased(logs: list[LogEntry], n: int, seed: int | None = None) -> list[LogEntry]:
    """Weighted sample without replacement, biased toward failure signals.

    Uses the "weighted random sample without replacement via exponential
    keys" approach (Efraimidis-Spirakis): each log gets a key = U^(1/weight)
    for U ~ Uniform(0,1), and we take the top-n by key. This gives a proper
    weighted sample without replacement in one pass, rather than the more
    common (and subtly biased) approach of repeatedly sampling-with-removal.
    """
    if n > len(logs):
        raise ValueError(f"Requested sample of {n} but only {len(logs)} logs available")
    rng = random.Random(seed)

    keyed = []
    for log in logs:
        w = failure_weight(log)
        u = rng.random()
        # guard against u=0 (measure zero, but be safe) and w<=0 (shouldn't
        # happen given weight starts at 1.0, but fail loudly if it does)
        if w <= 0:
            raise ValueError(f"Non-positive failure weight for log {log.log_id}: {w}")
        u = max(u, 1e-12)
        key = u ** (1.0 / w)
        keyed.append((key, log))

    keyed.sort(key=lambda pair: pair[0], reverse=True)
    return [log for _, log in keyed[:n]]
