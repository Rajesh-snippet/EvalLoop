"""
Cluster labeler (Phase 2.2) — gives each HDBSCAN cluster a human-readable
name and short description, using a small number of representative examples
(from cluster.representative_indices) rather than dumping the whole cluster
into the prompt.

Token footprint is deliberately tiny: one short call per cluster, ~5 examples
each capped at 150 chars. For a corpus this size (a handful to a few dozen
clusters), this is a trivial fraction of daily quota compared to the log
generator's cost.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import dataclass  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from groq import Groq  # noqa: E402

from src.logs.schema import LogEntry  # noqa: E402
from src.utils.constants import GROQ_GENERATION_MODEL, REASONING_MODELS  # noqa: E402
from src.utils.retry import retry_with_backoff  # noqa: E402

load_dotenv()

LABELING_SYSTEM_PROMPT = (
    "You are analyzing a cluster of similar user prompts sent to various AI "
    "assistants in production. Given a few representative examples from one "
    "cluster, give it a short, specific, human-readable label and a one-"
    "sentence description. Respond ONLY with JSON: "
    '{"label": "...", "description": "..."}. '
    "The label should be 2-6 words, specific enough to distinguish this "
    "cluster from other clusters (e.g. 'PTO and sick leave questions', not "
    "just 'HR questions')."
)


@dataclass
class ClusterLabel:
    cluster_id: int
    label: str
    description: str
    size: int
    example_prompts: list[str]


def _client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set (checked environment and .env file)")
    return Groq(api_key=api_key)


@retry_with_backoff(max_retries=3, base_delay=1.0)
def _label_one_cluster(client: Groq, examples: list[str]) -> dict:
    numbered = "\n".join(f"{i+1}. {ex[:150]}" for i, ex in enumerate(examples))
    user_msg = f"Representative prompts from this cluster:\n\n{numbered}"

    resp = client.chat.completions.create(
        model=GROQ_GENERATION_MODEL,
        messages=[
            {"role": "system", "content": LABELING_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=400,
        **({"reasoning_effort": "low"} if GROQ_GENERATION_MODEL in REASONING_MODELS else {}),
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Groq returned empty content while labeling a cluster")

    # Models sometimes wrap JSON in ```json fences despite instructions —
    # strip those defensively rather than letting json.loads choke on them.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    parsed = json.loads(text)  # let this raise -> triggers retry if the model
                                # returned malformed JSON; caught by caller's
                                # own try/except if it never recovers
    if "label" not in parsed or "description" not in parsed:
        raise RuntimeError(f"Cluster label response missing required keys: {parsed}")
    return parsed


def label_clusters(
    logs: list[LogEntry],
    cluster_labels_arr,  # np.ndarray, shape (n,) — avoiding a hard numpy
                          # import here isn't worth it, but kept untyped in
                          # the signature to not force a numpy import at
                          # module load time for callers that don't need it
    representative_fn,   # callable: (cluster_id, k) -> list[int] of indices
                          # into `logs` — pass cluster.representative_indices
                          # bound with embeddings/cluster_result already applied
    k_examples: int = 5,
) -> dict[int, ClusterLabel]:
    """Labels every non-noise cluster found in cluster_labels_arr.
    Cluster -1 (HDBSCAN's noise/outlier bucket) is skipped — those points are
    individually unusual, not a coherent group to name."""
    client = _client()
    unique_ids = sorted(set(int(c) for c in cluster_labels_arr) - {-1})

    results: dict[int, ClusterLabel] = {}
    for cid in unique_ids:
        idx = representative_fn(cid, k_examples)
        if not idx:
            continue
        examples = [logs[i].prompt for i in idx]
        size = int((cluster_labels_arr == cid).sum())

        try:
            parsed = _label_one_cluster(client, examples)
        except Exception as e:
            print(f"[warn] failed to label cluster {cid} after retries: {e}", file=sys.stderr)
            parsed = {"label": f"cluster_{cid}_unlabeled", "description": "(labeling failed)"}

        results[cid] = ClusterLabel(
            cluster_id=cid,
            label=parsed["label"],
            description=parsed["description"],
            size=size,
            example_prompts=examples,
        )

    return results
