# EvalLoop

**A production log-to-eval dataset builder — mining real usage data into a continuously-improving, human-reviewed LLM evaluation suite.**

EvalLoop is not a one-off test script. It's a closed feedback loop: production-like logs are mined for high-value examples, converted into evaluation cases, labeled by an LLM with a confidence score, routed to human review when uncertain, and then run against a target model on demand — with regressions tracked across runs and coverage gaps automatically steering what gets sampled next.

---

## Why this exists

Most "eval sets" for LLM products are either hand-written once and forgotten, or built by randomly sampling production traffic. Both approaches miss the point: **a good eval set should not mirror traffic — it should over-represent risk, failure, and edge cases**, because that's where regressions actually hurt users. EvalLoop operationalizes that principle end to end, rather than treating it as a slide in a deck.

---

## Architecture

```
Synthetic Production Logs
         │
         ▼
    DuckDB (raw_logs)
         │
         ▼
  Privacy / Redaction
         │
         ▼
   Sampling Engine
 ┌────────┼────────┐
 ▼        ▼        ▼
Random  Failure-  Diversity
        biased
 └────────┼────────┘
          ▼
   Embedding + Clustering
          │
          ▼
   Candidate Scoring
   (coverage-gap aware)
          │
          ▼
   Eval Case Builder
          │
          ▼
     LLM-as-Judge
    (confidence score)
      /        \
     /          \
High conf.   Low conf.
    │              │
    ▼              ▼
Auto-approved   Human Review
    │              │
    └──────┬───────┘
           ▼
   Approved Dataset
           │
           ▼
         JSONL
           │
           ▼
     Eval Runner
   (target ≠ judge model)
           │
           ▼
  Metrics / Regression Diff
           │
           ▼
    Dataset Health Dashboard
```

---

## Core design decisions

**1. Eval sets should over-represent risk, not mirror traffic.**
Sampling deliberately biases toward failure-flagged, low-confidence, and poorly-covered logs, rather than uniformly sampling everything.

**2. Coverage-gap-aware sampling (the actual flywheel).**
Before sampling new candidates, the pipeline checks which clusters are already well-covered by *approved* cases and re-embeds/re-maps approved cases to current cluster centroids (since clustering isn't stable across separate runs). Under-covered clusters get prioritized next. This is what makes it a loop rather than a single pass.

**3. Target model ≠ judge model.**
The eval runner never lets a model grade its own output. The response comes from one model (`qwen/qwen3.6-27b`); the pass/fail judgment comes from a different one (`openai/gpt-oss-20b`). This avoids the self-grading bias that undermines a lot of naive "LLM judges LLM" setups.

**4. Human review closes the loop on label quality — literally.**
Draft cases route to a reviewer when the auto-labeler's confidence is low. Every reviewer edit is logged with before/after values. Those edits now feed back automatically into the label-generation prompt (`build_review_guidance`) — the most-corrected fields for a given eval type get reinforced with real recent correction examples on every subsequent generation run.

**5. A failed eval doesn't mean "fix the case."**
Most failures mean the target model answered badly — that's the eval working correctly. The triage flow (deprecate + revise) exists specifically to prevent the failure mode where reviewers loosen a case until the model happens to pass it. Revising a case requires an explicit reason, deprecates the original (full history preserved, nothing deleted), and forces the revision through the *same* 6-point review bar as any new case — never auto-approved.

**6. Regression tracking is case-by-case, not just an aggregate score.**
Two eval runs can have the same pass rate while completely different cases flipped underneath. The regression diff surfaces exactly which cases improved or regressed between any two runs.

---

## Pipeline stages

| Phase | What happens | Key files |
|---|---|---|
| **1. Log generation** | 1,000 synthetic production-style logs (prompt, response, feedback, errors, retries) generated via Groq, with redaction for emails/phone numbers/secrets/names | `src/logs/` |
| **2. Sampling & clustering** | HDBSCAN clustering + three sampling strategies (random, failure-biased, diversity); candidates scored with coverage-gap awareness | `src/sampling/`, `src/clustering/`, `src/scoring/` |
| **3. Auto-label generation** | LLM decides eval type (golden answer / rubric / expected refusal), generates the label, self-scores confidence, dedups against existing cases; incorporates real reviewer-correction patterns into its own prompt | `src/eval_builder/` |
| **4. Human review** | Reviewer checklist (task alignment, correctness, testability, no eval leakage, no unjustified assumptions, appropriate difficulty); every edit tracked for provenance | `src/review/` |
| **5. Export & continuous evaluation** | Versioned JSONL export with changelog; eval runner (target model → separate judge model); regression diff across runs; dataset health dashboard | `src/export/`, `src/eval_runner/`, `dashboard/` (merged into frontend) |
| **6. API + unified frontend** | FastAPI layer wrapping the entire pipeline as pollable background jobs; single consolidated Streamlit app covering pipeline control, review, dashboard, and a "flywheel demo" tracing one log through every stage | `src/api/`, `frontend/` |

---

## Real numbers (not illustrative — actual results from this build)

- **1,000** synthetic logs generated
- **8 clusters** + 66 noise points discovered (min_cluster_size=5)
- **44 eval cases** produced, spanning all three eval types
- **11/44 (25%) pass rate** for `qwen/qwen3.6-27b` against the human-reviewed case set — a real, reproducible finding about how a smaller/cheaper model performs against rubric-strict criteria written around a stronger model's expected answers, not a vanity metric
- Multiple genuine defects caught purely through the confidence-routing → human-review pipeline: invalid SQL in a generated label, a duplicated before/after code example, eval-context leakage, a fabricated version-specific claim, a muddled multi-topic rubric

---

## Tech stack

- **Data layer:** DuckDB
- **LLM provider:** Groq (`openai/gpt-oss-20b` for generation/judging, `qwen/qwen3.6-27b` as the evaluated target model)
- **Clustering/embedding:** HDBSCAN, sentence-transformers, scikit-learn
- **Validation:** Pydantic v2
- **Backend:** FastAPI, background-threaded job execution for long-running LLM calls
- **Frontend:** Streamlit (multi-page, consolidated single app)
- **Language:** Python 3.13

---

## Running it

### Setup

```powershell
git clone <repo>
cd EvalLoop
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:
```
GROQ_API_KEY=your_key_here
```

### Start the app

```powershell
uvicorn src.api.main:app --reload
streamlit run frontend/Home.py --server.port 8501
```

That's it — every stage (log generation, clustering, label generation, human review, export, eval runs, dataset health, and the flywheel trace demo) is reachable from `http://localhost:8501`.

### Or run stages individually via CLI

```powershell
python -m src.logs.synthetic_generator      # Phase 1
python scripts/run_phase2.py                # Phase 2
python scripts/run_phase3.py                # Phase 3
python -m src.export.jsonl_exporter         # Phase 5 — export
python -m src.eval_runner.runner            # Phase 5 — run eval
```

---

## What breaks, and why that's part of the story

Every one of these was a real bug hit and fixed during this build, not a hypothetical:

- **Reasoning models silently burn their token budget on hidden `<think>` traces** unless `reasoning_effort` is set correctly — and different model families use different valid values (`"low"` for `gpt-oss`, `"none"`/`"default"` for `qwen`), discovered the hard way via empty/truncated outputs.
- **Per-minute output-token rate limits** can reject a single request outright if `max_tokens` alone exceeds the cap — no amount of retry/backoff helps until the request itself is resized.
- **DuckDB tables created via `CREATE TABLE IF NOT EXISTS` don't retroactively gain new columns** on an already-existing database file — schema changes need an explicit `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` migration alongside them.
- **A missing JSON-parse step** in a judge function silently returned a raw string instead of a `(bool, str)` tuple, surfacing as a confusing "too many values to unpack" error several layers away from the actual cause.
- **Offset-naive vs. offset-aware datetime subtraction** broke a dashboard freshness calculation the first time a timestamp came back without timezone info.

---

## Project structure

```
EvalLoop/
├── data/                   # DuckDB files, JSONL exports, changelog
├── src/
│   ├── logs/               # Phase 1 — schema, redaction, synthetic generation
│   ├── sampling/            # Phase 2 — random / failure-biased / diversity samplers
│   ├── clustering/          # Phase 2 — embedding + HDBSCAN
│   ├── scoring/              # Phase 2 — candidate scoring, coverage-gap logic
│   ├── eval_builder/         # Phase 3 — label generation, judging, dedup, schema
│   ├── review/                # Phase 4 — review queue, edit tracking
│   ├── export/                  # Phase 5 — JSONL export
│   ├── eval_runner/              # Phase 5 — target/judge run loop, metrics
│   ├── api/                       # Phase 6 — FastAPI layer
│   └── utils/                      # DB helpers, retry/backoff, constants
├── frontend/                # Phase 6 — unified Streamlit app (all pages)
├── scripts/                  # Verification and CLI entry points
└── requirements.txt
```

---

## Roadmap / honest gaps

- Nightly/scheduled eval runs (currently manual-trigger only)
- Auto-label acceptance rate (% approved without edit) — data exists (`ReviewEdit` table), not yet surfaced as a headline dashboard metric
- Formal `pytest` suite (testing so far is ad-hoc + verification scripts)
- Scaling beyond the current 1,000-log / 44-case demo dataset

---

## What this project demonstrates

LLM-as-judge design (with self-grading bias explicitly avoided), human-in-the-loop review workflows with full provenance tracking, eval-set construction philosophy grounded in risk over representativeness, regression testing for non-deterministic LLM outputs, a real production-style data flywheel with coverage-gap feedback, and full-stack ownership from data layer through API to frontend.
