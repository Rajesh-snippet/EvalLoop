# EvalLoop — Project Documentation
### Production Log-to-Eval Dataset Builder — Build Plan, Execution Log, and Status (Phases 1–3)

---

## 1. The Original Build Plan

**Project**: EvalLoop — a pipeline that mines production-like LLM logs, finds useful examples, converts them into evaluation cases, labels them with expected behavior or rubrics, and routes uncertain cases to a human review queue.

**Core narrative (the data flywheel)**:
```
Production Logs → Redaction → Sampling → Clustering → Candidate Scoring
→ Eval Case Builder → LLM-as-Judge → Confidence Score
→ [High confidence: Auto-approved] / [Low confidence: Human Review]
→ Approved Dataset → JSONL → Eval Runner → Metrics → Dataset Health
```
