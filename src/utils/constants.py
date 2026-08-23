"""Shared constants for the synthetic log generator and downstream phases."""

from __future__ import annotations

# Each "feature" simulates a distinct production LLM use case. Keeping these
# varied in style/domain matters for Phase 2 clustering — if all features
# were similar (e.g. all customer support), clustering would have nothing
# interesting to discover.
FEATURES = [
    {
        "name": "customer_support_bot",
        "system_prompt": "You are a helpful, empathetic customer support assistant for an e-commerce company. Help users with orders, returns, and account issues.",
        "topic_hint": "e-commerce customer support: orders, shipping, returns, refunds, account issues",
    },
    {
        "name": "sql_query_generator",
        "system_prompt": "You are a SQL assistant. Convert natural language questions into correct SQL queries for a Postgres database with tables: users, orders, products, payments.",
        "topic_hint": "natural language to SQL queries about users, orders, products, payments",
    },
    {
        "name": "code_review_assistant",
        "system_prompt": "You are a senior software engineer reviewing pull requests. Point out bugs, style issues, and suggest improvements.",
        "topic_hint": "code review comments on Python/JavaScript pull request snippets",
    },
    {
        "name": "meal_planner",
        "system_prompt": "You are a nutrition assistant that creates meal plans based on dietary preferences, allergies, and calorie goals.",
        "topic_hint": "meal planning requests with dietary restrictions, allergies, calorie targets",
    },
    {
        "name": "resume_screener",
        "system_prompt": "You are an HR assistant that screens resumes against job descriptions and summarizes candidate fit.",
        "topic_hint": "resume screening against job descriptions for various roles",
    },
    {
        "name": "internal_docs_qa",
        "system_prompt": "You are an internal knowledge base assistant. Answer employee questions using company policy and documentation. If unsure, say you don't know rather than guessing.",
        "topic_hint": "employee questions about HR policy, expense reports, PTO, IT support",
    },
]

# Models used to tag synthetic logs (not actually swapped per-call — just
# varied in metadata to simulate a multi-model production environment).
SIMULATED_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

# NOTE: llama-3.3-70b-versatile was deprecated for free/developer-tier Groq
# usage (mid-2026).
#
# We initially switched to openai/gpt-oss-120b, but that's a *reasoning*
# model — it spends hidden "reasoning" tokens on every call (counted against
# your daily quota even though you never see them), and its free-tier daily
# token limit (200K TPD) is easy to exhaust well before 1000 logs are
# generated (observed: ~213 logs consumed the full 200K budget).
#
# Switched to llama-3.1-8b-instant instead: not a reasoning model (no hidden
# token overhead), much cheaper per call, and draws from a separate quota
# bucket entirely, so it isn't affected by gpt-oss-120b's daily cap.
# If this model 404s on your account (deprecation status has been
# inconsistent across Groq's docs), try "openai/gpt-oss-20b" instead — same
# family as 120b but smaller, so it burns quota more slowly even though it's
# still a reasoning model.
GROQ_GENERATION_MODEL = "llama-3.1-8b-instant"

# Reasoning-capable models need reasoning_effort passed to avoid burning their
# whole token budget on hidden reasoning with nothing left for the answer.
# Non-reasoning models (like llama-3.1-8b-instant) don't support/need this
# parameter at all — passing it could error, so _chat() checks this list.
REASONING_MODELS = {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}

# Response category weights — deliberately NOT uniform. Real production
# traffic skews heavily toward "good", but we want a realistic minority of
# failure modes for the failure-biased sampler (Phase 2) to have signal on.
RESPONSE_CATEGORY_WEIGHTS = {
    "good": 0.70,
    "bad": 0.12,
    "malformed": 0.06,
    "retry_sequence": 0.07,
    "safety_edge_case": 0.05,
}
