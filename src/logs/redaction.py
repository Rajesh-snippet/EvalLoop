"""
Redaction rules — regex-based (v1). Detects and masks:
  - emails
  - phone numbers (US-style + generic international)
  - secrets/API keys (common key-shaped patterns: sk-..., AKIA..., ghp_..., etc.)
  - person names (heuristic — see NAME NOTE below)

Design note: this is deliberately regex-only, not an ML/NER-based redactor
(e.g. presidio). That's a documented tradeoff: regex is fast, dependency-free,
and fully explainable (every match traces to a rule you can point at), but it
will miss stylistic name variants and non-US phone formats that an NER-based
system would catch. redaction_method='regex_v1' is stored on every redacted
log specifically so a future swap to a stronger redactor is auditable —
you can query "which logs were only ever regex-redacted" and re-run them.

NAME NOTE: true name detection needs NER for real text. As a heuristic
stand-in, we redact capitalized two-word sequences that follow common
self-identification patterns ("my name is X Y", "I'm X Y", "this is X Y").
This intentionally has low recall (won't catch names in bare form) to avoid
mangling normal capitalized words (e.g. "New York", "Black Friday") with a
naive "any two capitalized words" rule. This limitation is called out again
in README.md as a Phase 6 talking point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Matches common US formats (555-123-4567, (555) 123-4567, 555.123.4567)
# and loose international formats with a leading +.
PHONE_RE = re.compile(
    r"(\+\d{1,3}[\s-]?)?(\(\d{3}\)[\s-]?|\d{3}[\s.-])\d{3}[\s.-]\d{4}\b"
)

SECRET_PATTERNS = {
    "openai_key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "generic_bearer": re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]{20,}"),
    "generic_secret_kv": re.compile(
        r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*[\"']?[A-Za-z0-9\-._~+/]{8,}[\"']?"
    ),
}

NAME_TRIGGER_RE = re.compile(
    r"(?:my name is|i'?m|this is|i am)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)"
)


@dataclass
class RedactionResult:
    text: str
    redacted: bool = False
    matched_types: list = field(default_factory=list)


def redact_text(text: str) -> RedactionResult:
    """Run all redaction rules over a single string, returning the cleaned
    text plus which rule types fired (for the 'redaction_method' audit trail
    at the log level, and for building a report of what got caught)."""
    matched: list[str] = []
    out = text

    if EMAIL_RE.search(out):
        matched.append("email")
        out = EMAIL_RE.sub("[REDACTED_EMAIL]", out)

    if PHONE_RE.search(out):
        matched.append("phone")
        out = PHONE_RE.sub("[REDACTED_PHONE]", out)

    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(out):
            matched.append(f"secret:{label}")
            out = pattern.sub("[REDACTED_SECRET]", out)

    def _mask_name(m: re.Match) -> str:
        return m.group(0).replace(m.group(1), "[REDACTED_NAME]")

    if NAME_TRIGGER_RE.search(out):
        matched.append("name")
        out = NAME_TRIGGER_RE.sub(_mask_name, out)

    return RedactionResult(text=out, redacted=bool(matched), matched_types=matched)


def redact_log_fields(prompt: str, response: str) -> dict:
    """Redact both prompt and response, return a dict ready to merge into a
    LogEntry: {prompt, response, redacted, redaction_method}."""
    p = redact_text(prompt)
    r = redact_text(response)
    any_redacted = p.redacted or r.redacted
    return {
        "prompt": p.text,
        "response": r.text,
        "redacted": any_redacted,
        "redaction_method": "regex_v1" if any_redacted else None,
        "matched_types": sorted(set(p.matched_types + r.matched_types)),
    }
