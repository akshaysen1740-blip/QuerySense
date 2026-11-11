"""Regex-based PII redaction applied *before* any text is embedded or stored."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Order matters: URLs and emails are removed before handles; IPs before card/phone numbers.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("URL", re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)),
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){12,15}\d\b")),
    (
        "PHONE",
        re.compile(r"(?<![\w#])(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\w)"),
    ),
    ("HANDLE", re.compile(r"(?<![\w\[])@\w{2,}")),
]


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)


def redact(text: str) -> RedactionResult:
    """Replace PII spans with ``[LABEL]`` placeholders and count what was removed."""
    counts: dict[str, int] = {}
    for label, pattern in PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            counts[label] = counts.get(label, 0) + n
    return RedactionResult(text=text, counts=counts)
