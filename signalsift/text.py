"""Shared text constants."""

from __future__ import annotations

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# PII placeholders and support-chat filler that carry no topical meaning.
FILLER_WORDS = {
    "email", "phone", "url", "card", "ip", "handle", "pls", "plz", "thanks", "thank", "hi",
    "hello", "hey", "help", "guys", "team", "ugh", "urgent", "dm", "just", "really", "today",
}  # fmt: skip
STOP_WORDS: list[str] = sorted(ENGLISH_STOP_WORDS | FILLER_WORDS)
