"""Load arbitrary ticket CSVs into a normalised frame: id, text, created_at (+ extra columns)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import pandas as pd

from .pii import redact

TEXT_CANDIDATES = [
    "text",
    "body",
    "message",
    "description",
    "ticket",
    "content",
    "comment",
    "feedback",
]
TIME_CANDIDATES = ["created_at", "timestamp", "date", "time", "created", "datetime"]
ID_CANDIDATES = ["id", "ticket_id", "tweet_id", "message_id"]
MIN_CHARS = 3
_WS = re.compile(r"\s+")


@dataclass
class IngestReport:
    rows_in: int
    rows_out: int
    dropped_empty: int
    truncated: int
    exact_duplicates: int
    text_col: str
    time_col: str | None
    pii_redactions: dict[str, int] = field(default_factory=dict)


def _pick(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def read_table(source: str | Path | IO[bytes] | IO[str] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(source)


def load_tickets(
    source: str | Path | IO[bytes] | IO[str] | pd.DataFrame,
    text_col: str | None = None,
    time_col: str | None = None,
    redact_pii: bool = True,
    max_rows: int = 50_000,
) -> tuple[pd.DataFrame, IngestReport]:
    """Normalise a raw table. Raises ``ValueError`` when no usable text column exists."""
    raw = read_table(source)
    rows_in = len(raw)
    columns = [str(c) for c in raw.columns]
    raw.columns = columns

    text_col = text_col or _pick(columns, TEXT_CANDIDATES)
    if text_col is None or text_col not in raw.columns:
        raise ValueError(
            f"Could not find a text column. Pass text_col explicitly. Columns: {columns}"
        )
    if time_col is not None and time_col not in raw.columns:
        raise ValueError(f"time_col '{time_col}' not in columns {columns}")
    time_col = time_col or _pick(columns, TIME_CANDIDATES)
    id_col = _pick(columns, ID_CANDIDATES)

    df = raw.copy()
    df["_text"] = (
        df[text_col].astype("string").fillna("").map(lambda s: _WS.sub(" ", str(s)).strip())
    )
    keep = df["_text"].str.len() >= MIN_CHARS
    dropped = int((~keep).sum())
    df = df[keep]

    truncated = max(0, len(df) - max_rows)
    df = df.head(max_rows)

    totals: dict[str, int] = {}
    texts: list[str] = []
    for t in df["_text"]:
        if redact_pii:
            res = redact(str(t))
            texts.append(res.text)
            for k, v in res.counts.items():
                totals[k] = totals.get(k, 0) + v
        else:
            texts.append(str(t))

    out = pd.DataFrame(
        {
            "id": df[id_col].astype(str).to_numpy()
            if id_col
            else [f"t{i}" for i in range(len(df))],
            "text": texts,
        }
    )
    if time_col:
        ts = pd.to_datetime(df[time_col], errors="coerce", utc=True, format="mixed")
        out["created_at"] = ts.dt.tz_convert(None).to_numpy()
    else:
        out["created_at"] = pd.NaT

    skip = {text_col, time_col, id_col, "_text", "id", "text", "created_at"}
    for col in df.columns:
        if col not in skip:
            out[col] = df[col].to_numpy()

    report = IngestReport(
        rows_in=rows_in,
        rows_out=len(out),
        dropped_empty=dropped,
        truncated=truncated,
        exact_duplicates=int(out["text"].duplicated().sum()),
        text_col=text_col,
        time_col=time_col,
        pii_redactions=totals,
    )
    return out, report
