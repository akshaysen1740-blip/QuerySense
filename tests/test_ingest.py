import io

import pandas as pd
import pytest

from signalsift.ingest import load_tickets


def test_autodetects_columns_and_redacts() -> None:
    raw = pd.DataFrame(
        {
            "ticket_id": ["a", "b", "c"],
            "body": ["refund please x@y.com", "  ", "app   crashed"],
            "created_at": ["2026-01-01T10:00", "2026-01-02", "not a date"],
            "channel": ["email", "chat", "app"],
        }
    )
    df, report = load_tickets(raw)
    assert list(df["id"]) == ["a", "c"]
    assert df.loc[0, "text"] == "refund please [EMAIL]"
    assert df.loc[1, "text"] == "app crashed"
    assert pd.isna(df.loc[1, "created_at"])
    assert "channel" in df.columns
    assert report.dropped_empty == 1 and report.pii_redactions == {"EMAIL": 1}
    assert report.text_col == "body" and report.time_col == "created_at"


def test_reads_csv_file_like() -> None:
    csv = io.StringIO("message\nhello there\nsecond ticket\n")
    df, report = load_tickets(csv)
    assert report.rows_out == 2 and df["created_at"].isna().all()
    assert list(df["id"]) == ["t0", "t1"]


def test_missing_text_column_raises() -> None:
    with pytest.raises(ValueError, match="text column"):
        load_tickets(pd.DataFrame({"foo": ["bar"]}))


def test_unknown_time_column_raises() -> None:
    with pytest.raises(ValueError, match="time_col"):
        load_tickets(pd.DataFrame({"text": ["hello"]}), time_col="nope")


def test_max_rows_truncates() -> None:
    df, report = load_tickets(
        pd.DataFrame({"text": [f"ticket {i}" for i in range(20)]}), max_rows=5
    )
    assert len(df) == 5 and report.truncated == 15


def test_redaction_can_be_disabled() -> None:
    df, _ = load_tickets(pd.DataFrame({"text": ["mail a@b.com"]}), redact_pii=False)
    assert df.loc[0, "text"] == "mail a@b.com"
