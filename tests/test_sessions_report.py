import pickle

import pytest

from signalsift.pipeline import Analysis
from signalsift.report import build_report
from signalsift.sessions import SessionNotFound, SessionStore


def test_roundtrip_and_list(tmp_path) -> None:
    store = SessionStore(tmp_path)
    sid = store.new_id()
    store.save(sid, {"x": 1}, {"session_id": sid, "created_at": "2026-01-01"})
    assert store.load(sid) == {"x": 1}
    assert store.list()[0]["session_id"] == sid
    store.delete(sid)
    with pytest.raises(SessionNotFound):
        store.load(sid)


@pytest.mark.parametrize("bad", ["../etc/passwd", "ABCDEF123456", "short", "abcdef12345g"])
def test_rejects_unsafe_ids(tmp_path, bad: str) -> None:
    with pytest.raises(SessionNotFound):
        SessionStore(tmp_path).load(bad)


def test_delete_missing_raises(tmp_path) -> None:
    with pytest.raises(SessionNotFound):
        SessionStore(tmp_path).delete("abcdef123456")


def test_analysis_pickles_without_engine(analysis: Analysis, settings) -> None:
    analysis.engine(settings)
    clone = pickle.loads(pickle.dumps(analysis))
    assert clone._engine is None
    assert clone.engine(settings).search("refund", k=1)


def test_report_mentions_incident_and_topics(analysis: Analysis) -> None:
    md = build_report(analysis, [{"system": "Hybrid", "recall": 0.9}])
    assert md.startswith("# SignalSift report")
    assert "incident" in md.lower() and "## Topics" in md and "## Retrieval quality" in md
