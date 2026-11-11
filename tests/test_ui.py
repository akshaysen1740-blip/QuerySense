"""Headless smoke test of the Streamlit UI (in-process client)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from signalsift.config import get_settings  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_sample_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGNALSIFT_SESSION_DIR", str(tmp_path))
    monkeypatch.setenv("SIGNALSIFT_UI_MODE", "local")
    get_settings.cache_clear()
    try:
        at = AppTest.from_file(APP, default_timeout=180).run()
        assert not at.exception
        at.sidebar.button[0].click().run()  # "Analyze sample dataset"
        assert not at.exception
        assert {m.label for m in at.metric} >= {"Tickets", "Topics", "Incidents"}

        at.text_input[0].input("app crashing since the update")
        at.button(key="FormSubmitter:search-Search").click().run()
        assert not at.exception

        next(b for b in at.button if b.label == "Run evaluation").click()
        at.run()
        assert not at.exception
    finally:
        get_settings.cache_clear()
