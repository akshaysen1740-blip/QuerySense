"""Both UI transports must behave identically."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from signalsift.api.main import create_app
from signalsift.client import ApiClient, ApiError, Client, LocalClient, get_client
from signalsift.config import Settings
from signalsift.datasets import generate_tickets


def _settings(tmp_path: Path) -> Settings:
    return Settings(session_dir=tmp_path, lsa_dims=32)


@pytest.fixture(params=["local", "api"])
def client(request: pytest.FixtureRequest, tmp_path: Path) -> Client:
    settings = _settings(tmp_path)
    if request.param == "local":
        return LocalClient(settings)
    return ApiClient("http://test", http=TestClient(create_app(settings)))


def test_full_flow(client: Client) -> None:
    csv = generate_tickets(n=400, seed=9).to_csv(index=False).encode()
    sid = client.create_session(csv, "flow.csv", "body", "created_at", 8)["session_id"]
    assert client.summary(sid)["topics"] == 8
    topics = client.topics(sid)
    assert client.topic_tickets(sid, topics[0]["topic_id"], 5)
    assert client.search(sid, "refund", 3, "hybrid", True)
    assert "series" in client.trends(sid)
    assert isinstance(client.incidents(sid), list)
    assert client.evaluate(sid, None, 10)["summary"]
    assert client.clusters_csv(sid).startswith("id,")
    assert client.report_md(sid).startswith("# SignalSift")
    assert any(s["session_id"] == sid for s in client.list_sessions())
    client.delete(sid)
    assert all(s["session_id"] != sid for s in client.list_sessions())


def test_sample_session(client: Client) -> None:
    assert client.create_sample_session()["rows"] == 3000


def test_api_errors_are_raised(tmp_path: Path) -> None:
    api = ApiClient("http://test", http=TestClient(create_app(_settings(tmp_path))))
    with pytest.raises(ApiError, match="404"):
        api.summary("abcdef123456")


def test_get_client_modes(tmp_path: Path) -> None:
    assert isinstance(get_client(_settings(tmp_path)), LocalClient)
    api_settings = Settings(session_dir=tmp_path, ui_mode="api", api_url="http://x")
    assert isinstance(get_client(api_settings), ApiClient)
