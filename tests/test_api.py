from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from signalsift.api.main import create_app
from signalsift.config import Settings
from signalsift.datasets import generate_tickets


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    settings = Settings(session_dir=Path(tmp_path_factory.mktemp("api")), lsa_dims=48)
    return TestClient(create_app(settings))


@pytest.fixture(scope="module")
def session_id(client: TestClient) -> str:
    csv = generate_tickets(n=600, seed=5).to_csv(index=False).encode()
    resp = client.post(
        "/sessions",
        files={"file": ("tickets.csv", io.BytesIO(csv), "text/csv")},
        data={"name": "api test"},
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["session_id"])


def test_health(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"


def test_summary_and_list(client: TestClient, session_id: str) -> None:
    summary = client.get(f"/sessions/{session_id}").json()
    assert summary["rows"] == 600 and summary["has_labels"]
    assert any(s["session_id"] == session_id for s in client.get("/sessions").json())


def test_topics_and_tickets(client: TestClient, session_id: str) -> None:
    topics = client.get(f"/sessions/{session_id}/topics").json()
    assert topics and {"label", "keywords", "examples"} <= topics[0].keys()
    tickets = client.get(f"/sessions/{session_id}/topics/{topics[0]['topic_id']}/tickets").json()
    assert tickets


def test_search(client: TestClient, session_id: str) -> None:
    resp = client.post(f"/sessions/{session_id}/search", json={"query": "lost suitcase", "k": 5})
    hits = resp.json()
    assert resp.status_code == 200 and len(hits) == 5 and hits[0]["rank"] == 1


def test_search_validation(client: TestClient, session_id: str) -> None:
    resp = client.post(f"/sessions/{session_id}/search", json={"query": "", "k": 5})
    assert resp.status_code == 422
    resp = client.post(f"/sessions/{session_id}/search", json={"query": "x", "mode": "magic"})
    assert resp.status_code == 422


def test_trends_and_incidents(client: TestClient, session_id: str) -> None:
    trends = client.get(f"/sessions/{session_id}/trends").json()
    assert trends["series"] and "spikes" in trends
    assert isinstance(client.get(f"/sessions/{session_id}/incidents").json(), list)


def test_evaluate_default_and_custom(client: TestClient, session_id: str) -> None:
    default = client.post(f"/sessions/{session_id}/evaluate").json()
    assert len(default["summary"]) == 5
    custom = client.post(
        f"/sessions/{session_id}/evaluate",
        json={"queries": [{"query": "refund", "relevant_labels": ["refund_delay"]}], "k": 5},
    ).json()
    assert custom["k"] == 5 and len(custom["per_query"]) == 5


def test_exports(client: TestClient, session_id: str) -> None:
    csv = client.get(f"/sessions/{session_id}/export/clusters.csv")
    assert csv.status_code == 200 and "topic_label" in csv.text.splitlines()[0]
    md = client.get(f"/sessions/{session_id}/export/report.md")
    assert "## Retrieval quality" in md.text  # evaluation above is persisted into the report


def test_unknown_session_404(client: TestClient) -> None:
    assert client.get("/sessions/abcdef123456").status_code == 404
    assert client.get("/sessions/../../etc").status_code == 404


def test_bad_uploads(client: TestClient) -> None:
    resp = client.post("/sessions", files={"file": ("x.csv", b"foo\nbar\n", "text/csv")})
    assert resp.status_code == 422
    resp = client.post("/sessions", files={"file": ("x.csv", b"text\nhi there\n", "text/csv")})
    assert resp.status_code == 422  # fewer than MIN_ROWS tickets


def test_sample_and_delete(client: TestClient) -> None:
    sid = client.post("/sessions/sample", json={"n": 300}).json()["session_id"]
    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert client.get(f"/sessions/{sid}").status_code == 404
