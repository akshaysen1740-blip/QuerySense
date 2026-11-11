"""Clients used by the Streamlit UI.

``LocalClient`` runs the service in-process (single container / Hugging Face Spaces).
``ApiClient`` talks to the FastAPI backend over HTTP (docker-compose, decoupled deploys).
Both expose the same methods and return the same JSON shapes.
"""

from __future__ import annotations

import io
from typing import Any, Protocol

import httpx
import pandas as pd

from .config import Settings
from .service import InsightService


class Client(Protocol):
    def create_session(
        self, csv_bytes: bytes, name: str, text_col: str | None, time_col: str | None,
        n_topics: int | None,
    ) -> dict[str, Any]: ...  # fmt: skip
    def create_sample_session(self) -> dict[str, Any]: ...
    def list_sessions(self) -> list[dict[str, Any]]: ...
    def delete(self, session_id: str) -> None: ...
    def summary(self, session_id: str) -> dict[str, Any]: ...
    def topics(self, session_id: str) -> list[dict[str, Any]]: ...
    def topic_tickets(
        self, session_id: str, topic_id: int, limit: int = 50
    ) -> list[dict[str, Any]]: ...
    def trends(self, session_id: str) -> dict[str, Any]: ...
    def incidents(self, session_id: str) -> list[dict[str, Any]]: ...
    def search(
        self, session_id: str, query: str, k: int, mode: str, rerank: bool
    ) -> list[dict[str, Any]]: ...
    def evaluate(
        self, session_id: str, queries: list[dict[str, Any]] | None, k: int
    ) -> dict[str, Any]: ...
    def clusters_csv(self, session_id: str) -> str: ...
    def report_md(self, session_id: str) -> str: ...


class LocalClient:
    def __init__(self, settings: Settings) -> None:
        self.service = InsightService(settings)

    def create_session(
        self, csv_bytes: bytes, name: str, text_col: str | None, time_col: str | None,
        n_topics: int | None,
    ) -> dict[str, Any]:  # fmt: skip
        frame = pd.read_csv(io.BytesIO(csv_bytes))
        return self.service.create_session(frame, name, text_col, time_col, n_topics)

    def create_sample_session(self) -> dict[str, Any]:
        return self.service.create_sample_session()

    def list_sessions(self) -> list[dict[str, Any]]:
        return self.service.list_sessions()

    def delete(self, session_id: str) -> None:
        self.service.delete(session_id)

    def summary(self, session_id: str) -> dict[str, Any]:
        return self.service.summary(session_id)

    def topics(self, session_id: str) -> list[dict[str, Any]]:
        return self.service.topics(session_id)

    def topic_tickets(
        self, session_id: str, topic_id: int, limit: int = 50
    ) -> list[dict[str, Any]]:
        return self.service.topic_tickets(session_id, topic_id, limit)

    def trends(self, session_id: str) -> dict[str, Any]:
        return self.service.trends(session_id)

    def incidents(self, session_id: str) -> list[dict[str, Any]]:
        return self.service.incidents(session_id)

    def search(
        self, session_id: str, query: str, k: int, mode: str, rerank: bool
    ) -> list[dict[str, Any]]:
        return self.service.search(session_id, query, k, mode, rerank)

    def evaluate(
        self, session_id: str, queries: list[dict[str, Any]] | None, k: int
    ) -> dict[str, Any]:
        return self.service.evaluate(session_id, queries, k)

    def clusters_csv(self, session_id: str) -> str:
        return self.service.clusters_csv(session_id)

    def report_md(self, session_id: str) -> str:
        return self.service.report_md(session_id)


class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(
        self, base_url: str, timeout: float = 300.0, http: httpx.Client | None = None
    ) -> None:
        self._http = http or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def _call(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        resp = self._http.request(method, path, **kwargs)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise ApiError(f"{resp.status_code}: {detail}")
        return resp

    def create_session(
        self, csv_bytes: bytes, name: str, text_col: str | None, time_col: str | None,
        n_topics: int | None,
    ) -> dict[str, Any]:  # fmt: skip
        form = {"name": name, "text_col": text_col or "", "time_col": time_col or ""}
        if n_topics is not None:
            form["n_topics"] = str(n_topics)
        files = {"file": (name, csv_bytes, "text/csv")}
        result: dict[str, Any] = self._call("POST", "/sessions", data=form, files=files).json()
        return result

    def create_sample_session(self) -> dict[str, Any]:
        result: dict[str, Any] = self._call("POST", "/sessions/sample").json()
        return result

    def list_sessions(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._call("GET", "/sessions").json()
        return result

    def delete(self, session_id: str) -> None:
        self._call("DELETE", f"/sessions/{session_id}")

    def summary(self, session_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._call("GET", f"/sessions/{session_id}").json()
        return result

    def topics(self, session_id: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._call("GET", f"/sessions/{session_id}/topics").json()
        return result

    def topic_tickets(
        self, session_id: str, topic_id: int, limit: int = 50
    ) -> list[dict[str, Any]]:
        path = f"/sessions/{session_id}/topics/{topic_id}/tickets"
        result: list[dict[str, Any]] = self._call("GET", path, params={"limit": limit}).json()
        return result

    def trends(self, session_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._call("GET", f"/sessions/{session_id}/trends").json()
        return result

    def incidents(self, session_id: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._call("GET", f"/sessions/{session_id}/incidents").json()
        return result

    def search(
        self, session_id: str, query: str, k: int, mode: str, rerank: bool
    ) -> list[dict[str, Any]]:
        body = {"query": query, "k": k, "mode": mode, "rerank": rerank}
        result: list[dict[str, Any]] = self._call(
            "POST", f"/sessions/{session_id}/search", json=body
        ).json()
        return result

    def evaluate(
        self, session_id: str, queries: list[dict[str, Any]] | None, k: int
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"k": k}
        if queries:
            body["queries"] = queries
        result: dict[str, Any] = self._call(
            "POST", f"/sessions/{session_id}/evaluate", json=body
        ).json()
        return result

    def clusters_csv(self, session_id: str) -> str:
        return self._call("GET", f"/sessions/{session_id}/export/clusters.csv").text

    def report_md(self, session_id: str) -> str:
        return self._call("GET", f"/sessions/{session_id}/export/report.md").text


def get_client(settings: Settings) -> Client:
    if settings.ui_mode == "api":
        return ApiClient(settings.api_url)
    return LocalClient(settings)
