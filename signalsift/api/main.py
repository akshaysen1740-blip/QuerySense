"""FastAPI backend. Run with ``uvicorn signalsift.api.main:app --reload``."""

from __future__ import annotations

import io
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from .. import __version__
from ..config import Settings, get_settings
from ..service import InsightService, SessionNotFound


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(10, ge=1, le=100)
    mode: Literal["tfidf", "bm25", "dense", "hybrid"] = "hybrid"
    rerank: bool = True


class GoldenQueryIn(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    relevant_ids: list[str] | None = None
    relevant_labels: list[str] | None = None


class EvaluateRequest(BaseModel):
    queries: list[GoldenQueryIn] | None = Field(
        None, description="Omit to use the bundled golden set (labelled datasets only)"
    )
    k: int = Field(10, ge=1, le=100)
    label_column: str = "intent"


class SampleRequest(BaseModel):
    n: int = Field(3000, ge=100, le=20000)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    service = InsightService(settings)
    app = FastAPI(
        title="SignalSift API",
        version=__version__,
        description="Topic discovery, semantic search, trend and incident detection for tickets.",
    )
    app.state.service = service

    @app.exception_handler(SessionNotFound)
    async def _not_found(_: Request, exc: SessionNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"Session {exc} not found"})

    @app.exception_handler(ValueError)
    async def _bad_input(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "embedder": settings.embedder}

    @app.post("/sessions", status_code=201)
    async def create_session(
        file: UploadFile = File(...),
        name: str | None = Form(None),
        text_col: str | None = Form(None),
        time_col: str | None = Form(None),
        n_topics: int | None = Form(None, ge=0, le=200),
    ) -> dict[str, Any]:
        data = await file.read()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
        try:
            frame = pd.read_csv(io.BytesIO(data))
        except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
            raise HTTPException(422, f"Could not parse CSV: {exc}") from exc
        return service.create_session(
            frame, name or file.filename or "upload", text_col or None, time_col or None, n_topics
        )

    @app.post("/sessions/sample", status_code=201)
    def create_sample(body: SampleRequest | None = None) -> dict[str, Any]:
        return service.create_sample_session(n=(body or SampleRequest()).n)

    @app.get("/sessions")
    def list_sessions() -> list[dict[str, Any]]:
        return service.list_sessions()

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        return service.summary(session_id)

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> Response:
        service.delete(session_id)
        return Response(status_code=204)

    @app.get("/sessions/{session_id}/topics")
    def topics(session_id: str) -> list[dict[str, Any]]:
        return service.topics(session_id)

    @app.get("/sessions/{session_id}/topics/{topic_id}/tickets")
    def topic_tickets(session_id: str, topic_id: int, limit: int = 50) -> list[dict[str, Any]]:
        return service.topic_tickets(session_id, topic_id, min(limit, 500))

    @app.get("/sessions/{session_id}/trends")
    def trends(session_id: str) -> dict[str, Any]:
        return service.trends(session_id)

    @app.get("/sessions/{session_id}/incidents")
    def incidents(session_id: str) -> list[dict[str, Any]]:
        return service.incidents(session_id)

    @app.post("/sessions/{session_id}/search")
    def search(session_id: str, body: SearchRequest) -> list[dict[str, Any]]:
        return service.search(session_id, body.query, body.k, body.mode, body.rerank)

    @app.post("/sessions/{session_id}/evaluate")
    def evaluate(session_id: str, body: EvaluateRequest | None = None) -> dict[str, Any]:
        body = body or EvaluateRequest()
        queries = [q.model_dump() for q in body.queries] if body.queries else None
        return service.evaluate(session_id, queries, body.k, body.label_column)

    @app.get("/sessions/{session_id}/export/clusters.csv", response_class=PlainTextResponse)
    def export_clusters(session_id: str) -> PlainTextResponse:
        return PlainTextResponse(
            service.clusters_csv(session_id),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="clusters_{session_id}.csv"'},
        )

    @app.get("/sessions/{session_id}/export/report.md", response_class=PlainTextResponse)
    def export_report(session_id: str) -> PlainTextResponse:
        return PlainTextResponse(
            service.report_md(session_id),
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="report_{session_id}.md"'},
        )

    return app


app = create_app()
