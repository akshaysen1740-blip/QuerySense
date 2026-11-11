"""End-to-end analysis: ingest -> redact -> embed -> topics -> trends -> incidents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .clustering import TopicResult, discover_topics
from .config import Settings
from .embeddings import Embedder, build_embedder
from .ingest import IngestReport, load_tickets
from .rerank import build_reranker
from .search import SearchEngine
from .trends import Incident, Spike, detect_incidents, detect_spikes, topic_volume

MIN_ROWS = 10


@dataclass
class Analysis:
    session_id: str
    name: str
    created_at: str
    df: pd.DataFrame
    embeddings: np.ndarray
    embedder: Embedder
    topics: TopicResult
    volume: pd.DataFrame
    spikes: list[Spike]
    incidents: list[Incident]
    ingest: IngestReport
    config: dict[str, Any]
    last_eval: list[dict[str, Any]] | None = None
    _engine: SearchEngine | None = field(default=None, repr=False)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_engine"] = None  # rebuilt on load; FAISS indexes aren't picklable
        return state

    def engine(self, settings: Settings) -> SearchEngine:
        if self._engine is None:
            self._engine = SearchEngine(
                self.df["text"].tolist(),
                self.embeddings,
                self.embedder,
                vector_backend=settings.vector_backend,
                reranker=build_reranker(settings),
                rerank_candidates=settings.rerank_candidates,
            )
        return self._engine

    @property
    def has_time(self) -> bool:
        return bool(self.df["created_at"].notna().any())


def analyze(
    source: Any,
    settings: Settings,
    session_id: str,
    name: str = "untitled",
    text_col: str | None = None,
    time_col: str | None = None,
    n_topics: int | None = None,
) -> Analysis:
    df, report = load_tickets(
        source,
        text_col=text_col,
        time_col=time_col,
        redact_pii=settings.redact_pii,
        max_rows=settings.max_rows,
    )
    if len(df) < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} non-empty tickets, got {len(df)}")

    texts = df["text"].tolist()
    embedder = build_embedder(settings).fit(texts)
    embeddings = embedder.encode(texts)
    topics = discover_topics(
        texts,
        embeddings,
        n_topics=settings.n_topics if n_topics is None else n_topics,
        method=settings.cluster_method,
    )
    df["topic"] = topics.labels

    volume = topic_volume(df["created_at"], topics.labels, settings.trend_freq)
    spikes = detect_spikes(
        volume, settings.spike_z, settings.spike_min_count, settings.spike_window
    )
    incidents = detect_incidents(
        embeddings,
        df["created_at"],
        topics.labels,
        texts,
        threshold=settings.duplicate_threshold,
        min_size=settings.incident_min_size,
        window_hours=settings.incident_window_hours,
        burst_ratio=settings.incident_burst_ratio,
        merge_threshold=settings.incident_merge_threshold,
    )
    return Analysis(
        session_id=session_id,
        name=name,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        df=df,
        embeddings=embeddings,
        embedder=embedder,
        topics=topics,
        volume=volume,
        spikes=spikes,
        incidents=incidents,
        ingest=report,
        config={
            "embedder": embedder.name,
            "cluster_method": topics.method,
            "trend_freq": settings.trend_freq,
            "reranker": settings.reranker,
            "vector_backend": settings.vector_backend,
        },
    )
