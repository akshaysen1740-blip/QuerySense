"""Application service shared by the FastAPI routes and the in-process UI client.

Every method returns JSON-serialisable data so both transports behave identically.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import pandas as pd

from .config import Settings
from .datasets import GOLDEN_QUERIES, generate_tickets
from .evaluation import evaluate, golden_from_records
from .pipeline import Analysis, analyze
from .report import build_report
from .sessions import SessionNotFound, SessionStore

__all__ = ["InsightService", "SessionNotFound"]

SAMPLE_NAME = "Sample: synthetic support tickets"


def _ts(value: Any) -> str | None:
    return None if value is None or pd.isna(value) else pd.Timestamp(value).isoformat()


class InsightService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = SessionStore(settings.session_dir)
        self._cache: OrderedDict[str, Analysis] = OrderedDict()
        self._lock = threading.Lock()

    # -- session lifecycle -------------------------------------------------
    def _get(self, session_id: str) -> Analysis:
        with self._lock:
            if session_id in self._cache:
                self._cache.move_to_end(session_id)
                return self._cache[session_id]
        analysis: Analysis = self.store.load(session_id)
        self._remember(analysis)
        return analysis

    def _remember(self, analysis: Analysis) -> None:
        with self._lock:
            self._cache[analysis.session_id] = analysis
            self._cache.move_to_end(analysis.session_id)
            while len(self._cache) > self.settings.session_cache_size:
                self._cache.popitem(last=False)

    def _persist(self, analysis: Analysis) -> None:
        self.store.save(analysis.session_id, analysis, self._meta(analysis))

    def _meta(self, a: Analysis) -> dict[str, Any]:
        return {
            "session_id": a.session_id,
            "name": a.name,
            "created_at": a.created_at,
            "rows": len(a.df),
            "topics": len([t for t in a.topics.topics if t.topic_id >= 0]),
            "incidents": len(a.incidents),
            "spikes": len(a.spikes),
            "has_time": a.has_time,
            "has_labels": "intent" in a.df.columns,
        }

    def create_session(
        self,
        source: Any,
        name: str = "untitled",
        text_col: str | None = None,
        time_col: str | None = None,
        n_topics: int | None = None,
    ) -> dict[str, Any]:
        sid = self.store.new_id()
        analysis = analyze(source, self.settings, sid, name, text_col, time_col, n_topics)
        self._persist(analysis)
        self._remember(analysis)
        return self.summary(sid)

    def create_sample_session(self, n: int = 3000) -> dict[str, Any]:
        return self.create_session(generate_tickets(n=n), name=SAMPLE_NAME)

    def list_sessions(self) -> list[dict[str, Any]]:
        return self.store.list()

    def delete(self, session_id: str) -> None:
        self.store.delete(session_id)
        with self._lock:
            self._cache.pop(session_id, None)

    # -- read models -------------------------------------------------------
    def summary(self, session_id: str) -> dict[str, Any]:
        a = self._get(session_id)
        ing = a.ingest
        return {
            **self._meta(a),
            "config": a.config,
            "ingest": {
                "rows_in": ing.rows_in,
                "rows_out": ing.rows_out,
                "dropped_empty": ing.dropped_empty,
                "truncated": ing.truncated,
                "exact_duplicates": ing.exact_duplicates,
                "text_col": ing.text_col,
                "time_col": ing.time_col,
                "pii_redactions": ing.pii_redactions,
            },
            "time_range": [_ts(a.df["created_at"].min()), _ts(a.df["created_at"].max())]
            if a.has_time
            else None,
        }

    def topics(self, session_id: str) -> list[dict[str, Any]]:
        a = self._get(session_id)
        total = len(a.df)
        return [
            {
                "topic_id": t.topic_id,
                "label": t.label,
                "size": t.size,
                "share": round(t.size / total, 4),
                "keywords": t.keywords,
                "examples": [a.df.at[i, "text"] for i in t.representative_idx],
            }
            for t in a.topics.topics
        ]

    def topic_tickets(
        self, session_id: str, topic_id: int, limit: int = 50
    ) -> list[dict[str, Any]]:
        a = self._get(session_id)
        rows = a.df[a.df["topic"] == topic_id].head(limit)
        return [
            {"id": r.id, "text": r.text, "created_at": _ts(r.created_at)}
            for r in rows.itertuples(index=False)
        ]

    def trends(self, session_id: str) -> dict[str, Any]:
        a = self._get(session_id)
        series: list[dict[str, Any]] = []
        if not a.volume.empty:
            long = a.volume.stack().reset_index()
            long.columns = ["period", "topic_id", "count"]
            series = [
                {"period": _ts(r.period), "topic_id": int(r.topic_id), "count": int(r.count)}
                for r in long.itertuples(index=False)
            ]
        labels = {t.topic_id: t.label for t in a.topics.topics}
        return {
            "freq": a.config.get("trend_freq"),
            "labels": {str(k): v for k, v in labels.items()},
            "series": series,
            "spikes": [
                {
                    "topic_id": s.topic_id,
                    "label": labels.get(s.topic_id, str(s.topic_id)),
                    "period": _ts(s.period),
                    "count": s.count,
                    "baseline": round(s.baseline, 2),
                    "zscore": round(s.zscore, 2),
                }
                for s in a.spikes
            ],
        }

    def incidents(self, session_id: str) -> list[dict[str, Any]]:
        a = self._get(session_id)
        return [
            {
                "incident_id": inc.incident_id,
                "size": inc.size,
                "group_size": inc.group_size,
                "topic_id": inc.topic_id,
                "sample_text": inc.sample_text,
                "start": _ts(inc.start),
                "end": _ts(inc.end),
                "ticket_ids": a.df.loc[inc.member_idx[:200], "id"].astype(str).tolist(),
            }
            for inc in a.incidents
        ]

    def search(
        self, session_id: str, query: str, k: int = 10, mode: str = "hybrid", rerank: bool = True
    ) -> list[dict[str, Any]]:
        a = self._get(session_id)
        hits = a.engine(self.settings).search(query, k=k, mode=mode, rerank=rerank)
        labels = {t.topic_id: t.label for t in a.topics.topics}
        out = []
        for rank, (i, score) in enumerate(hits, start=1):
            row = a.df.iloc[i]
            out.append(
                {
                    "rank": rank,
                    "id": str(row["id"]),
                    "score": round(score, 4),
                    "text": row["text"],
                    "topic_id": int(row["topic"]),
                    "topic": labels.get(int(row["topic"])),
                    "created_at": _ts(row["created_at"]),
                }
            )
        return out

    def evaluate(
        self,
        session_id: str,
        queries: list[dict[str, Any]] | None = None,
        k: int = 10,
        label_column: str = "intent",
    ) -> dict[str, Any]:
        a = self._get(session_id)
        if queries is None:
            if label_column not in a.df.columns:
                raise ValueError("No golden queries supplied and the dataset has no labels")
            queries = GOLDEN_QUERIES
        golden = golden_from_records(queries, a.df, label_column)
        report = evaluate(a.engine(self.settings), a.df["id"].astype(str).tolist(), golden, k=k)
        result = report.to_dict()
        a.last_eval = result["summary"]
        self._persist(a)
        return result

    # -- exports -----------------------------------------------------------
    def clusters_csv(self, session_id: str) -> str:
        a = self._get(session_id)
        labels = {t.topic_id: t.label for t in a.topics.topics}
        out = a.df.copy()
        out["topic_label"] = out["topic"].map(labels)
        return str(out.to_csv(index=False))

    def report_md(self, session_id: str) -> str:
        a = self._get(session_id)
        return build_report(a, a.last_eval)
