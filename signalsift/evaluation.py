"""Retrieval evaluation against golden queries with standard IR metrics.

Recall@k uses the capped definition ``hits / min(k, |relevant|)`` (as in BEIR), because
in ticket data a single intent often has hundreds of relevant tickets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .search import SearchEngine

# (display name, mode, rerank)
DEFAULT_CONFIGS: list[tuple[str, str, bool]] = [
    ("TF-IDF baseline", "tfidf", False),
    ("BM25", "bm25", False),
    ("Dense (bi-encoder)", "dense", False),
    ("Hybrid (RRF)", "hybrid", False),
    ("Hybrid + rerank", "hybrid", True),
]
METRICS = ("recall", "precision", "ndcg", "mrr", "hit_rate")


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / min(k, len(relevant))


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / k if k else 0.0


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    for rank, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def hit_rate_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


@dataclass
class GoldenQuery:
    query: str
    relevant_ids: set[str]


@dataclass
class EvalReport:
    k: int
    summary: pd.DataFrame
    per_query: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "summary": self.summary.to_dict(orient="records"),
            "per_query": self.per_query.to_dict(orient="records"),
        }


def golden_from_records(
    records: list[dict[str, Any]], df: pd.DataFrame, label_column: str = "intent"
) -> list[GoldenQuery]:
    """Accepts ``{"query", "relevant_ids": [...]}`` or ``{"query", "relevant_labels": [...]}``.

    Label-based relevance resolves to every ticket whose ``label_column`` matches.
    """
    out: list[GoldenQuery] = []
    for rec in records:
        query = str(rec.get("query", "")).strip()
        if not query:
            raise ValueError("Every golden query needs a non-empty 'query'")
        if rec.get("relevant_ids"):
            ids = {str(x) for x in rec["relevant_ids"]}
        elif rec.get("relevant_labels"):
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' is not in the dataset")
            mask = df[label_column].astype(str).isin([str(x) for x in rec["relevant_labels"]])
            ids = set(df.loc[mask, "id"].astype(str))
        else:
            raise ValueError(f"Golden query '{query}' needs relevant_ids or relevant_labels")
        out.append(GoldenQuery(query, ids))
    return out


def evaluate(
    engine: SearchEngine,
    doc_ids: list[str],
    golden: list[GoldenQuery],
    k: int = 10,
    configs: list[tuple[str, str, bool]] | None = None,
) -> EvalReport:
    configs = configs or DEFAULT_CONFIGS
    if engine.reranker is None:
        configs = [c for c in configs if not c[2]]
    rows: list[dict[str, Any]] = []
    for name, mode, rerank in configs:
        for g in golden:
            ranked = [doc_ids[i] for i, _ in engine.search(g.query, k=k, mode=mode, rerank=rerank)]
            rows.append(
                {
                    "system": name,
                    "query": g.query,
                    "n_relevant": len(g.relevant_ids),
                    "recall": recall_at_k(ranked, g.relevant_ids, k),
                    "precision": precision_at_k(ranked, g.relevant_ids, k),
                    "ndcg": ndcg_at_k(ranked, g.relevant_ids, k),
                    "mrr": mrr_at_k(ranked, g.relevant_ids, k),
                    "hit_rate": hit_rate_at_k(ranked, g.relevant_ids, k),
                }
            )
    per_query = pd.DataFrame(rows)
    order = [c[0] for c in configs]
    summary = (
        per_query.groupby("system", sort=False)[list(METRICS)]
        .mean()
        .reindex(order)
        .round(3)
        .reset_index()
    )
    return EvalReport(k=k, summary=summary, per_query=per_query.round(3))
