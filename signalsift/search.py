"""Retrieval: TF-IDF baseline, BM25, dense, and hybrid (reciprocal rank fusion) + reranking."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .embeddings import Embedder
from .index import build_index
from .lexical import BM25
from .rerank import Reranker

Mode = Literal["tfidf", "bm25", "dense", "hybrid"]
MODES: tuple[str, ...] = ("tfidf", "bm25", "dense", "hybrid")
RRF_K = 60
Hits = list[tuple[int, float]]


def top_k(scores: np.ndarray, k: int, positive_only: bool = False) -> Hits:
    k = min(k, len(scores))
    if k <= 0:
        return []
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.argsort(-scores[part], kind="stable")]
    return [(int(i), float(scores[i])) for i in order if not positive_only or scores[i] > 0]


def reciprocal_rank_fusion(*rankings: Hits, k: int = RRF_K) -> Hits:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


class SearchEngine:
    def __init__(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        embedder: Embedder,
        vector_backend: str = "numpy",
        reranker: Reranker | None = None,
        rerank_candidates: int = 50,
    ) -> None:
        self.texts = list(texts)
        self.embedder = embedder
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        self.index = build_index(vector_backend, embeddings.shape[1])
        self.index.add(embeddings)
        self.bm25 = BM25().fit(self.texts)
        self._tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english")
        self._tfidf_matrix = self._tfidf.fit_transform(self.texts)

    # -- first stage -------------------------------------------------------
    def _dense(self, query: str, n: int) -> Hits:
        scores, idx = self.index.search(self.embedder.encode([query]), n)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0], strict=True)]

    def _bm25(self, query: str, n: int) -> Hits:
        return top_k(self.bm25.get_scores(query), n, positive_only=True)

    def _tfidf_search(self, query: str, n: int) -> Hits:
        sims = (self._tfidf_matrix @ self._tfidf.transform([query]).T).toarray().ravel()
        return top_k(sims, n, positive_only=True)

    def retrieve(self, query: str, n: int, mode: str = "hybrid") -> Hits:
        if mode == "dense":
            return self._dense(query, n)
        if mode == "bm25":
            return self._bm25(query, n)
        if mode == "tfidf":
            return self._tfidf_search(query, n)
        if mode == "hybrid":
            pool = max(n, 50)
            return reciprocal_rank_fusion(self._dense(query, pool), self._bm25(query, pool))[:n]
        raise ValueError(f"Unknown search mode '{mode}'. Use one of {MODES}")

    # -- full pipeline -----------------------------------------------------
    def search(self, query: str, k: int = 10, mode: str = "hybrid", rerank: bool = False) -> Hits:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")
        use_rerank = rerank and self.reranker is not None
        hits = self.retrieve(query, max(k, self.rerank_candidates) if use_rerank else k, mode)
        if use_rerank and hits and self.reranker is not None:
            idxs = [i for i, _ in hits]
            prior = np.array([s for _, s in hits], dtype=np.float32)
            scores = self.reranker.rerank(query, [self.texts[i] for i in idxs], prior)
            order = np.argsort(-scores, kind="stable")
            hits = [(idxs[j], float(scores[j])) for j in order]
        return hits[:k]
