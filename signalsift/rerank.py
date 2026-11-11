"""Second-stage rerankers.

* ``CrossEncoderReranker`` — scores (query, ticket) pairs jointly with a pretrained
  cross-encoder. Best quality, needs the ``ml`` extra and a model download.
* ``HeuristicReranker`` — a no-download fallback that blends the first-stage score with
  query-term coverage and bigram overlap (with light suffix stemming). It exists so the
  rerank stage is exercised offline and in CI; it is *not* a learned model.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .config import Settings

_TOKEN = re.compile(r"[a-z0-9]+")
_SUFFIXES = ("ing", "ed", "es", "ly", "s")


def _stem(token: str) -> str:
    for suf in _SUFFIXES:
        if len(token) > len(suf) + 3 and token.endswith(suf):
            return token[: -len(suf)]
    return token


def _terms(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def _minmax(x: np.ndarray) -> np.ndarray:
    span = float(x.max() - x.min()) if len(x) else 0.0
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, texts: list[str], prior: np.ndarray | None = None
    ) -> np.ndarray: ...


class HeuristicReranker:
    name = "heuristic"

    def __init__(self, alpha: float = 0.6) -> None:
        self.alpha = alpha

    def rerank(self, query: str, texts: list[str], prior: np.ndarray | None = None) -> np.ndarray:
        q_terms = _terms(query)
        q_set = set(q_terms)
        q_bigrams = set(zip(q_terms, q_terms[1:], strict=False))
        lexical = np.zeros(len(texts), dtype=np.float32)
        if q_set:
            for i, text in enumerate(texts):
                d_terms = _terms(text)
                coverage = len(q_set & set(d_terms)) / len(q_set)
                bigram = (
                    len(q_bigrams & set(zip(d_terms, d_terms[1:], strict=False))) / len(q_bigrams)
                    if q_bigrams
                    else 0.0
                )
                lexical[i] = 0.8 * coverage + 0.2 * bigram
        if prior is None:
            return lexical
        return self.alpha * _minmax(np.asarray(prior, np.float32)) + (1 - self.alpha) * lexical


class CrossEncoderReranker:  # pragma: no cover - needs model download
    name = "cross-encoder"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None

    def rerank(self, query: str, texts: list[str], prior: np.ndarray | None = None) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("Install the ml extra: pip install -e '.[ml]'") from exc
            self._model = CrossEncoder(self.model_name)
        return np.asarray(self._model.predict([(query, t) for t in texts]), dtype=np.float32)


def build_reranker(settings: Settings) -> Reranker | None:
    if settings.reranker == "none":
        return None
    if settings.reranker == "cross-encoder":
        return CrossEncoderReranker(settings.cross_encoder_model)
    return HeuristicReranker()
