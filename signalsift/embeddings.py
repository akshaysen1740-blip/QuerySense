"""Pluggable text embedders.

* ``LSAEmbedder``  — TF-IDF (1-2 grams) + TruncatedSVD. Fitted on the corpus, no downloads,
  runs anywhere. This is the default so the whole app works offline and in CI.
* ``SentenceTransformerEmbedder`` — pretrained bi-encoder (install the ``ml`` extra).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Settings
from .text import STOP_WORDS


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> Embedder: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


class LSAEmbedder:
    name = "lsa"

    def __init__(self, dims: int = 128, random_state: int = 42) -> None:
        self.dims = dims
        self.random_state = random_state
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> LSAEmbedder:
        if not texts:
            raise ValueError("Cannot fit an embedder on an empty corpus")
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words=STOP_WORDS,
            min_df=2 if len(texts) >= 200 else 1,
            max_df=0.95 if len(texts) >= 50 else 1.0,
            max_features=50_000,
        )
        matrix = self._vectorizer.fit_transform(texts)
        k = min(self.dims, matrix.shape[1] - 1, matrix.shape[0] - 1)
        self._svd = (
            TruncatedSVD(n_components=k, random_state=self.random_state).fit(matrix)
            if k >= 2
            else None
        )
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._vectorizer is None:
            raise RuntimeError("LSAEmbedder.encode called before fit")
        matrix = self._vectorizer.transform(texts)
        dense = self._svd.transform(matrix) if self._svd is not None else matrix.toarray()
        return l2_normalize(np.asarray(dense, dtype=np.float32))


class SentenceTransformerEmbedder:  # pragma: no cover - needs model download
    name = "sbert"

    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install the ml extra: pip install -e '.[ml]'") from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: list[str]) -> SentenceTransformerEmbedder:
        self._load()
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._load().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_model"] = None  # never pickle model weights into a session
        return state


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "sbert":
        return SentenceTransformerEmbedder(settings.sbert_model)
    return LSAEmbedder(dims=settings.lsa_dims)
