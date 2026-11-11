"""Pluggable inner-product vector indexes (vectors are expected to be L2-normalised)."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class VectorIndex(Protocol):
    dim: int

    def add(self, vectors: np.ndarray) -> None: ...

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...

    def __len__(self) -> int: ...


def _check(vectors: np.ndarray, dim: int) -> np.ndarray:
    arr = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    if arr.shape[1] != dim:
        raise ValueError(f"Expected vectors of dim {dim}, got {arr.shape[1]}")
    return arr


class NumpyIndex:
    """Exact brute-force search. Fast enough for tens of thousands of tickets."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors = np.empty((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self._vectors = np.vstack([self._vectors, _check(vectors, self.dim)])

    def __len__(self) -> int:
        return int(self._vectors.shape[0])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = _check(queries, self.dim)
        k = min(k, len(self))
        if k <= 0:
            return np.empty((len(q), 0), np.float32), np.empty((len(q), 0), np.int64)
        sims = q @ self._vectors.T
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1, kind="stable")
        idx = np.take_along_axis(part, order, axis=1)
        return np.take_along_axis(sims, idx, axis=1), idx.astype(np.int64)


class FaissIndex:  # pragma: no cover - optional dependency
    def __init__(self, dim: int) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Install faiss-cpu (pip install -e '.[ml]')") from exc
        self.dim = dim
        self._index: Any = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        self._index.add(_check(vectors, self.dim))

    def __len__(self) -> int:
        return int(self._index.ntotal)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores, idx = self._index.search(_check(queries, self.dim), min(k, len(self)))
        return scores, idx.astype(np.int64)


def build_index(backend: str, dim: int) -> VectorIndex:
    if backend == "faiss":
        return FaissIndex(dim)
    if backend == "numpy":
        return NumpyIndex(dim)
    raise ValueError(f"Unknown vector backend: {backend}")
