"""Okapi BM25 over a sparse term-frequency matrix."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._vectorizer = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b\w\w+\b")
        self._tf: sparse.csc_matrix | None = None
        self._idf: np.ndarray | None = None
        self._norm: np.ndarray | None = None

    def fit(self, texts: list[str]) -> BM25:
        tf = self._vectorizer.fit_transform(texts).tocsc().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.diff(tf.indptr)  # documents containing each term (csc column counts)
        self._idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg = doc_len.mean() if n_docs else 1.0
        self._norm = (self.k1 * (1 - self.b + self.b * doc_len / max(avg, 1e-9))).astype(np.float32)
        self._tf = tf
        return self

    def get_scores(self, query: str) -> np.ndarray:
        if self._tf is None or self._idf is None or self._norm is None:
            raise RuntimeError("BM25.get_scores called before fit")
        scores = np.zeros(self._tf.shape[0], dtype=np.float32)
        vocab = self._vectorizer.vocabulary_
        analyzer = self._vectorizer.build_analyzer()
        for term in set(analyzer(query)):
            j = vocab.get(term)
            if j is None:
                continue
            start, end = self._tf.indptr[j], self._tf.indptr[j + 1]
            rows = self._tf.indices[start:end]
            tf = self._tf.data[start:end]
            scores[rows] += self._idf[j] * tf * (self.k1 + 1) / (tf + self._norm[rows])
        return scores
