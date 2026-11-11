"""Topic discovery: KMeans on embeddings + class-based TF-IDF keywords (BERTopic-style).

``method="bertopic"`` delegates to BERTopic (HDBSCAN + UMAP) when the ``ml`` extra is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

from .text import STOP_WORDS


@dataclass
class Topic:
    topic_id: int
    size: int
    keywords: list[str]
    label: str
    representative_idx: list[int] = field(default_factory=list)


@dataclass
class TopicResult:
    labels: np.ndarray
    topics: list[Topic]
    method: str

    def get(self, topic_id: int) -> Topic:
        for t in self.topics:
            if t.topic_id == topic_id:
                return t
        raise KeyError(topic_id)


def choose_k(
    embeddings: np.ndarray, k_min: int = 4, k_max: int = 30, sample: int = 2000, seed: int = 42
) -> int:
    """Pick the number of clusters with the best cosine silhouette on a sample."""
    n = len(embeddings)
    upper = max(2, min(k_max, n // 10))
    lower = min(k_min, upper)
    if lower >= upper:
        return upper
    rng = np.random.default_rng(seed)
    x = embeddings[rng.choice(n, size=min(sample, n), replace=False)]
    best_k, best = lower, -1.0
    for k in range(lower, upper + 1):
        labels = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(x)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(x, labels, metric="cosine"))
        if score > best:
            best_k, best = k, score
    return best_k


def ctfidf_keywords(texts: list[str], labels: np.ndarray, top_n: int = 6) -> dict[int, list[str]]:
    """Class-based TF-IDF: treat every cluster as one document and score terms per cluster."""
    ids = sorted({int(x) for x in labels})
    docs = [" ".join(texts[i] for i in np.flatnonzero(labels == c)) for c in ids]
    cv = CountVectorizer(
        stop_words=STOP_WORDS, ngram_range=(1, 2), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
    )
    try:
        tf = cv.fit_transform(docs).astype(np.float64)
    except ValueError:  # empty vocabulary
        return {c: [] for c in ids}
    row_sums = np.asarray(tf.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    tf_norm = sparse.diags(1.0 / row_sums) @ tf
    avg_words = float(tf.sum()) / len(ids)
    freq = np.asarray(tf.sum(axis=0)).ravel()
    idf = np.log(1.0 + avg_words / np.maximum(freq, 1.0))
    scores = (tf_norm @ sparse.diags(idf)).toarray()
    vocab = cv.get_feature_names_out()

    out: dict[int, list[str]] = {}
    for row, c in enumerate(ids):
        chosen: list[str] = []
        covered: set[str] = set()
        for j in np.argsort(-scores[row]):
            if scores[row, j] <= 0 or len(chosen) >= top_n:
                break
            words = set(vocab[j].split())
            if words <= covered:
                continue
            chosen.append(str(vocab[j]))
            covered |= words
        out[c] = chosen
    return out


def _representatives(embeddings: np.ndarray, members: np.ndarray, n: int = 3) -> list[int]:
    centroid = embeddings[members].mean(axis=0)
    sims = embeddings[members] @ centroid
    return [int(members[i]) for i in np.argsort(-sims)[:n]]


def _build(
    texts: list[str], embeddings: np.ndarray, raw_labels: np.ndarray, method: str
) -> TopicResult:
    # Renumber so topic 0 is the largest (outliers, label -1, keep their id).
    ids, counts = np.unique(raw_labels[raw_labels >= 0], return_counts=True)
    mapping = {int(old): new for new, old in enumerate(ids[np.argsort(-counts, kind="stable")])}
    mapping[-1] = -1
    labels = np.array([mapping[int(x)] for x in raw_labels], dtype=np.int64)
    keywords = ctfidf_keywords(texts, labels)
    topics = []
    for tid in sorted(set(labels.tolist()), key=lambda t: (t < 0, t)):
        members = np.flatnonzero(labels == tid)
        kw = keywords.get(tid, [])
        label = "Outliers" if tid < 0 else (" / ".join(kw[:3]) or f"Topic {tid}")
        topics.append(Topic(tid, len(members), kw, label, _representatives(embeddings, members)))
    return TopicResult(labels=labels, topics=topics, method=method)


def discover_topics(
    texts: list[str],
    embeddings: np.ndarray,
    n_topics: int = 0,
    method: str = "kmeans",
    seed: int = 42,
) -> TopicResult:
    n = len(texts)
    if n == 0:
        raise ValueError("No texts to cluster")
    if method == "bertopic":  # pragma: no cover - optional dependency
        return _bertopic(texts, embeddings, n_topics)
    if n < 4:
        return _build(texts, embeddings, np.zeros(n, dtype=np.int64), "single")
    k = n_topics or choose_k(embeddings, seed=seed)
    k = max(1, min(k, n))
    raw = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(embeddings)
    return _build(texts, embeddings, raw, "kmeans")


def _bertopic(
    texts: list[str], embeddings: np.ndarray, n_topics: int
) -> TopicResult:  # pragma: no cover
    try:
        from bertopic import BERTopic
    except ImportError as exc:
        raise RuntimeError("Install the ml extra for BERTopic: pip install -e '.[ml]'") from exc
    model = BERTopic(nr_topics=n_topics or "auto", calculate_probabilities=False, verbose=False)
    raw, _ = model.fit_transform(texts, embeddings)
    return _build(texts, embeddings, np.asarray(raw, dtype=np.int64), "bertopic")
