import numpy as np
import pytest

from signalsift.embeddings import LSAEmbedder, l2_normalize
from signalsift.index import NumpyIndex, build_index

CORPUS = [
    "app crashes on launch",
    "the app crashed when launching",
    "refund has not arrived",
    "still waiting for my refund",
    "flight was cancelled",
    "my flight got cancelled today",
]


def test_lsa_vectors_are_unit_norm_and_semantic() -> None:
    emb = LSAEmbedder(dims=4).fit(CORPUS)
    vecs = emb.encode(CORPUS)
    assert vecs.shape == (6, 4)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    q = emb.encode(["waiting on refund"])[0]
    assert int(np.argmax(vecs @ q)) in (2, 3)


def test_lsa_encode_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        LSAEmbedder().encode(["x"])


def test_lsa_handles_tiny_corpus() -> None:
    vecs = LSAEmbedder().fit(["one ticket", "another ticket"]).encode(["ticket"])
    assert vecs.shape[0] == 1


def test_l2_normalize_zero_row() -> None:
    out = l2_normalize(np.zeros((1, 3)))
    assert np.all(out == 0)


def test_numpy_index_returns_sorted_topk() -> None:
    rng = np.random.default_rng(0)
    vecs = l2_normalize(rng.normal(size=(50, 8)))
    index = NumpyIndex(8)
    index.add(vecs)
    scores, idx = index.search(vecs[7], 5)
    assert idx[0, 0] == 7
    assert np.all(np.diff(scores[0]) <= 1e-6)
    assert len(index) == 50


def test_index_rejects_wrong_dim() -> None:
    with pytest.raises(ValueError):
        NumpyIndex(4).add(np.ones((2, 3)))


def test_index_k_larger_than_size() -> None:
    index = NumpyIndex(2)
    index.add(np.eye(2, dtype=np.float32))
    _, idx = index.search(np.array([1.0, 0.0]), 10)
    assert idx.shape == (1, 2)


def test_unknown_backend() -> None:
    with pytest.raises(ValueError):
        build_index("annoy", 3)
