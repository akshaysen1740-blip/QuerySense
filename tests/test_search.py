import numpy as np
import pytest

from signalsift.embeddings import LSAEmbedder
from signalsift.lexical import BM25
from signalsift.rerank import HeuristicReranker
from signalsift.search import SearchEngine, reciprocal_rank_fusion, top_k

DOCS = [
    "app crashes on launch after update",
    "refund not received for my order",
    "flight cancelled no rebooking offered",
    "card declined at checkout",
    "app keeps crashing on startup",
    "luggage lost at the airport",
]


@pytest.fixture(scope="module")
def engine() -> SearchEngine:
    emb = LSAEmbedder(dims=4).fit(DOCS)
    return SearchEngine(
        DOCS, emb.encode(DOCS), emb, reranker=HeuristicReranker(), rerank_candidates=10
    )


def test_bm25_prefers_matching_doc() -> None:
    scores = BM25().fit(DOCS).get_scores("card declined")
    assert int(np.argmax(scores)) == 3
    assert scores[0] == 0


def test_bm25_unknown_terms_score_zero() -> None:
    assert BM25().fit(DOCS).get_scores("zzzz qqqq").sum() == 0


def test_rrf_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion([(1, 0.9), (2, 0.8)], [(2, 5.0), (3, 4.0)])
    assert fused[0][0] == 2


def test_top_k_positive_only() -> None:
    assert top_k(np.array([0.0, 2.0, 1.0]), 3, positive_only=True) == [(1, 2.0), (2, 1.0)]


@pytest.mark.parametrize("mode", ["tfidf", "bm25", "dense", "hybrid"])
def test_every_mode_finds_refund(engine: SearchEngine, mode: str) -> None:
    hits = engine.search("refund", k=3, mode=mode)
    assert hits and hits[0][0] == 1


def test_rerank_keeps_k(engine: SearchEngine) -> None:
    hits = engine.search("app crashing", k=2, mode="hybrid", rerank=True)
    assert len(hits) == 2 and {i for i, _ in hits} <= {0, 4, 1, 2, 3, 5}
    assert hits[0][0] in (0, 4)


def test_empty_query_rejected(engine: SearchEngine) -> None:
    with pytest.raises(ValueError):
        engine.search("   ")


def test_unknown_mode_rejected(engine: SearchEngine) -> None:
    with pytest.raises(ValueError):
        engine.search("refund", mode="magic")


def test_heuristic_reranker_scores_coverage() -> None:
    scores = HeuristicReranker().rerank(
        "lost luggage", ["luggage lost at airport", "card declined"]
    )
    assert scores[0] > scores[1]
