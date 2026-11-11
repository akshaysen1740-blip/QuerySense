import pandas as pd
import pytest

from signalsift.evaluation import (
    evaluate,
    golden_from_records,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from signalsift.pipeline import Analysis


def test_metrics_on_known_ranking() -> None:
    ranked = ["a", "x", "b", "y"]
    rel = {"a", "b", "c"}
    assert recall_at_k(ranked, rel, 4) == pytest.approx(2 / 3)
    assert precision_at_k(ranked, rel, 4) == 0.5
    assert mrr_at_k(["x", "a"], rel, 4) == 0.5
    assert hit_rate_at_k(["x", "y"], rel, 2) == 0.0
    assert ndcg_at_k(["a", "b", "c"], rel, 3) == pytest.approx(1.0)


def test_recall_is_capped_by_k() -> None:
    rel = {str(i) for i in range(100)}
    assert recall_at_k([str(i) for i in range(10)], rel, 10) == 1.0


def test_empty_relevant_set() -> None:
    assert recall_at_k(["a"], set(), 10) == 0.0 and ndcg_at_k(["a"], set(), 10) == 0.0


def test_golden_from_labels_and_ids() -> None:
    df = pd.DataFrame({"id": ["1", "2", "3"], "intent": ["a", "b", "a"]})
    golden = golden_from_records(
        [{"query": "q", "relevant_labels": ["a"]}, {"query": "r", "relevant_ids": [2]}], df
    )
    assert golden[0].relevant_ids == {"1", "3"} and golden[1].relevant_ids == {"2"}


def test_golden_requires_relevance() -> None:
    with pytest.raises(ValueError):
        golden_from_records([{"query": "q"}], pd.DataFrame({"id": []}))


def test_evaluate_beats_chance(analysis: Analysis, settings) -> None:
    from signalsift.datasets import GOLDEN_QUERIES

    golden = golden_from_records(GOLDEN_QUERIES, analysis.df)
    report = evaluate(analysis.engine(settings), analysis.df["id"].tolist(), golden, k=10)
    summary = report.summary.set_index("system")
    assert len(summary) == 5
    assert (summary["hit_rate"] >= 0.8).all()
    assert set(report.per_query["system"]) == set(summary.index)
