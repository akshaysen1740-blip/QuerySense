import numpy as np
import pandas as pd

from signalsift.clustering import choose_k, ctfidf_keywords, discover_topics
from signalsift.pipeline import Analysis
from signalsift.trends import detect_spikes, near_duplicate_groups, topic_volume


def test_topics_are_sorted_by_size_and_labelled(analysis: Analysis) -> None:
    sizes = [t.size for t in analysis.topics.topics if t.topic_id >= 0]
    assert sizes == sorted(sizes, reverse=True)
    assert sum(sizes) == len(analysis.df)
    assert all(t.keywords and t.label for t in analysis.topics.topics)


def test_topics_far_better_than_chance(analysis: Analysis) -> None:
    # 22 balanced intents -> a random assignment scores ~0.05 purity / ~0 ARI. LSA cannot group
    # lexically disjoint paraphrases, so we only assert a solid margin over chance here.
    from sklearn.metrics import adjusted_rand_score

    df = analysis.df
    purity = df.groupby("topic")["intent"].agg(lambda s: s.value_counts().iloc[0] / len(s))
    weighted = (purity * df["topic"].value_counts(normalize=True)).sum()
    assert weighted > 0.45
    assert adjusted_rand_score(df["intent"], df["topic"]) > 0.2


def test_keywords_skip_redaction_placeholders() -> None:
    texts = ["refund [EMAIL] late", "refund missing [EMAIL]", "crash on launch", "app crash"]
    kws = ctfidf_keywords(texts, np.array([0, 0, 1, 1]))
    assert "email" not in kws[0] and "refund" in kws[0]


def test_fixed_topic_count() -> None:
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(40, 5)).astype(np.float32)
    result = discover_topics([f"doc {i}" for i in range(40)], emb, n_topics=3)
    assert len({int(x) for x in result.labels}) == 3


def test_choose_k_small_input() -> None:
    assert choose_k(np.random.default_rng(0).normal(size=(12, 3))) == 2


def test_topic_volume_fills_gaps() -> None:
    ts = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-03"]))
    vol = topic_volume(ts, np.array([0, 0, 1]), "D")
    assert vol.shape == (3, 2) and vol.loc["2026-01-02"].sum() == 0


def test_topic_volume_without_time_is_empty() -> None:
    assert topic_volume(pd.Series([pd.NaT, pd.NaT]), np.array([0, 1])).empty


def test_spike_detected_on_synthetic_series() -> None:
    idx = pd.date_range("2026-01-01", periods=20, freq="D")
    vol = pd.DataFrame({0: [3] * 19 + [30], 1: [3] * 20}, index=idx)
    spikes = detect_spikes(vol, z_threshold=3, min_count=5, window=7)
    assert len(spikes) == 1 and spikes[0].topic_id == 0 and spikes[0].count == 30


def test_planted_payment_spike_is_found(analysis: Analysis) -> None:
    labels = analysis.df.groupby("topic")["intent"].agg(lambda s: s.value_counts().index[0])
    in_spikes = any(labels[s.topic_id] == "payment_declined" for s in analysis.spikes)
    in_incidents = any(
        (analysis.df.loc[i.member_idx, "intent"] == "payment_declined").mean() > 0.9
        for i in analysis.incidents
    )
    assert in_spikes or in_incidents


def test_near_duplicate_groups() -> None:
    emb = np.array([[1, 0], [0.99, 0.141], [0, 1]], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    comp = near_duplicate_groups(emb, 0.95)
    assert comp[0] == comp[1] != comp[2]


def test_planted_incident_is_top_and_pure(analysis: Analysis) -> None:
    assert analysis.incidents
    top = analysis.incidents[0]
    intents = analysis.df.loc[top.member_idx, "intent"]
    assert (intents == "crash_after_update").mean() > 0.9
    assert top.end - top.start <= pd.Timedelta(hours=48)
