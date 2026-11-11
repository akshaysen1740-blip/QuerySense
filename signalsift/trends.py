"""Trend spikes per topic and near-duplicate burst incidents."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components


@dataclass
class Spike:
    topic_id: int
    period: pd.Timestamp
    count: int
    baseline: float
    zscore: float


@dataclass
class Incident:
    incident_id: int
    size: int
    group_size: int
    topic_id: int
    sample_text: str
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    member_idx: list[int] = field(default_factory=list)


def topic_volume(created_at: pd.Series, labels: np.ndarray, freq: str = "D") -> pd.DataFrame:
    """Ticket counts per period (rows) and topic (columns), with empty periods filled with 0."""
    frame = pd.DataFrame({"t": pd.to_datetime(created_at), "topic": labels}).dropna(subset=["t"])
    if frame.empty:
        return pd.DataFrame()
    period_freq = "h" if freq == "h" else freq
    frame["period"] = frame["t"].dt.to_period(period_freq)
    counts = frame.groupby(["period", "topic"]).size().unstack(fill_value=0)
    full = pd.period_range(frame["period"].min(), frame["period"].max(), freq=period_freq)
    counts = counts.reindex(full, fill_value=0)
    counts.index = full.to_timestamp()
    counts.columns = [int(c) for c in counts.columns]
    return counts


def detect_spikes(
    volume: pd.DataFrame, z_threshold: float = 3.0, min_count: int = 5, window: int = 7
) -> list[Spike]:
    """Flag periods whose count is far above the trailing baseline.

    The denominator is ``max(rolling std, sqrt(baseline))`` — a Poisson floor so that very
    quiet topics don't produce huge z-scores from a single extra ticket.
    """
    spikes: list[Spike] = []
    if volume.empty:
        return spikes
    min_periods = max(3, window // 2)
    for topic in volume.columns:
        s = volume[topic].astype(float)
        prev = s.shift(1)
        base = prev.rolling(window, min_periods=min_periods).mean()
        sd = prev.rolling(window, min_periods=min_periods).std().fillna(0.0)
        denom = np.maximum(sd, np.sqrt(np.maximum(base, 1.0)))
        z = (s - base) / denom
        hit = (z >= z_threshold) & (s >= min_count)
        for period in s.index[hit.fillna(False).to_numpy()]:
            spikes.append(
                Spike(int(topic), period, int(s[period]), float(base[period]), float(z[period]))
            )
    return sorted(spikes, key=lambda sp: -sp.zscore)


def near_duplicate_groups(embeddings: np.ndarray, threshold: float, chunk: int = 512) -> np.ndarray:
    """Connected components of the graph whose edges are pairs with cosine >= threshold."""
    n = len(embeddings)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for start in range(0, n, chunk):
        sims = embeddings[start : start + chunk] @ embeddings.T
        r, c = np.nonzero(sims >= threshold)
        rows.append(r + start)
        cols.append(c)
    r_all = np.concatenate(rows) if rows else np.empty(0, np.int64)
    c_all = np.concatenate(cols) if cols else np.empty(0, np.int64)
    graph = sparse.coo_matrix((np.ones(len(r_all), np.int8), (r_all, c_all)), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    return comp


def _densest_window(times: np.ndarray, window: np.timedelta64) -> tuple[int, int]:
    """Return [lo, hi) bounds of the sorted-times window containing the most points."""
    best, lo_best, j = 0, 0, 0
    for i in range(len(times)):
        while times[i] - times[j] > window:
            j += 1
        if i - j + 1 > best:
            best, lo_best = i - j + 1, j
    return lo_best, lo_best + best


def _merge(incidents: list[Incident], embeddings: np.ndarray, threshold: float) -> list[Incident]:
    """Merge bursts that overlap in time and are either similar (centroid cosine >=
    ``threshold``) or belong to the same topic cluster.

    Paraphrases of the same complaint often land in separate near-duplicate groups.
    """
    if len(incidents) < 2:
        return incidents
    cents = np.stack([embeddings[inc.member_idx].mean(axis=0) for inc in incidents])
    cents /= np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-9)
    parent = list(range(len(incidents)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sims = cents @ cents.T
    for i in range(len(incidents)):
        for j in range(i + 1, len(incidents)):
            a, b = incidents[i], incidents[j]
            overlap = (
                a.start is None or b.start is None or (a.start <= b.end and b.start <= a.end)  # type: ignore[operator]
            )
            if overlap and (sims[i, j] >= threshold or a.topic_id == b.topic_id):
                parent[find(j)] = find(i)
    merged: dict[int, Incident] = {}
    for i, inc in enumerate(incidents):
        root = find(i)
        if root not in merged:
            merged[root] = inc
            continue
        m = merged[root]
        m.member_idx = sorted(set(m.member_idx) | set(inc.member_idx))
        m.size = len(m.member_idx)
        m.group_size += inc.group_size
        if inc.start is not None and m.start is not None:
            m.start, m.end = min(m.start, inc.start), max(m.end, inc.end)  # type: ignore[type-var]
    return list(merged.values())


def detect_incidents(
    embeddings: np.ndarray,
    created_at: pd.Series,
    labels: np.ndarray,
    texts: list[str],
    threshold: float = 0.85,
    min_size: int = 8,
    window_hours: int = 48,
    burst_ratio: float = 4.0,
    merge_threshold: float = 0.6,
) -> list[Incident]:
    """Find bursts of near-identical tickets.

    1. Group tickets into near-duplicate groups (cosine >= ``threshold``).
    2. Inside each group find the densest ``window_hours`` window.
    3. It is a burst when that window holds >= ``min_size`` tickets *and* at least
       ``burst_ratio`` x what the group's rate *outside* the window predicts (a group that
       only exists inside the window has an expected count of ~0, so it always qualifies).
    4. Bursts that are paraphrases of each other and overlap in time are merged.
    Without timestamps, large duplicate groups are reported instead.
    """
    comp = near_duplicate_groups(embeddings, threshold)
    times = pd.to_datetime(created_at).to_numpy(dtype="datetime64[ns]")
    valid = ~np.isnat(times)
    has_time = bool(valid.any())
    window = np.timedelta64(window_hours, "h")
    total_h = (
        float((times[valid].max() - times[valid].min()) / np.timedelta64(1, "h"))
        if has_time
        else 0.0
    )
    found: list[Incident] = []

    for gid in np.unique(comp):
        members = np.flatnonzero(comp == gid)
        if len(members) < min_size:
            continue
        start = end = None
        burst = members
        if has_time:
            timed = members[valid[members]]
            if len(timed) < min_size:
                continue
            timed = timed[np.argsort(times[timed], kind="stable")]
            t = times[timed]
            lo, hi = _densest_window(t, window)
            inside = hi - lo
            outside_rate = (len(timed) - inside) / max(total_h - window_hours, float(window_hours))
            expected = outside_rate * window_hours
            if inside < min_size or inside < burst_ratio * max(expected, 1.0):
                continue
            burst = timed[lo:hi]
            start, end = pd.Timestamp(t[lo]), pd.Timestamp(t[hi - 1])
        topic = int(pd.Series(labels[burst]).mode().iloc[0])
        found.append(Incident(0, len(burst), len(members), topic, "", start, end, burst.tolist()))

    found = _merge(found, embeddings, merge_threshold)
    for inc in found:
        idx = np.asarray(inc.member_idx)
        sub = embeddings[idx[:500]]
        inc.sample_text = texts[int(idx[int(np.argmax((sub @ sub.T).mean(axis=1)))])]
        inc.topic_id = int(pd.Series(labels[idx]).mode().iloc[0])
    found.sort(key=lambda inc: -inc.size)
    for i, inc in enumerate(found):
        inc.incident_id = i
    return found
