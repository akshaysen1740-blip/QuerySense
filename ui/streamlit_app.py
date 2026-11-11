"""Streamlit front end. Thin by design: every action goes through a ``Client``."""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd
import streamlit as st

from signalsift.client import Client, get_client
from signalsift.config import get_settings

MODES = {
    "Hybrid (BM25 + dense)": "hybrid",
    "Dense (semantic)": "dense",
    "BM25 (keyword)": "bm25",
    "TF-IDF (baseline)": "tfidf",
}


@st.cache_resource
def client() -> Client:
    return get_client(get_settings())


def _err(exc: Exception) -> None:
    st.error(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- sidebar
def sidebar() -> str | None:
    c = client()
    st.sidebar.title("SignalSift")
    st.sidebar.caption("Find what's breaking in your support queue.")

    st.sidebar.subheader("New analysis")
    if st.sidebar.button("Analyze sample dataset", width="stretch"):
        with st.spinner("Generating 3,000 sample tickets and analyzing them…"):
            try:
                st.session_state["sid"] = c.create_sample_session()["session_id"]
            except Exception as exc:  # noqa: BLE001 - surface any backend failure
                _err(exc)

    upload = st.sidebar.file_uploader("Or upload a CSV of tickets", type=["csv"])
    if upload is not None:
        data = upload.getvalue()
        try:
            preview = pd.read_csv(io.BytesIO(data), nrows=50)
        except Exception as exc:  # noqa: BLE001
            _err(exc)
            preview = None
        if preview is not None:
            cols = list(preview.columns)
            text_col = st.sidebar.selectbox("Text column", cols, index=_guess(cols, "text"))
            time_options = ["(no timestamps)", *cols]
            time_col = st.sidebar.selectbox(
                "Timestamp column", time_options, index=_guess(time_options, "time")
            )
            n_topics = st.sidebar.number_input("Topics (0 = automatic)", 0, 200, 0)
            if st.sidebar.button("Analyze upload", type="primary", width="stretch"):
                with st.spinner("Redacting, embedding and clustering…"):
                    try:
                        res = c.create_session(
                            data,
                            upload.name,
                            text_col,
                            None if time_col == "(no timestamps)" else time_col,
                            int(n_topics) or None,
                        )
                        st.session_state["sid"] = res["session_id"]
                    except Exception as exc:  # noqa: BLE001
                        _err(exc)

    st.sidebar.subheader("Saved analyses")
    try:
        sessions = c.list_sessions()
    except Exception as exc:  # noqa: BLE001
        _err(exc)
        return None
    if not sessions:
        st.sidebar.caption("Nothing saved yet.")
        return None
    ids = [s["session_id"] for s in sessions]
    current = st.session_state.get("sid")
    index = ids.index(current) if current in ids else 0
    picked = st.sidebar.selectbox(
        "Open",
        ids,
        index=index,
        format_func=lambda sid: next(
            f"{s['name']} · {s['rows']:,} rows" for s in sessions if s["session_id"] == sid
        ),
    )
    st.session_state["sid"] = picked
    if st.sidebar.button("Delete this analysis", width="stretch"):
        c.delete(picked)
        st.session_state.pop("sid", None)
        st.rerun()
    return str(picked)


def _guess(options: list[str], kind: str) -> int:
    hints = {
        "text": ("text", "body", "message", "description", "content"),
        "time": ("created", "time", "date"),
    }[kind]
    for i, opt in enumerate(options):
        if any(h in str(opt).lower() for h in hints):
            return i
    return 0


# --------------------------------------------------------------------------- tabs
def overview(sid: str) -> None:
    c = client()
    s = c.summary(sid)
    cols = st.columns(5)
    cols[0].metric("Tickets", f"{s['rows']:,}")
    cols[1].metric("Topics", s["topics"])
    cols[2].metric("Incidents", s["incidents"])
    cols[3].metric("Volume spikes", s["spikes"])
    cols[4].metric("PII spans redacted", sum(s["ingest"]["pii_redactions"].values()))

    if s["incidents"]:
        top = c.incidents(sid)[0]
        st.warning(
            f"**Largest incident:** {top['size']} near-identical tickets between "
            f"{_short(top['start'])} and {_short(top['end'])}. Example: “{top['sample_text']}”"
        )

    topics = pd.DataFrame(c.topics(sid))
    st.subheader("Where the volume is")
    chart = topics[topics.topic_id >= 0].head(15).set_index("label")["size"]
    st.bar_chart(chart, horizontal=True)

    with st.expander("Ingestion details"):
        st.json(s["ingest"])
        st.json(s["config"])


def topics_tab(sid: str) -> None:
    c = client()
    topics = c.topics(sid)
    frame = pd.DataFrame(topics)
    frame["keywords"] = frame["keywords"].map(", ".join)
    st.dataframe(
        frame[["topic_id", "label", "size", "share", "keywords"]],
        hide_index=True,
        width="stretch",
        column_config={"share": st.column_config.ProgressColumn("share", min_value=0, max_value=1)},
    )
    labels = {t["topic_id"]: f"{t['topic_id']}: {t['label']}" for t in topics}
    tid = st.selectbox("Read tickets from a topic", list(labels), format_func=labels.get)
    chosen = next(t for t in topics if t["topic_id"] == tid)
    st.markdown("**Most representative**")
    for ex in chosen["examples"]:
        st.markdown(f"> {ex}")
    st.dataframe(pd.DataFrame(c.topic_tickets(sid, tid)), hide_index=True, width="stretch")


def search_tab(sid: str) -> None:
    c = client()
    with st.form("search"):
        query = st.text_input("Describe the problem", placeholder="app crashes after the update")
        left, mid, right = st.columns([2, 1, 1])
        mode = left.radio("Retriever", list(MODES), horizontal=True)
        k = mid.slider("Results", 5, 50, 10)
        rerank = right.checkbox("Rerank", value=True)
        submitted = st.form_submit_button("Search", type="primary")
    if submitted and query.strip():
        try:
            hits = c.search(sid, query, k, MODES[mode], rerank)
        except Exception as exc:  # noqa: BLE001
            _err(exc)
            return
        if not hits:
            st.info("No tickets matched. Try describing the symptom in different words.")
            return
        st.dataframe(
            pd.DataFrame(hits)[["rank", "score", "text", "topic", "created_at", "id"]],
            hide_index=True,
            width="stretch",
        )


def trends_tab(sid: str) -> None:
    c = client()
    trends = c.trends(sid)
    if not trends["series"]:
        st.info("This dataset has no timestamps, so trends and incident windows are unavailable.")
    else:
        series = pd.DataFrame(trends["series"])
        labels = {int(k): v for k, v in trends["labels"].items()}
        series["topic"] = series["topic_id"].map(lambda t: f"{t}: {labels.get(t, t)}")
        spiking = list(dict.fromkeys(f"{s['topic_id']}: {s['label']}" for s in trends["spikes"]))
        options = sorted(series["topic"].unique(), key=lambda s: int(s.split(":")[0]))
        chosen = st.multiselect("Topics", options, default=spiking[:4] or options[:4])
        pivot = series[series.topic.isin(chosen)].pivot(
            index="period", columns="topic", values="count"
        )
        st.line_chart(pivot)
        st.subheader("Volume spikes")
        if trends["spikes"]:
            st.dataframe(pd.DataFrame(trends["spikes"]), hide_index=True, width="stretch")
        else:
            st.caption("No topic exceeded its trailing baseline.")

    st.subheader("Duplicate-burst incidents")
    incidents = c.incidents(sid)
    if not incidents:
        st.caption("No bursts of near-identical tickets found.")
    for inc in incidents:
        title = f"#{inc['incident_id']} · {inc['size']} tickets · {_short(inc['start'])}"
        with st.expander(title, expanded=inc["incident_id"] == 0):
            st.markdown(f"> {inc['sample_text']}")
            st.caption(
                f"Window {_short(inc['start'])} to {_short(inc['end'])}; "
                f"{inc['group_size']} similar tickets in total; topic {inc['topic_id']}."
            )
            st.code(", ".join(inc["ticket_ids"][:50]), language=None)


def evaluation_tab(sid: str) -> None:
    c = client()
    summary = c.summary(sid)
    st.markdown(
        "Compare retrievers on golden queries: Recall@k (capped), Precision@k, nDCG@k, "
        "MRR@k and HitRate@k."
    )
    queries: list[dict[str, Any]] | None = None
    if not summary["has_labels"]:
        st.caption('Upload golden queries as JSON: `[{"query": "...", "relevant_ids": ["..."]}]`')
        golden = st.file_uploader("Golden queries", type=["json"])
        if golden is None:
            return
        queries = json.loads(golden.getvalue())
    else:
        st.caption("Using the bundled 25-query golden set for the sample dataset.")
    k = st.slider("k", 5, 50, 10, key="eval_k")
    if st.button("Run evaluation", type="primary"):
        with st.spinner("Running every retriever on every query…"):
            try:
                st.session_state[f"eval_{sid}"] = c.evaluate(sid, queries, k)
            except Exception as exc:  # noqa: BLE001
                _err(exc)
    result = st.session_state.get(f"eval_{sid}")
    if not result:
        return
    table = pd.DataFrame(result["summary"]).set_index("system")
    st.dataframe(table.style.highlight_max(axis=0, color="#fde68a"), width="stretch")
    st.bar_chart(table[["recall", "ndcg", "mrr"]])
    st.subheader("Error analysis")
    per_query = pd.DataFrame(result["per_query"])
    system = st.selectbox("System", list(table.index), index=len(table.index) - 1)
    view = per_query[per_query.system == system].sort_values("ndcg")
    st.dataframe(view, hide_index=True, width="stretch")


def export_tab(sid: str) -> None:
    c = client()
    st.download_button(
        "Download clusters CSV", c.clusters_csv(sid), f"clusters_{sid}.csv", "text/csv"
    )
    report = c.report_md(sid)
    st.download_button("Download Markdown report", report, f"report_{sid}.md", "text/markdown")
    st.divider()
    st.markdown(report)


def _short(ts: str | None) -> str:
    return pd.Timestamp(ts).strftime("%b %d, %H:%M") if ts else "n/a"


def main() -> None:
    st.set_page_config(page_title="SignalSift", page_icon="🔎", layout="wide")
    sid = sidebar()
    if not sid:
        st.title("SignalSift")
        st.markdown(
            "Turn a pile of support tickets into topics, searchable history, trend spikes "
            "and incident alerts.\n\nStart with **Analyze sample dataset** in the sidebar, "
            "or upload a CSV with one ticket per row."
        )
        return
    try:
        name = client().summary(sid)["name"]
    except Exception as exc:  # noqa: BLE001
        _err(exc)
        return
    st.title(name)
    tabs = st.tabs(["Overview", "Topics", "Search", "Trends & incidents", "Evaluation", "Export"])
    renderers = [overview, topics_tab, search_tab, trends_tab, evaluation_tab, export_tab]
    for tab, render in zip(tabs, renderers, strict=True):
        with tab:
            render(sid)
