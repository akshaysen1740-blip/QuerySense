"""Deterministic Markdown executive report built from an analysis."""

from __future__ import annotations

from typing import Any

from .pipeline import Analysis


def _fmt_ts(ts: Any) -> str:
    return ts.strftime("%Y-%m-%d %H:%M") if ts is not None else "n/a"


def build_report(analysis: Analysis, eval_summary: list[dict[str, Any]] | None = None) -> str:
    df = analysis.df
    total = len(df)
    lines = [
        f"# SignalSift report: {analysis.name}",
        "",
        f"_Generated from session `{analysis.session_id}` ({analysis.created_at})._",
        "",
        "## Executive summary",
        "",
    ]
    top = [t for t in analysis.topics.topics if t.topic_id >= 0][:3]
    if top:
        parts = [f"**{t.label}** ({t.size / total:.0%})" for t in top]
        lines.append(f"- {total:,} tickets analysed; the largest themes are {', '.join(parts)}.")
    if analysis.has_time:
        span = f"{df['created_at'].min():%Y-%m-%d} to {df['created_at'].max():%Y-%m-%d}"
        lines.append(f"- Time range: {span}.")
    if analysis.incidents:
        inc = analysis.incidents[0]
        lines.append(
            f"- **{len(analysis.incidents)} incident(s) detected.** Largest: {inc.size} "
            f"near-identical tickets between {_fmt_ts(inc.start)} and {_fmt_ts(inc.end)}, "
            f'e.g. "{inc.sample_text}".'
        )
    else:
        lines.append("- No duplicate-burst incidents detected.")
    if analysis.spikes:
        sp = analysis.spikes[0]
        label = analysis.topics.get(sp.topic_id).label
        lines.append(
            f"- **{len(analysis.spikes)} volume spike(s).** Strongest: *{label}* on "
            f"{sp.period:%Y-%m-%d} with {sp.count} tickets vs a baseline of {sp.baseline:.1f} "
            f"(z = {sp.zscore:.1f})."
        )
    redactions = sum(analysis.ingest.pii_redactions.values())
    lines.append(f"- {redactions:,} PII spans were redacted before embedding.")

    lines += [
        "",
        "## Topics",
        "",
        "| # | Topic | Tickets | Share | Keywords |",
        "|---|---|---|---|---|",
    ]
    for t in analysis.topics.topics:
        lines.append(
            f"| {t.topic_id} | {t.label} | {t.size} | {t.size / total:.1%} | "
            f"{', '.join(t.keywords)} |"
        )

    if analysis.incidents:
        lines += ["", "## Incidents", "", "| # | Tickets | Window | Example |", "|---|---|---|---|"]
        for inc in analysis.incidents:
            window = f"{_fmt_ts(inc.start)} to {_fmt_ts(inc.end)}"
            lines.append(f"| {inc.incident_id} | {inc.size} | {window} | {inc.sample_text} |")

    if analysis.spikes:
        lines += [
            "",
            "## Spikes",
            "",
            "| Topic | Period | Count | Baseline | z |",
            "|---|---|---|---|---|",
        ]
        for sp in analysis.spikes[:15]:
            lines.append(
                f"| {analysis.topics.get(sp.topic_id).label} | {sp.period:%Y-%m-%d %H:%M} | "
                f"{sp.count} | {sp.baseline:.1f} | {sp.zscore:.1f} |"
            )

    if eval_summary:
        lines += ["", "## Retrieval quality", ""]
        cols = list(eval_summary[0].keys())
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for row in eval_summary:
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")

    lines += ["", "## Configuration", ""]
    lines += [f"- {k}: `{v}`" for k, v in analysis.config.items()]
    return "\n".join(lines) + "\n"
