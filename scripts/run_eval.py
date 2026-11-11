"""Run the full pipeline + retrieval evaluation from the command line.

Examples:
    python scripts/run_eval.py                                   # bundled sample data
    python scripts/run_eval.py --csv my.csv --golden golden.json --text-col body
    SIGNALSIFT_EMBEDDER=sbert SIGNALSIFT_RERANKER=cross-encoder python scripts/run_eval.py
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from signalsift.config import Settings
from signalsift.service import InsightService


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--text-col")
    parser.add_argument("--time-col")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()

    settings = Settings(session_dir=Path(tempfile.mkdtemp(prefix="signalsift-eval-")))
    service = InsightService(settings)
    if args.csv:
        summary = service.create_session(
            pd.read_csv(args.csv), args.csv.name, args.text_col, args.time_col
        )
    else:
        summary = service.create_sample_session()
    sid = summary["session_id"]
    queries = json.loads(args.golden.read_text()) if args.golden else None
    result = service.evaluate(sid, queries, k=args.k)

    table = pd.DataFrame(result["summary"]).set_index("system")
    print(f"\nEmbedder={settings.embedder}  reranker={settings.reranker}  k={args.k}\n")
    print(table.to_markdown())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_summary.md").write_text(table.to_markdown() + "\n")
    pd.DataFrame(result["per_query"]).to_csv(args.out / "eval_per_query.csv", index=False)
    (args.out / "report.md").write_text(service.report_md(sid))
    print(f"\nWrote {args.out}/eval_summary.md, eval_per_query.csv and report.md")


if __name__ == "__main__":
    main()
