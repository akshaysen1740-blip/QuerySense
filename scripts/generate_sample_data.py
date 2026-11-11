"""Write the synthetic dataset and golden queries to data/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from signalsift.datasets import GOLDEN_QUERIES, generate_tickets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    generate_tickets(n=args.rows, seed=args.seed).to_csv(
        args.out / "sample_tickets.csv", index=False
    )
    (args.out / "golden_queries.json").write_text(json.dumps(GOLDEN_QUERIES, indent=2))
    print(f"Wrote {args.rows} tickets and {len(GOLDEN_QUERIES)} golden queries to {args.out}/")


if __name__ == "__main__":
    main()
