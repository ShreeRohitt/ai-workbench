"""Print chunk-length stats per strategy as a Markdown table.

    cd projects/01-rag-with-evals && uv run python -m rag.stats
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rag.chunking import chunk_by_heading, chunk_fixed
from rag.ingest import load_docs
from rag.tokens import ENCODING

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "handbook"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--min-tokens", type=int, default=50)
    args = p.parse_args()

    docs = load_docs(args.root)
    strategies = {
        f"fixed ({args.max_tokens}/{args.overlap})": lambda d: chunk_fixed(
            d, args.max_tokens, args.overlap
        ),
        f"heading (max {args.max_tokens}, min {args.min_tokens})": lambda d: chunk_by_heading(
            d, args.max_tokens, args.min_tokens
        ),
    }

    print(f"{len(docs)} docs, tokenizer {ENCODING}\n")
    print("| strategy | chunks | p50 tok | p95 tok | max tok | < 50 tok |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, fn in strategies.items():
        lens = np.array([c.n_tokens for d in docs for c in fn(d)])
        print(
            f"| {name} | {len(lens)} | {np.percentile(lens, 50):.0f} | "
            f"{np.percentile(lens, 95):.0f} | {lens.max()} | {(lens < 50).sum()} |"
        )


if __name__ == "__main__":
    main()
