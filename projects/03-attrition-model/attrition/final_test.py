"""Score plain LR on the locked test set, once. (Written by Claude at Rohit's request, 03-M5.)

Run from projects/03-attrition-model:  uv run python -m attrition.final_test
Fits on all 1,176 train rows, scores the 294 locked test rows, writes docs/test_results.json.
Refuses to run again once that file exists: the test set is a one-time look.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from attrition.baselines import logistic_model
from attrition.capacity import KS
from attrition.cv import ScoreFn
from attrition.features import CATEGORICAL, NUMERIC
from attrition.load import TARGET, load_raw
from attrition.metrics import precision_at_k
from attrition.split import make_split

RESULTS = Path(__file__).resolve().parents[1] / "docs" / "test_results.json"
HEADLINE_K = 30  # 10% of 294, the M3 operating point


def score_test(
    train: pd.DataFrame, test: pd.DataFrame, make_scores: ScoreFn, ks: list[int]
) -> dict:
    """Fit on all of train, score test once (without its labels), return metrics."""
    y_test = test[TARGET].to_numpy()
    scores = np.asarray(make_scores(train, test.drop(columns=[TARGET])), dtype=float)
    n_leavers = int(y_test.sum())

    capacity = []
    for k in ks:
        precision = precision_at_k(y_test, scores, k)
        capacity.append(
            {
                "k": k,
                "share_called": k / len(test),
                "precision": precision,
                "recall": precision * k / n_leavers,
                "leavers_found": round(precision * k),
                "random_finds": n_leavers / len(test) * k,
            }
        )
    return {
        "n_train": len(train),
        "n_test": len(test),
        "n_test_leavers": n_leavers,
        "pr_auc": float(average_precision_score(y_test, scores)),
        "capacity": capacity,
    }


def main() -> None:
    if RESULTS.exists() and "--force" not in sys.argv:
        sys.exit(f"{RESULTS} exists: the test set was already scored. Read that file instead.")

    train, test = make_split(load_raw())  # same seed (42) as M1, so the same 294 rows
    results = score_test(train, test, logistic_model(NUMERIC, CATEGORICAL), KS)
    results = {"model": "plain LR (NUMERIC + CATEGORICAL)", "date": str(date.today())} | results

    print(f"Plain LR on locked test: {results['n_test']} rows, {results['n_test_leavers']} leavers")
    print(f"PR-AUC {results['pr_auc']:.3f}\n")
    header = ["k", "% called", "precision", "recall", "found", "random"]
    widths = [3, 8, 9, 6, 5, 6]
    print("  ".join(f"{h:>{w}}" for h, w in zip(header, widths, strict=True)))
    for r in results["capacity"]:
        print(
            f"{r['k']:>3}  {r['share_called']:>8.1%}  {r['precision']:>9.3f}  {r['recall']:>6.3f}  "
            f"{r['leavers_found']:>5}  {r['random_finds']:>6.1f}"
        )

    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten: {RESULTS}")


if __name__ == "__main__":
    main()
