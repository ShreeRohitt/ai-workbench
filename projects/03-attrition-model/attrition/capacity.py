"""How many people should HR call? Precision and recall at several k. (Boilerplate: Claude.)

Run from projects/03-attrition-model:  uv run python -m attrition.capacity
Uses the same 25 folds as `cross_validate` (same seed), so numbers line up with the report.
Each eval fold holds ~235 employees; k is "people HR calls out of that group".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from attrition.baselines import logistic_model
from attrition.cv import ScoreFn
from attrition.features import CATEGORICAL, NUMERIC
from attrition.load import TARGET, load_raw
from attrition.metrics import precision_at_k
from attrition.plots import BAR, INK, INK_2, RULE
from attrition.split import make_split

KS = [10, 20, 30, 40, 50, 60]
FIGURE = Path(__file__).resolve().parents[1] / "docs" / "figures" / "capacity_curve.png"


def at_k_by_fold(
    train: pd.DataFrame,
    make_scores: ScoreFn,
    ks: list[int],
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """One row per (fold, k): precision@k and recall@k on that fold's eval rows.

    recall@k = leavers in the top k / all leavers in the fold = precision@k * k / n_leavers.
    """
    y = train[TARGET].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for i, (fit_idx, eval_idx) in enumerate(cv.split(np.zeros(len(train)), y)):
        fit_df, eval_df = train.iloc[fit_idx].copy(), train.iloc[eval_idx].copy()
        y_eval = eval_df[TARGET].to_numpy()
        scores = np.asarray(make_scores(fit_df, eval_df.drop(columns=[TARGET])), dtype=float)
        n_leavers = int(y_eval.sum())
        for k in ks:
            precision = precision_at_k(y_eval, scores, k)
            rows.append(
                {
                    "fold": i,
                    "n": len(eval_df),
                    "n_leavers": n_leavers,
                    "k": k,
                    "precision": precision,
                    "recall": precision * k / n_leavers,
                    "leavers_found": round(precision * k),
                }
            )
    return pd.DataFrame(rows)


def summarise(by_fold: pd.DataFrame) -> pd.DataFrame:
    """Mean and std across folds for each k."""
    summary = by_fold.groupby("k").agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        found_mean=("leavers_found", "mean"),
        n_leavers=("n_leavers", "mean"),
        n=("n", "mean"),
    )
    return summary.reset_index()


def format_table(summary: pd.DataFrame, base_rate: float) -> str:
    lines = [
        f"{'k':>3}  {'precision':>15}  {'recall':>15}  {'leavers found':>13}  {'random finds':>12}"
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"{int(r.k):>3}  {r.precision_mean:>6.3f} ± {r.precision_std:.3f}  "
            f"{r.recall_mean:>6.3f} ± {r.recall_std:.3f}  "
            f"{r.found_mean:>6.1f} of {r.n_leavers:>4.1f}  {base_rate * r.k:>12.1f}"
        )
    return "\n".join(lines)


def plot(summary: pd.DataFrame, base_rate: float, path: Path) -> None:
    """Two panels sharing the k axis: precision (left) and recall (right), band = ±1 std."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    k = summary["k"].to_numpy()
    n = summary["n"].mean()
    panels = [
        ("precision", "Precision: share of calls that reach a leaver", base_rate * np.ones_like(k)),
        ("recall", "Recall: share of all leavers reached", k / n),
    ]
    for ax, (metric, title, random_line) in zip(axes, panels, strict=True):
        mean = summary[f"{metric}_mean"].to_numpy()
        std = summary[f"{metric}_std"].to_numpy()
        ax.fill_between(k, mean - std, mean + std, color=BAR, alpha=0.15, linewidth=0)
        ax.plot(k, mean, color=BAR, linewidth=2, marker="o", markersize=5)
        ax.plot(k, random_line, color=INK_2, linewidth=1, linestyle="--")
        ax.text(k[-1], random_line[-1], " random", va="center", fontsize=9, color=INK_2)
        ax.text(k[-1], mean[-1], " logistic regression", va="center", fontsize=9, color=INK)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel(f"k = people HR calls, out of ~{n:.0f} employees", color=INK_2)
        ax.set_ylim(0, 1)
        ax.set_xlim(k[0] - 2, k[-1] + 22)
        ax.set_xticks(k)
        ax.grid(axis="y", color=RULE, linewidth=0.5)
        ax.tick_params(colors=INK_2, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(RULE)
    fig.text(0.01, 0.01, "Mean across 25 CV folds; band = ±1 std.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    train, _ = make_split(load_raw())
    base_rate = train[TARGET].mean()
    summary = summarise(at_k_by_fold(train, logistic_model(NUMERIC, CATEGORICAL), KS))
    print("Plain logistic regression, 25 folds; mean ± std across folds\n")
    print(format_table(summary, base_rate))
    plot(summary, base_rate, FIGURE)
    print(f"\nplot: {FIGURE}")


if __name__ == "__main__":
    main()
