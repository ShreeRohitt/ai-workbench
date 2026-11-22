"""Which inputs drive plain LR, and is that ranking stable across folds? (Boilerplate: Claude.)

Run from projects/03-attrition-model:  uv run python -m attrition.importance
Importance = mean |contribution| on the log-odds score (see attrition.explain).
Stability: refit on each of the same 25 CV folds (same seed as cross_validate), explain that
fold's eval rows against its fit rows, and rank. Test set never loaded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from attrition.baselines import logistic_pipeline
from attrition.explain import group_contributions, lr_contributions
from attrition.features import CATEGORICAL, GROUPS, NUMERIC
from attrition.load import TARGET, load_raw
from attrition.plots import BAR, INK, INK_2, RULE
from attrition.split import make_split

FEATURES = NUMERIC + CATEGORICAL
FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures"


def fit_and_explain(fit_df: pd.DataFrame, explain_df: pd.DataFrame) -> pd.DataFrame:
    """Fit plain LR on fit_df; grouped contributions for explain_df, average = fit_df."""
    model = logistic_pipeline(NUMERIC, CATEGORICAL).fit(fit_df[FEATURES], fit_df[TARGET])
    _, contrib = lr_contributions(model, explain_df[FEATURES], fit_df[FEATURES])
    return group_contributions(contrib, GROUPS)


def importance_by_fold(
    train: pd.DataFrame, n_splits: int = 5, n_repeats: int = 5, seed: int = 42
) -> pd.DataFrame:
    """One row per fold, one column per input (grouped): mean |contribution| on eval rows."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for fit_idx, eval_idx in cv.split(np.zeros(len(train)), train[TARGET]):
        grouped = fit_and_explain(train.iloc[fit_idx], train.iloc[eval_idx])
        rows.append(grouped.abs().mean())
    return pd.DataFrame(rows)


def stability_table(full: pd.Series, by_fold: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    ranks = by_fold.rank(axis=1, ascending=False)
    table = pd.DataFrame(
        {
            "full_train": full,
            "fold_mean": by_fold.mean(),
            "fold_std": by_fold.std(),
            "median_rank": ranks.median(),
            "best_rank": ranks.min(),
            "worst_rank": ranks.max(),
            f"in_top{top}": (ranks <= top).sum(),
        }
    )
    return table.sort_values("full_train", ascending=False)


def plot_importance(table: pd.DataFrame, folds: int, path: Path) -> None:
    """Horizontal bars: full-train importance; whisker = ±1 std across folds."""
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(table) + 1.4))
    labels = list(table.index)
    ax.barh(labels, table["full_train"], height=0.6, color=BAR)
    ax.errorbar(table["full_train"], labels, xerr=table["fold_std"], fmt="none", ecolor=INK_2, lw=1)
    ax.invert_yaxis()
    pad = table["full_train"].max() * 0.03
    for y, (value, std) in enumerate(zip(table["full_train"], table["fold_std"], strict=True)):
        ax.text(value + std + pad, y, f"{value:.2f}", va="center", fontsize=9, color=INK_2)
    ax.set_title(
        "What pushes plain LR's risk score, on average", loc="left", fontsize=11, color=INK
    )
    ax.set_xlabel(
        f"mean |contribution| to log-odds score   |   whisker = ±1 std across {folds} folds",
        color=INK_2,
    )
    ax.set_xlim(0, (table["full_train"] + table["fold_std"]).max() * 1.2)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_strip(grouped: pd.DataFrame, y: pd.Series, order: list[str], path: Path) -> None:
    """One dot per employee per input: which way and how hard it pushes.

    Stayers (grey) sit in the upper half of each row, leavers (blue) in the lower half, so
    neither hides the other where many people share one value (e.g. OverTime).
    """
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.6))
    for row, col in enumerate(order):
        for is_leaver, colour, size, label in [(0, INK_2, 5, "stayed"), (1, BAR, 9, "left")]:
            mask = (y == is_leaver).to_numpy()
            low, high = (-0.38, -0.04) if not is_leaver else (0.04, 0.38)
            ax.scatter(
                grouped[col].to_numpy()[mask],
                row + rng.uniform(low, high, mask.sum()),
                s=size,
                color=colour,
                alpha=0.7 if is_leaver else 0.5,
                linewidths=0,
                label=label if row == 0 else None,
            )
    ax.axvline(0, color=INK_2, lw=1, zorder=0)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_title(
        "Each employee's push from each input (0 = same as the average employee)",
        loc="left",
        fontsize=11,
        color=INK,
    )
    ax.set_xlabel("contribution to log-odds score   |   right = raises risk", color=INK_2)
    ax.legend(loc="lower right", frameon=False, fontsize=9, markerscale=2)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    train, _ = make_split(load_raw())
    grouped = fit_and_explain(train, train)
    full = grouped.abs().mean()
    by_fold = importance_by_fold(train)
    table = stability_table(full, by_fold)

    pd.set_option("display.width", 120)
    print("Plain LR, grouped inputs. full_train = fit and explain all 1,176 train rows;")
    print(f"fold_* = refit per fold, explain eval rows ({len(by_fold)} folds)\n")
    print(table.round(3).to_string())

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_importance(table, len(by_fold), FIGURES / "importance_grouped.png")
    plot_strip(grouped, train[TARGET], list(table.index), FIGURES / "contributions_strip.png")
    print(f"\nplots in {FIGURES}: importance_grouped.png, contributions_strip.png")


if __name__ == "__main__":
    main()
