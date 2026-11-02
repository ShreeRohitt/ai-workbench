"""Score every baseline on the same CV folds of train and print a table. (Boilerplate: Claude.)

Run from projects/03-attrition-model:  uv run python -m attrition.report
The test set is never loaded into a model here: only make_split's train half is scored.
"""

from __future__ import annotations

import pandas as pd

from attrition.baselines import logistic_model, overtime_level1_rule, random_model
from attrition.cv import ScoreFn, cross_validate
from attrition.features import CATEGORICAL, NUMERIC, SURVEY
from attrition.load import TARGET, load_raw
from attrition.split import make_split

METRICS = {"pr_auc": "PR-AUC", "precision_at_k": "precision@10%"}


def baseline_models() -> dict[str, ScoreFn]:
    """The models in the table, in display order. Add a line here to add a row."""
    return {
        "random": random_model(),
        "rule: overtime + level 1": overtime_level1_rule,
        "logistic regression": logistic_model(NUMERIC, CATEGORICAL),
        "logistic regression + survey": logistic_model(NUMERIC + SURVEY, CATEGORICAL),
    }


def compare(train: pd.DataFrame, models: dict[str, ScoreFn]) -> pd.DataFrame:
    """One row per model: mean, std, min and max of each metric across folds.

    cross_validate's default seed is used for every model, so all of them see identical folds.
    """
    rows = []
    for name, make_scores in models.items():
        folds = cross_validate(train, make_scores)
        row: dict[str, str | float | int] = {"model": name, "folds": len(folds)}
        for metric in METRICS:
            row[f"{metric}_mean"] = folds[metric].mean()
            row[f"{metric}_std"] = folds[metric].std()
            row[f"{metric}_min"] = folds[metric].min()
            row[f"{metric}_max"] = folds[metric].max()
        rows.append(row)
    return pd.DataFrame(rows)


def format_table(summary: pd.DataFrame) -> str:
    """Render compare()'s output as aligned text: mean ± std (min to max) per metric."""
    rows = [["model", *METRICS.values()]]
    for _, row in summary.iterrows():
        cells = [row["model"]]
        for metric in METRICS:
            cells.append(
                f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f} "
                f"({row[f'{metric}_min']:.3f} to {row[f'{metric}_max']:.3f})"
            )
        rows.append(cells)
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows]
    return "\n".join(line.rstrip() for line in lines)


def main() -> None:
    train, _ = make_split(load_raw())
    summary = compare(train, baseline_models())
    print(f"train rows: {len(train)}, leavers: {train[TARGET].mean():.1%}")
    print(f"folds per model: {summary['folds'].iloc[0]}; mean ± std (min to max) across folds\n")
    print(format_table(summary))


if __name__ == "__main__":
    main()
