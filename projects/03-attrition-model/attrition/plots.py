"""Plotting helpers for EDA. Boilerplate: you compute the numbers, these draw them."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

BAR = "#2a78d6"  # single series → one hue
INK = "#0b0b0b"
INK_2 = "#52514e"
RULE = "#c9c8c3"


def plot_rate_table(
    table: pd.DataFrame, title: str, base_rate: float, ax: plt.Axes | None = None
) -> plt.Axes:
    """Horizontal bars of attrition rate per group, labelled with the rate and group size.

    `table` is exactly what `train.groupby(col)["Attrition"].agg(["mean", "size"])` returns.
    The vertical line is the overall base rate, so bars read as "above / below average".
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.45 * len(table) + 1.2))

    labels = [str(i) for i in table.index]
    rates = table["mean"].to_numpy() * 100
    ax.barh(labels, rates, height=0.55, color=BAR)
    ax.invert_yaxis()  # first group on top, in table order

    pad = rates.max() * 0.02
    for y, (rate, n) in enumerate(zip(rates, table["size"], strict=True)):
        ax.text(
            rate + pad,
            y,
            f"{rate:.1f}%  (n={n})",
            va="center",
            fontsize=9,
            color=INK_2,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )

    ax.axvline(base_rate * 100, color=INK_2, linewidth=1, zorder=0)

    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(f"Attrition rate (%)   |   line = base rate {base_rate:.1%}", color=INK_2)
    ax.set_xlim(0, rates.max() * 1.35)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    return ax
