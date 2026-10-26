import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from attrition.plots import plot_rate_table  # noqa: E402


def test_one_bar_per_group_with_labels():
    table = pd.DataFrame({"mean": [0.111, 0.285], "size": [836, 340]}, index=["No", "Yes"])
    ax = plot_rate_table(table, "Attrition by OverTime", base_rate=0.162)
    assert len(ax.patches) == 2
    texts = [t.get_text() for t in ax.texts]
    assert "28.5%  (n=340)" in texts
    assert "base rate 16.2%" in ax.get_xlabel()
