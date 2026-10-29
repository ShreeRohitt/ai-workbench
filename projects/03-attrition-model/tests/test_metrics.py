import pandas as pd
import pytest

from attrition.metrics import precision_at_k


def test_perfect_ranking():
    # both leavers have the highest scores
    assert precision_at_k([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], k=2) == 1.0


def test_worst_ranking():
    assert precision_at_k([1, 0, 1, 0], [0.1, 0.9, 0.2, 0.8], k=2) == 0.0


def test_half_right():
    # top 2 by score are rows 0 and 1: one leaver, one stayer
    assert precision_at_k([1, 0, 0, 1, 0], [0.5, 0.4, 0.3, 0.2, 0.1], k=2) == 0.5


def test_k_equals_n_is_base_rate():
    y = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert precision_at_k(y, list(range(10)), k=10) == pytest.approx(0.2)


def test_scores_need_not_be_sorted():
    # the function must do the ranking itself
    assert precision_at_k([0, 1, 0], [0.2, 0.9, 0.5], k=1) == 1.0


def test_ties_keep_original_order():
    # equal scores: the earlier row ranks higher, so the stayer at row 0 is picked
    assert precision_at_k([0, 1], [0.5, 0.5], k=1) == 0.0


def test_accepts_pandas_series():
    y = pd.Series([1, 0, 1], index=[10, 20, 30])  # non-default index, like a CV fold
    s = pd.Series([0.9, 0.8, 0.1], index=[10, 20, 30])
    assert precision_at_k(y, s, k=2) == 0.5


@pytest.mark.parametrize("k", [0, -1, 4])
def test_k_out_of_range_raises(k):
    with pytest.raises(ValueError):
        precision_at_k([1, 0, 1], [0.3, 0.2, 0.1], k=k)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        precision_at_k([1, 0, 1], [0.3, 0.2], k=1)
