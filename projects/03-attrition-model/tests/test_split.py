import pandas as pd
import pytest

from attrition.load import TARGET
from attrition.split import make_split


@pytest.fixture
def df():
    # 1000 rows, 16% positives — same imbalance as the real data
    return pd.DataFrame({"EmployeeNumber": range(1000), TARGET: [1] * 160 + [0] * 840})


def test_sizes(df):
    train, test = make_split(df)
    assert len(test) == 200 and len(train) == 800


def test_no_row_in_both(df):
    train, test = make_split(df)
    assert set(train.EmployeeNumber).isdisjoint(test.EmployeeNumber)
    assert len(train) + len(test) == len(df)


def test_stratified(df):
    train, test = make_split(df)
    assert test[TARGET].mean() == pytest.approx(0.16, abs=0.005)
    assert train[TARGET].mean() == pytest.approx(0.16, abs=0.005)


def test_reproducible(df):
    a, _ = make_split(df, seed=7)
    b, _ = make_split(df, seed=7)
    c, _ = make_split(df, seed=8)
    assert a.EmployeeNumber.tolist() == b.EmployeeNumber.tolist()
    assert a.EmployeeNumber.tolist() != c.EmployeeNumber.tolist()
