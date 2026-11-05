import pandas as pd
import pytest

from attrition.features import add_overtime_and_level1


def people():
    # index labels deliberately not 0..n-1, like a CV fold
    return pd.DataFrame(
        {
            "OverTime": ["Yes", "Yes", "No", "No", "Yes"],
            "JobLevel": [1, 2, 1, 3, 1],
            "Age": [25, 40, 30, 50, 22],
        },
        index=[40, 7, 13, 2, 99],
    )


def test_flags_only_overtime_and_level1():
    out = add_overtime_and_level1(people())
    assert list(out["overtime_and_level1"]) == [1, 0, 0, 0, 1]


def test_column_is_int():
    out = add_overtime_and_level1(people())
    assert out["overtime_and_level1"].dtype.kind == "i"


def test_input_is_not_changed():
    df = people()
    add_overtime_and_level1(df)
    assert "overtime_and_level1" not in df.columns


def test_keeps_rows_index_and_other_columns():
    df = people()
    out = add_overtime_and_level1(df)
    assert list(out.index) == list(df.index)
    pd.testing.assert_frame_equal(out[df.columns], df)


def test_rejects_unexpected_overtime_values():
    df = people().assign(OverTime=["yes", "Yes", "No", "No", "Yes"])
    with pytest.raises(ValueError, match="OverTime must be 'Yes' or 'No'"):
        add_overtime_and_level1(df)
