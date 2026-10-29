import pandas as pd
from sklearn.model_selection import train_test_split

from attrition.load import TARGET


def make_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test), stratified on the target column."""
    train, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    return train, test
