"""Load the raw IBM attrition CSV with clean types."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "attrition.csv"
TARGET = "Attrition"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")  # the file starts with a byte-order mark
    df[TARGET] = (df[TARGET] == "Yes").astype(int)  # 1 = left the company
    return df
