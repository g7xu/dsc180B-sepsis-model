from typing import Tuple, List

import pandas as pd

KEYS = ["subject_id", "hadm_id", "stay_id", "hour"]
TIMESTAMP_COLS = ["intime", "outtime", "charttime_hour"]


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "sepsis",
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Select features and target from the full dataframe.

    Returns (X, y, feature_cols).
    """
    drop_cols = set(KEYS + [target_col])
    for c in TIMESTAMP_COLS:
        if c in df.columns:
            drop_cols.add(c)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df[target_col].astype(int).copy()

    # Drop any datetime columns that slipped through
    dt_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    if dt_cols:
        X = X.drop(columns=dt_cols)
        feature_cols = [c for c in feature_cols if c not in dt_cols]

    return X, y, feature_cols
