import numpy as np
import pandas as pd

KEYS = ["subject_id", "hadm_id", "stay_id", "hour"]


def load_data(path: str) -> pd.DataFrame:
    """Load the cohort feature matrix CSV and compute the integer hour column."""
    df = pd.read_csv(path)

    for c in ("intime", "outtime", "charttime_hour"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "hour" not in df.columns:
        if {"charttime_hour", "intime"}.issubset(df.columns):
            df["hour"] = np.floor(
                (df["charttime_hour"] - df["intime"]).dt.total_seconds() / 3600.0
            ).astype("Int64")
        else:
            raise ValueError("Need either 'hour' OR both 'charttime_hour' and 'intime'.")

    df = df.dropna(subset=["stay_id", "hour"]).copy()
    df["hour"] = df["hour"].astype(int)
    return df
