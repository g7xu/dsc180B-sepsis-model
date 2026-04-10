import pandas as pd


def to01(s: pd.Series) -> pd.Series:
    """Coerce a mixed-type series to binary 0/1 integers."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)

    ss = s.astype(str).str.strip().str.lower()
    mapping = {
        "true": 1, "false": 0,
        "1": 1, "0": 0,
        "yes": 1, "no": 0,
        "t": 1, "f": 0,
    }
    out = ss.map(mapping)
    out = out.fillna(pd.to_numeric(s, errors="coerce"))
    return out.astype(float).fillna(0).astype(int)


def make_target(df: pd.DataFrame, target_col: str = "sepsis") -> pd.Series:
    """Return a clean binary target series."""
    return to01(df[target_col])
