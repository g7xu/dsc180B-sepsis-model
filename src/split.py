from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def group_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split by ICU stay so no stay appears in both train and test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    return (
        X.iloc[train_idx].copy(),
        X.iloc[test_idx].copy(),
        y.iloc[train_idx].copy(),
        y.iloc[test_idx].copy(),
    )
