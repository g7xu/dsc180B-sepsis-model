import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(
    X: pd.DataFrame,
    n_estimators: int = 500,
    min_samples_leaf: int = 5,
    max_depth: int | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Build the preprocessing + Random Forest pipeline."""
    cat_cols = [c for c in X.columns if pd.api.types.is_string_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("prep", preprocess),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )),
    ])
