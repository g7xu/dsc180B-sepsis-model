"""CLI entry point: train a Random Forest sepsis prediction model.

Usage:
    python -m src.train --data data/cohort_feature_matrix_labeled_outcome.csv
"""
import argparse
import logging
import time
from pathlib import Path

import joblib
import numpy as np

from src.data import load_data
from src.labels import make_target
from src.features import build_feature_matrix
from src.split import group_train_test_split
from src.model import build_pipeline
from src.evaluate import evaluate

logger = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s"


def main(args: argparse.Namespace) -> None:
    t_start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Parameters ───────────────────────────────────────────────────
    logger.info("── Parameters ──")
    logger.info("  data:              %s", args.data)
    logger.info("  output_dir:        %s", output_dir)
    logger.info("  test_size:         %.2f", args.test_size)
    logger.info("  n_estimators:      %d", args.n_estimators)
    logger.info("  min_samples_leaf:  %d", args.min_samples_leaf)
    logger.info("  max_depth:         %s", args.max_depth or "None (unlimited)")
    logger.info("  seed:              %d", args.seed)

    # ── Load & prepare ──────────────────────────────────────────────
    logger.info("Loading data from %s", args.data)
    t0 = time.time()
    df = load_data(args.data)
    logger.info("Loaded %d rows x %d cols in %s", *df.shape, _fmt_duration(time.time() - t0))

    df["sepsis"] = make_target(df)
    logger.info("Target distribution:\n%s", df["sepsis"].value_counts().to_string())

    X, y, feature_cols = build_feature_matrix(df)
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    logger.info(
        "Feature matrix: %d rows x %d features (%d numeric, %d categorical)  |  positive rate: %.3f",
        X.shape[0], X.shape[1], len(num_cols), len(cat_cols), y.mean(),
    )

    # ── Split ───────────────────────────────────────────────────────
    groups = df.loc[X.index, "stay_id"].values
    n_stays = len(np.unique(groups))
    X_train, X_test, y_train, y_test = group_train_test_split(
        X, y, groups, test_size=args.test_size, random_state=args.seed,
    )
    logger.info(
        "Split %d unique stays → Train: %d rows (%.1f%% pos) | Test: %d rows (%.1f%% pos)",
        n_stays, len(X_train), y_train.mean() * 100, len(X_test), y_test.mean() * 100,
    )

    # ── Train ───────────────────────────────────────────────────────
    logger.info(
        "Training Random Forest (n_estimators=%d, min_samples_leaf=%d, max_depth=%s, seed=%d) ...",
        args.n_estimators, args.min_samples_leaf, args.max_depth or "None", args.seed,
    )
    t0 = time.time()
    pipeline = build_pipeline(
        X_train,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_depth=args.max_depth,
        random_state=args.seed,
    )
    pipeline.fit(X_train, y_train)
    train_duration = time.time() - t0
    logger.info("Training complete in %s", _fmt_duration(train_duration))

    # ── Evaluate ────────────────────────────────────────────────────
    logger.info("Running evaluation on test set ...")
    t0 = time.time()
    p_hat = pipeline.predict_proba(X_test)[:, 1]
    results = evaluate(np.asarray(y_test), p_hat)
    logger.info("Evaluation complete in %s", _fmt_duration(time.time() - t0))

    logger.info("── Results ──")
    logger.info("  ROC-AUC:  %.4f", results["roc_auc"])
    logger.info("  PR-AUC:   %.4f", results["pr_auc"])
    logger.info("  Best F1 threshold:  %.2f", results["best_f1_threshold"])
    if not np.isnan(results["best_spec_threshold"]):
        logger.info("  Best specificity (recall >= 0.80) threshold:  %.2f", results["best_spec_threshold"])

    # ── Save artifacts ──────────────────────────────────────────────
    model_path = output_dir / "sepsis_rf_pipeline.joblib"
    cols_path = output_dir / "feature_cols.joblib"
    preds_path = output_dir / "predictions.csv"

    joblib.dump(pipeline, model_path)
    joblib.dump(feature_cols, cols_path)
    logger.info("Model saved to %s", model_path)
    logger.info("Feature cols saved to %s", cols_path)

    df.loc[X_test.index, "pred_proba_sepsis"] = p_hat
    thr = results["best_f1_threshold"]
    df.loc[X_test.index, "pred_sepsis"] = (p_hat >= thr).astype(int)
    df.to_csv(preds_path, index=False)
    logger.info("Predictions saved to %s", preds_path)

    logger.info("── Done in %s ──", _fmt_duration(time.time() - t_start))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sepsis RF model")
    parser.add_argument("--data", required=True, help="Path to cohort feature matrix CSV")
    parser.add_argument("--output-dir", default="data", help="Directory for output artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=10, help="Max tree depth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main(args)
