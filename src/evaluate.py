from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


def eval_at_threshold(y_true: np.ndarray, p_hat: np.ndarray, thr: float) -> Dict:
    """Compute classification metrics at a single probability threshold."""
    y_pred = (p_hat >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else np.nan
    bal_acc = (sensitivity + specificity) / 2 if (np.isfinite(sensitivity) and np.isfinite(specificity)) else np.nan
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan

    return {
        "threshold": float(thr),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "accuracy": acc, "balanced_acc": bal_acc,
        "precision": precision, "recall": sensitivity,
        "specificity": specificity, "NPV": npv,
        "F1": f1, "FPR": fpr, "FNR": fnr,
    }


def evaluate(y_true: np.ndarray, p_hat: np.ndarray, target_recall: float = 0.80) -> Dict:
    """Run full evaluation: AUC metrics, threshold sweep, and best operating points."""
    results = {}

    # Threshold-free metrics
    if len(np.unique(y_true)) == 2:
        results["roc_auc"] = roc_auc_score(y_true, p_hat)
        results["pr_auc"] = average_precision_score(y_true, p_hat)
    else:
        results["roc_auc"] = np.nan
        results["pr_auc"] = np.nan

    # Threshold sweep
    thresholds = np.linspace(0.01, 0.99, 99)
    sweep = pd.DataFrame([eval_at_threshold(y_true, p_hat, t) for t in thresholds])
    results["sweep"] = sweep

    # Best F1
    best_f1_row = sweep.loc[sweep["F1"].idxmax()]
    results["best_f1_threshold"] = best_f1_row["threshold"]

    # Best specificity with recall >= target
    eligible = sweep[sweep["recall"] >= target_recall]
    if len(eligible) > 0:
        best_spec_row = eligible.loc[eligible["specificity"].idxmax()]
        results["best_spec_threshold"] = best_spec_row["threshold"]
    else:
        results["best_spec_threshold"] = np.nan

    return results
