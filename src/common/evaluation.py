"""Shared evaluation helpers.

Three report layers used by every training run:
  1) global metrics for the overall comparison,
  2) precipitation-bin metrics for behavior in the rare heavy-rain regimes,
  3) context-flag metrics as the direct test of the heavy-rain hypothesis.
Model selection is based on layers 2 and 3, not only on global scores.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score
)

from .features import TARGET

PRECIP_BINS = [-0.101, 2, 5, 10, 100]
PRECIP_LABELS = ["0–2mm", "2–5mm", "5–10mm", ">10mm"]

CONTEXT_COLS = [
    "heavy_rain_safe_context",
    "heavy_rain_risky_context",
    "heavy_rain_high_wind",
    "heavy_rain_night"
]


def print_precipitation_distribution(df, title):
    # sanity check: train and test should keep similar heavy-rain coverage
    # after the group split
    temp_df = df.copy()

    temp_df["precip_bin"] = pd.cut(
        temp_df["precipitation"],
        bins=PRECIP_BINS,
        labels=PRECIP_LABELS
    )

    report = (
        temp_df.groupby("precip_bin", observed=False)[TARGET]
        .agg(
            total="count",
            positive="sum",
            positive_ratio="mean"
        )
        .reset_index()
    )

    print(f"\n=== {title} ===")
    print(report)


def evaluate_global_metrics(run_name, model_name, y_true, y_pred, y_proba=None):
    metrics = {
        "run_name": run_name,
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = None

    return metrics


def _group_metrics_row(run_name, model_name, group):
    row = {
        "run_name": run_name,
        "model": model_name,
        "total": len(group),
        "positive": int(group["y_true"].sum()),
        "true_positive_ratio": group["y_true"].mean(),
        "predicted_positive": int(group["y_pred"].sum()),
        "predicted_positive_ratio": group["y_pred"].mean(),
        "accuracy": accuracy_score(group["y_true"], group["y_pred"]),
        "precision": precision_score(group["y_true"], group["y_pred"], zero_division=0),
        "recall": recall_score(group["y_true"], group["y_pred"], zero_division=0),
        "f1": f1_score(group["y_true"], group["y_pred"], zero_division=0),
        "confusion_matrix": confusion_matrix(group["y_true"], group["y_pred"]).tolist()
    }

    if "y_proba" in group.columns:
        row["mean_predicted_probability"] = group["y_proba"].mean()

    return row


def evaluate_precipitation_bins(run_name, model_name, test_df, y_true, y_pred, y_proba=None):
    # global metrics hide what happens in the rare heavy-rain bins, so each
    # rain interval is evaluated separately. comparing predicted vs true
    # positive ratio per bin is what exposed the NN over-predicting in >10mm
    eval_df = test_df.copy()
    eval_df["y_true"] = np.asarray(y_true)
    eval_df["y_pred"] = np.asarray(y_pred)

    if y_proba is not None:
        eval_df["y_proba"] = np.asarray(y_proba)

    eval_df["precip_bin"] = pd.cut(
        eval_df["precipitation"],
        bins=PRECIP_BINS,
        labels=PRECIP_LABELS
    )

    rows = []

    for precip_bin, group in eval_df.groupby("precip_bin", observed=False):
        if group.empty:
            continue

        row = _group_metrics_row(run_name, model_name, group)
        row["precip_bin"] = str(precip_bin)
        rows.append(row)

    # keep the original column order of the saved reports
    front = ["run_name", "model", "precip_bin"]
    df_out = pd.DataFrame(rows)
    return df_out[front + [c for c in df_out.columns if c not in front]]


def evaluate_context_bins(run_name, model_name, test_df, y_true, y_pred, y_proba=None):
    # direct test of the hypothesis: same rain amount, different context,
    # the model should behave differently
    eval_df = test_df.copy()
    eval_df["y_true"] = np.asarray(y_true)
    eval_df["y_pred"] = np.asarray(y_pred)

    if y_proba is not None:
        eval_df["y_proba"] = np.asarray(y_proba)

    rows = []

    for col in CONTEXT_COLS:
        for value in [0, 1]:
            group = eval_df[eval_df[col] == value]

            if group.empty:
                continue

            row = _group_metrics_row(run_name, model_name, group)
            row["context_feature"] = col
            row["context_value"] = value
            rows.append(row)

    front = ["run_name", "model", "context_feature", "context_value"]
    df_out = pd.DataFrame(rows)
    return df_out[front + [c for c in df_out.columns if c not in front]]
