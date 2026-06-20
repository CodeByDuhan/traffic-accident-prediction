import sys
from pathlib import Path

# make the repo root importable
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

from src.common.features import add_features, TARGET
from src.common.artifacts import load_artifacts, build_model_matrix


# ============================================================
# PATH SETTINGS
# ============================================================

DATA_PATH = BASE_DIR / "data" / "processed" / "test_corrected_augmented_dataset_t+1.csv"

MODEL_DIR = BASE_DIR / "models" / "train_new_dataset_op"
FIGURE_DIR = BASE_DIR / "reports" / "figures" / "op"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RUN_NAME = "train_new_dataset_op"

XGB_MODEL_PATH = MODEL_DIR / f"xgboost_{RUN_NAME}.pkl"
NN_MODEL_PATH = MODEL_DIR / f"neural_network_{RUN_NAME}.keras"


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
NN_PERMUTATION_SAMPLE_SIZE = 5000


def predict_nn_binary(model, X_array):
    proba = model.predict(X_array, verbose=0).reshape(-1)
    return (proba > 0.5).astype(int)


# ============================================================
# MAIN
# ============================================================

def main():
    df = pd.read_csv(DATA_PATH)
    df = add_features(df)

    artifacts = load_artifacts(MODEL_DIR, RUN_NAME)
    feature_order = artifacts["feature_order"]

    X = build_model_matrix(df, artifacts)

    xgb_model = joblib.load(XGB_MODEL_PATH)
    nn_model = load_model(NN_MODEL_PATH)

    # --------------------------------------------------------
    # XGBoost Feature Importance
    # --------------------------------------------------------

    importances = xgb_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    top_n = min(30, len(feature_order))

    plt.figure(figsize=(12, 7))
    plt.title("XGBoost Feature Importance - Unweighted")
    plt.bar(range(top_n), importances[sorted_idx][:top_n])
    plt.xticks(
        range(top_n),
        np.array(feature_order)[sorted_idx][:top_n],
        rotation=90
    )
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "xgb_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # Neural Network Permutation Importance
    # --------------------------------------------------------

    # full test set is too slow for NN permutation, a 5k sample is enough
    # for a stable importance ranking
    sample_df = df.sample(
        n=min(len(df), NN_PERMUTATION_SAMPLE_SIZE),
        random_state=RANDOM_STATE
    )

    X_sample = build_model_matrix(sample_df, artifacts)
    y_sample = sample_df[TARGET].astype(int)

    X_np = X_sample.to_numpy().astype(np.float32)
    y_np = y_sample.to_numpy()

    baseline_pred = predict_nn_binary(nn_model, X_np)
    baseline_acc = accuracy_score(y_np, baseline_pred)

    rng = np.random.default_rng(RANDOM_STATE)
    importance_rows = []

    for col_idx, feature_name in enumerate(X_sample.columns):
        X_perm = X_np.copy()
        X_perm[:, col_idx] = rng.permutation(X_perm[:, col_idx])

        perm_pred = predict_nn_binary(nn_model, X_perm)
        perm_acc = accuracy_score(y_np, perm_pred)

        importance_rows.append({
            "feature": feature_name,
            "importance": baseline_acc - perm_acc
        })

    importance_df = pd.DataFrame(importance_rows)
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(FIGURE_DIR / "nn_permutation_importance_values.csv", index=False)

    top_df = importance_df.head(30)

    plt.figure(figsize=(12, 7))
    plt.title("Neural Network Permutation Importance - OP")
    plt.bar(range(len(top_df)), top_df["importance"])
    plt.xticks(range(len(top_df)), top_df["feature"], rotation=90)
    plt.ylabel("Accuracy drop after permutation")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "nn_permutation_importance.png", dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Feature importance figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
