import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score


# ============================================================
# PATH SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "test_corrected_augmented_dataset_t+1.csv"

MODEL_DIR = BASE_DIR / "models" / "train_new_dataset_op"
FIGURE_DIR = BASE_DIR / "reports" / "figures" / "op"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILE PATHS
# ============================================================

XGB_MODEL_PATH = MODEL_DIR / "xgboost_train_new_dataset_op.pkl"
NN_MODEL_PATH = MODEL_DIR / "neural_network_train_new_dataset_op.keras"

ENCODER_PATH = MODEL_DIR / "encoder_train_new_dataset_op.pkl"
SCALER_PATH = MODEL_DIR / "scaler_train_new_dataset_op.pkl"
FEATURE_ORDER_PATH = MODEL_DIR / "feature_order_train_new_dataset_op.pkl"


# ============================================================
# CONFIG
# ============================================================

TARGET = "accident_happened_t+1"
RANDOM_STATE = 42
NN_PERMUTATION_SAMPLE_SIZE = 5000


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_saganak"] = (df["precipitation"] > 5).astype(int)

    df["is_night"] = (
        (df["hour"] >= 22) |
        (df["hour"] <= 6)
    ).astype(int)

    df["dangerous_temp"] = (
        (df["temperature_2m"] < 0) |
        (df["temperature_2m"] > 35)
    ).astype(int)

    df["high_wind"] = (df["windspeed_10m"] > 20).astype(int)

    df["rain_0_2"] = (
        (df["precipitation"] > 0) &
        (df["precipitation"] <= 2)
    ).astype(int)

    df["rain_2_5"] = (
        (df["precipitation"] > 2) &
        (df["precipitation"] <= 5)
    ).astype(int)

    df["rain_5_10"] = (
        (df["precipitation"] > 5) &
        (df["precipitation"] <= 10)
    ).astype(int)

    df["rain_gt_10"] = (df["precipitation"] > 10).astype(int)

    df["heavy_rain_safe_context"] = (
        (df["precipitation"] > 5) &
        (df["temperature_2m"].between(10, 25)) &
        (df["windspeed_10m"] < 10) &
        (df["hour"].between(9, 18))
    ).astype(int)

    df["heavy_rain_risky_context"] = (
        (df["precipitation"] > 5) &
        (
            (df["is_night"] == 1) |
            (df["high_wind"] == 1) |
            (df["dangerous_temp"] == 1)
        )
    ).astype(int)

    df["heavy_rain_high_wind"] = (
        (df["precipitation"] > 5) &
        (df["high_wind"] == 1)
    ).astype(int)

    df["heavy_rain_night"] = (
        (df["precipitation"] > 5) &
        (df["is_night"] == 1)
    ).astype(int)

    return df


def build_model_matrix(df, encoder, scaler, feature_order):
    cat_cols = ["city", "yağış_türü"]

    num_cols = [
        "hour",
        "temperature_2m",
        "windspeed_10m",
        "precipitation"
    ]

    binary_cols = [
        "is_night",
        "dangerous_temp",
        "high_wind",
        "is_saganak",

        "rain_0_2",
        "rain_2_5",
        "rain_5_10",
        "rain_gt_10",

        "heavy_rain_safe_context",
        "heavy_rain_risky_context",
        "heavy_rain_high_wind",
        "heavy_rain_night"
    ]

    X_cat = pd.DataFrame(
        encoder.transform(df[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=df.index
    )

    X_num = pd.DataFrame(
        scaler.transform(df[num_cols]),
        columns=num_cols,
        index=df.index
    )

    X_bin = df[binary_cols].astype(int)

    X = pd.concat([X_num, X_bin, X_cat], axis=1)
    X = X[feature_order]

    return X


def predict_nn_binary(model, X_array):
    proba = model.predict(X_array, verbose=0).reshape(-1)
    return (proba > 0.5).astype(int)


# ============================================================
# MAIN
# ============================================================

def main():
    df = pd.read_csv(DATA_PATH)
    df = add_features(df)

    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_order = joblib.load(FEATURE_ORDER_PATH)

    X = build_model_matrix(df, encoder, scaler, feature_order)
    y = df[TARGET].astype(int)

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

    sample_df = df.sample(
        n=min(len(df), NN_PERMUTATION_SAMPLE_SIZE),
        random_state=RANDOM_STATE
    )

    X_sample = build_model_matrix(sample_df, encoder, scaler, feature_order)
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