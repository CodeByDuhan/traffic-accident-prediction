import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import load_model


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

XGB_SHAP_SAMPLE_SIZE = 20000
NN_SHAP_SAMPLE_SIZE = 250
NN_BACKGROUND_SIZE = 50


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(df):
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


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    df = pd.read_csv(DATA_PATH)
    df = add_features(df)

    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_order = joblib.load(FEATURE_ORDER_PATH)

    xgb_model = joblib.load(XGB_MODEL_PATH)
    nn_model = load_model(NN_MODEL_PATH)

    # --------------------------------------------------------
    # XGBoost SHAP
    # --------------------------------------------------------

    xgb_df = df.sample(
        n=min(len(df), XGB_SHAP_SAMPLE_SIZE),
        random_state=RANDOM_STATE
    )

    X_xgb = build_model_matrix(xgb_df, encoder, scaler, feature_order)

    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_shap_values = xgb_explainer.shap_values(X_xgb)

    if isinstance(xgb_shap_values, list):
        xgb_shap_values = xgb_shap_values[1]

    shap.summary_plot(
        xgb_shap_values,
        features=X_xgb,
        feature_names=feature_order,
        plot_type="bar",
        show=False,
        max_display=30
    )
    plt.title("XGBoost SHAP Summary Bar - Unweighted")
    save_plot(FIGURE_DIR / "xgb_shap_summary_bar.png")

    shap.summary_plot(
        xgb_shap_values,
        features=X_xgb,
        feature_names=feature_order,
        show=False,
        max_display=30
    )
    plt.title("XGBoost SHAP Summary Dot - Unweighted")
    save_plot(FIGURE_DIR / "xgb_shap_summary_dot.png")

    # --------------------------------------------------------
    # Neural Network SHAP
    # --------------------------------------------------------

    nn_df = df.sample(
        n=min(len(df), NN_SHAP_SAMPLE_SIZE),
        random_state=RANDOM_STATE
    )

    X_nn = build_model_matrix(nn_df, encoder, scaler, feature_order)
    X_nn_np = X_nn.to_numpy().astype(np.float32)

    background = X_nn_np[:NN_BACKGROUND_SIZE]

    def model_fn(x):
        return nn_model.predict(x, verbose=0).reshape(-1)

    nn_explainer = shap.KernelExplainer(model_fn, background)
    nn_shap_values = nn_explainer.shap_values(X_nn_np)

    nn_shap_array = np.array(nn_shap_values)

    if isinstance(nn_shap_values, list):
        nn_shap_array = np.array(nn_shap_values[0])

    if nn_shap_array.ndim == 3:
        nn_shap_array = nn_shap_array[:, :, 0]

    print("NN SHAP shape:", nn_shap_array.shape)
    print("NN X shape:", X_nn.shape)

    shap.summary_plot(
        nn_shap_array,
        features=X_nn,
        feature_names=feature_order,
        plot_type="bar",
        show=False,
        max_display=30
    )
    plt.title("Neural Network SHAP Summary Bar - OP")
    save_plot(FIGURE_DIR / "nn_shap_summary_bar.png")

    print(f"SHAP summary figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()