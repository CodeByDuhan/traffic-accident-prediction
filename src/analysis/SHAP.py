"""
SHAP dependence plots WITH interaction coloring.

Goal: prove that the engineered context features actually modulate how the
model reacts to rain. We plot precipitation's SHAP value colored by
heavy_rain_safe_context / heavy_rain_risky_context. If the two colors separate
(same rain, different SHAP depending on context), the context features are
doing real work the raw precipitation value cannot do on its own.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from src.common.features import add_features
from src.common.artifacts import load_artifacts, build_model_matrix


# ============================================================
# PATH SETTINGS
# ============================================================
DATA_PATH = BASE_DIR / "data" / "processed" / "test_corrected_augmented_dataset_t+1.csv"
MODEL_DIR = BASE_DIR / "models" / "train_new_dataset_op"
FIGURE_DIR = BASE_DIR / "reports" / "figures" / "op" / "interaction"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RUN_NAME = "train_new_dataset_op"
XGB_MODEL_PATH = MODEL_DIR / f"xgboost_{RUN_NAME}.pkl"
NN_MODEL_PATH = MODEL_DIR / f"neural_network_{RUN_NAME}.keras"


# ============================================================
# CONFIG
# ============================================================
RANDOM_STATE = 42

# TreeExplainer is exact + fast, so we can afford a big sample for XGBoost.
XGB_SAMPLE_SIZE = 20000
# KernelExplainer for the NN is slow -> small sample + small background.
NN_SAMPLE_SIZE = 400
NN_BACKGROUND_SIZE = 50

# Which (main feature, interaction feature) pairs to draw.
# The main feature is on the x-axis; the interaction feature colors the points.
INTERACTION_PAIRS = [
    ("precipitation", "heavy_rain_safe_context"),
    ("precipitation", "heavy_rain_risky_context"),
    ("precipitation", "is_night"),
    ("precipitation", "high_wind"),
    ("is_saganak", "heavy_rain_safe_context"),
    ("is_saganak", "heavy_rain_risky_context"),
]


# ============================================================
# BALANCED SAMPLING
# saganak (rain>5) rows are rare, and the context flags only fire on those
# rows. A plain random sample would be ~all no-rain points and the
# interaction would be invisible. So we oversample saganak rows.
# ============================================================
def balanced_sample(df, n_total, random_state):
    rain = df[df["precipitation"] > 5]
    dry = df[df["precipitation"] <= 5]

    n_rain = min(len(rain), n_total // 2)
    n_dry = min(len(dry), n_total - n_rain)

    parts = [
        rain.sample(n=n_rain, random_state=random_state),
        dry.sample(n=n_dry, random_state=random_state),
    ]
    return pd.concat(parts).sample(frac=1, random_state=random_state)


def save_dependence(feature, interaction, shap_array, X, model_tag):
    if feature not in X.columns or interaction not in X.columns:
        print(f"Skipped {model_tag}: {feature} x {interaction} (missing column)")
        return

    shap.dependence_plot(
        feature,
        shap_array,
        X,
        interaction_index=interaction,   # <-- THE KEY CHANGE: color by context
        show=False,
    )
    plt.title(f"{model_tag} SHAP: {feature} colored by {interaction}")
    plt.tight_layout()
    out = FIGURE_DIR / f"{model_tag.lower()}_dep_{feature}_x_{interaction}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# MAIN
# ============================================================
def main():
    df = pd.read_csv(DATA_PATH)
    df = add_features(df)

    artifacts = load_artifacts(MODEL_DIR, RUN_NAME)
    feature_order = artifacts["feature_order"]

    xgb_model = joblib.load(XGB_MODEL_PATH)
    nn_model = load_model(NN_MODEL_PATH)

    # --------------------------------------------------------
    # XGBoost (TreeExplainer, exact)
    # --------------------------------------------------------
    xgb_df = balanced_sample(df, XGB_SAMPLE_SIZE, RANDOM_STATE)
    X_xgb = build_model_matrix(xgb_df, artifacts)

    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_shap = xgb_explainer.shap_values(X_xgb)
    if isinstance(xgb_shap, list):       # binary classifier -> take positive class
        xgb_shap = xgb_shap[1]

    print("XGB SHAP shape:", np.array(xgb_shap).shape, "| X:", X_xgb.shape)

    for feat, inter in INTERACTION_PAIRS:
        save_dependence(feat, inter, xgb_shap, X_xgb, "XGB")

    # --------------------------------------------------------
    # Neural Network (KernelExplainer, model-agnostic, slow)
    # --------------------------------------------------------
    nn_df = balanced_sample(df, NN_SAMPLE_SIZE, RANDOM_STATE)
    X_nn = build_model_matrix(nn_df, artifacts)
    X_nn_np = X_nn.to_numpy().astype(np.float32)

    background = X_nn_np[:NN_BACKGROUND_SIZE]

    def model_fn(x):
        return nn_model.predict(x, verbose=0).reshape(-1)

    nn_explainer = shap.KernelExplainer(model_fn, background)
    nn_shap = nn_explainer.shap_values(X_nn_np)

    nn_shap_array = np.array(nn_shap)
    if isinstance(nn_shap, list):
        nn_shap_array = np.array(nn_shap[0])
    if nn_shap_array.ndim == 3:
        nn_shap_array = nn_shap_array[:, :, 0]

    print("NN SHAP shape:", nn_shap_array.shape, "| X:", X_nn.shape)

    for feat, inter in INTERACTION_PAIRS:
        save_dependence(feat, inter, nn_shap_array, X_nn, "NN")

    print(f"\nAll interaction dependence plots saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()