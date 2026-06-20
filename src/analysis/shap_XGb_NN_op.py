import sys
from pathlib import Path

# make the repo root importable
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
FIGURE_DIR = BASE_DIR / "reports" / "figures" / "op"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RUN_NAME = "train_new_dataset_op"

XGB_MODEL_PATH = MODEL_DIR / f"xgboost_{RUN_NAME}.pkl"
NN_MODEL_PATH = MODEL_DIR / f"neural_network_{RUN_NAME}.keras"


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

# TreeExplainer is fast -> 20k rows ok. KernelExplainer for the NN is
# model-agnostic and slow, so a much smaller sample
XGB_SHAP_SAMPLE_SIZE = 20000
NN_SHAP_SAMPLE_SIZE = 250
NN_BACKGROUND_SIZE = 50


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

    artifacts = load_artifacts(MODEL_DIR, RUN_NAME)
    feature_order = artifacts["feature_order"]

    xgb_model = joblib.load(XGB_MODEL_PATH)
    nn_model = load_model(NN_MODEL_PATH)

    # --------------------------------------------------------
    # XGBoost SHAP
    # --------------------------------------------------------

    xgb_df = df.sample(
        n=min(len(df), XGB_SHAP_SAMPLE_SIZE),
        random_state=RANDOM_STATE
    )

    X_xgb = build_model_matrix(xgb_df, artifacts)

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

    X_nn = build_model_matrix(nn_df, artifacts)
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
