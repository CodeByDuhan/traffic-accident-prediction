import sys
from pathlib import Path

# make the repo root importable
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

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

NN_MODEL_PATH = MODEL_DIR / f"neural_network_{RUN_NAME}.keras"


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
SAMPLE_SIZE = 250
BACKGROUND_SIZE = 50

FEATURES_TO_PLOT = [
    "precipitation",
    "is_saganak",
    "rain_2_5",
    "rain_5_10",
    "rain_gt_10",
    "heavy_rain_safe_context",
    "heavy_rain_risky_context",
    "heavy_rain_high_wind",
    "heavy_rain_night"
]


# ============================================================
# MAIN
# ============================================================

def main():
    df = pd.read_csv(DATA_PATH)
    df = add_features(df)

    df = df.sample(
        n=min(len(df), SAMPLE_SIZE),
        random_state=RANDOM_STATE
    )

    artifacts = load_artifacts(MODEL_DIR, RUN_NAME)

    X = build_model_matrix(df, artifacts)
    X_np = X.to_numpy().astype(np.float32)

    model = load_model(NN_MODEL_PATH)

    background = X_np[:BACKGROUND_SIZE]

    def model_fn(x):
        return model.predict(x, verbose=0).reshape(-1)

    explainer = shap.KernelExplainer(model_fn, background)
    shap_values = explainer.shap_values(X_np)

    shap_array = np.array(shap_values)

    if isinstance(shap_values, list):
        shap_array = np.array(shap_values[0])

    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]

    print("NN SHAP shape:", shap_array.shape)
    print("X shape:", X.shape)

    for feature in FEATURES_TO_PLOT:
        if feature not in X.columns:
            print(f"Atlandı: {feature} feature_order içinde yok.")
            continue

        shap.dependence_plot(
            feature,
            shap_array,
            X,
            show=False,
            interaction_index=None
        )

        plt.title(f"Neural Network SHAP Dependence - {feature} - OP")
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / f"nn_shap_dependence_{feature}.png",
            dpi=160,
            bbox_inches="tight"
        )
        plt.close()

    print(f"NN SHAP dependence plots saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
