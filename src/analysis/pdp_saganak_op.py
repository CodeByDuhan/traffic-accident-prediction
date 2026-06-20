import sys
from pathlib import Path

# make the repo root importable
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import PartialDependenceDisplay

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


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
FEATURE = "is_saganak"

# saganak rows are rare -> balanced sampling per class so the PDP isn't
# dominated by the no-rain majority
SAMPLE_PER_CLASS = 3000


# ============================================================
# MAIN
# ============================================================

def main():
    df = pd.read_csv(DATA_PATH)
    df = add_features(df)

    df_sampled = (
        df.groupby(FEATURE, group_keys=False)
        .apply(lambda x: x.sample(min(len(x), SAMPLE_PER_CLASS), random_state=RANDOM_STATE))
        .sample(frac=1, random_state=RANDOM_STATE)
    )

    artifacts = load_artifacts(MODEL_DIR, RUN_NAME)

    X = build_model_matrix(df_sampled, artifacts)

    model = joblib.load(XGB_MODEL_PATH)

    fig, ax = plt.subplots(figsize=(7, 5))

    PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=[FEATURE],
        ax=ax
    )

    plt.title("XGBoost PDP - is_saganak - OP")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "xgb_pdp_is_saganak_balanced.png",
        dpi=160,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Saganak PDP saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
