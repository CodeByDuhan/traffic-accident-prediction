import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.inspection import PartialDependenceDisplay


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

ENCODER_PATH = MODEL_DIR / "encoder_train_new_dataset_op.pkl"
SCALER_PATH = MODEL_DIR / "scaler_train_new_dataset_op.pkl"
FEATURE_ORDER_PATH = MODEL_DIR / "feature_order_train_new_dataset_op.pkl"


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
FEATURE = "is_saganak"
SAMPLE_PER_CLASS = 3000


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

    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_order = joblib.load(FEATURE_ORDER_PATH)

    X = build_model_matrix(df_sampled, encoder, scaler, feature_order)

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