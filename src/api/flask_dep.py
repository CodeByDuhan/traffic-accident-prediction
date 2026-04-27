from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

RUN_NAME = "train_new_dataset_op"
MODEL_DIR = BASE_DIR / "models" / RUN_NAME

app = Flask(__name__)

model = joblib.load(MODEL_DIR / f"xgboost_{RUN_NAME}.pkl")
encoder = joblib.load(MODEL_DIR / f"encoder_{RUN_NAME}.pkl")
scaler = joblib.load(MODEL_DIR / f"scaler_{RUN_NAME}.pkl")
feature_order = joblib.load(MODEL_DIR / f"feature_order_{RUN_NAME}.pkl")
cat_cols = joblib.load(MODEL_DIR / f"cat_cols_{RUN_NAME}.pkl")
num_cols = joblib.load(MODEL_DIR / f"num_cols_{RUN_NAME}.pkl")
binary_cols = joblib.load(MODEL_DIR / f"binary_cols_{RUN_NAME}.pkl")


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

    df["rain_gt_10"] = (
        df["precipitation"] > 10
    ).astype(int)

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


def prepare_input(input_df: pd.DataFrame) -> pd.DataFrame:
    input_df = add_features(input_df)

    missing_cols = [
        col for col in cat_cols + num_cols + binary_cols
        if col not in input_df.columns
    ]

    if missing_cols:
        raise ValueError(f"Eksik input kolonları: {missing_cols}")

    X_cat = encoder.transform(input_df[cat_cols])
    X_num = scaler.transform(input_df[num_cols])
    X_bin = input_df[binary_cols].astype(int).values

    X_final = np.hstack([
        X_num,
        X_bin,
        X_cat
    ])

    final_input = pd.DataFrame(X_final, columns=feature_order)

    return final_input


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_df = pd.DataFrame([data])

        final_input = prepare_input(input_df)

        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)