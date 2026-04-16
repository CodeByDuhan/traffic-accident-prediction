from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


app = Flask(__name__)


model = joblib.load(BASE_DIR / "models" / "op2" / "xgboost_op2.pkl")
encoder = joblib.load(BASE_DIR / "models" / "op2" / "encoder_op2.pkl")
scaler = joblib.load(BASE_DIR / "models" / "op2" / "scaler_op2.pkl")
feature_order = joblib.load(BASE_DIR / "models" / "op2" / "feature_order_op2.pkl")

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # girişleri dataframe'e çevir
    input_df = pd.DataFrame([data])

    # feature engineering
    input_df["is_night"] = input_df["hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)
    input_df["dangerous_temp"] = ((input_df["temperature_2m"] < 0) | (input_df["temperature_2m"] > 35)).astype(int)
    input_df["high_wind"] = (input_df["windspeed_10m"] > 20).astype(int)
    input_df["risky_rain"] = 0.0
    input_df.loc[(input_df["precipitation"] > 0) & (input_df["precipitation"] <= 2), "risky_rain"] = 1.0
    input_df.loc[(input_df["precipitation"] > 2) & (input_df["precipitation"] <= 5), "risky_rain"] = 0.5
    input_df.loc[input_df["precipitation"] > 5, "risky_rain"] = 0.0
    input_df["is_saganak"] = (input_df["precipitation"] > 5).astype(int)

    # kategorik veriyi encode
    cat_cols = ["city", "yağış_türü"]
    encoded = encoder.transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

    # sayısal veriyi scale
    num_cols = ["hour", "temperature_2m", "windspeed_10m", "precipitation",
                "is_night", "dangerous_temp", "high_wind", "risky_rain"]
    scaled = scaler.transform(input_df[num_cols])
    scaled_df = pd.DataFrame(scaled, columns=num_cols)
    scaled_df["is_saganak"] = input_df["is_saganak"].values

    # final input
    final_input = pd.concat([scaled_df, encoded_df], axis=1)
    final_input = final_input.reindex(columns=feature_order, fill_value=0)

    # tahmin
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "probability": round(float(probability), 4)
    })

if __name__ == '__main__':
    app.run(debug=True)
