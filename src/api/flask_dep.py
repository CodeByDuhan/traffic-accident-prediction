import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

import joblib
import pandas as pd
from flask import Flask, request, jsonify

from src.common.artifacts import load_artifacts, prepare_input

RUN_NAME = "train_new_dataset_op"
MODEL_DIR = BASE_DIR / "models" / RUN_NAME

app = Flask(__name__)

# deployed model: XGBoost from the optimized run. picked over the NN because
# of better heavy-rain bin behavior, not just global scores
model = joblib.load(MODEL_DIR / f"xgboost_{RUN_NAME}.pkl")

# encoder/scaler/feature_order saved at training time -> serving applies
# the exact same transform as training
artifacts = load_artifacts(MODEL_DIR, RUN_NAME)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_df = pd.DataFrame([data])

        # raw 6-field request -> engineered features -> model matrix,
        # all handled by the shared prepare_input
        final_input = prepare_input(input_df, artifacts)

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
