import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

RUN_NAME = "train_new_dataset_op"
MODEL_DIR = BASE_DIR / "models" / RUN_NAME

model = load_model(MODEL_DIR / f"neural_network_{RUN_NAME}.keras")
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


# ------------------
# Kullanıcı Arayüzü
# ------------------

st.title("Saatlik Kaza Tahmin Sistemi - Neural Network")

city = st.selectbox("Şehir", encoder.categories_[0])

hour = st.slider("Saat (0–23)", min_value=0, max_value=23, value=12)
temperature = st.number_input("Sıcaklık (°C)", value=20.0, step=0.5)
windspeed = st.number_input("Rüzgar Hızı (km/saat)", value=10.0, step=0.5)
precipitation = st.number_input("Yağış Miktarı (mm)", value=0.0, step=0.1)

if precipitation == 0:
    rain_type = "yok"
    st.info("Yağış miktarı 0 mm olduğu için yağış türü 'yok' olarak ayarlandı.")
else:
    rain_type = st.selectbox("Yağış Türü", encoder.categories_[1])


input_dict = {
    "city": city,
    "yağış_türü": rain_type,
    "hour": hour,
    "temperature_2m": temperature,
    "windspeed_10m": windspeed,
    "precipitation": precipitation
}

input_df = pd.DataFrame([input_dict])
final_input = prepare_input(input_df)


if st.button("Tahmini Gör"):
    prob = float(model.predict(final_input)[0][0])
    pred = int(prob > 0.5)

    st.markdown(f"### Saat: {hour} - Şehir: {city}")

    if pred == 1:
        st.error(f"1 saat içinde kaza olabilir. Olasılık: %{prob * 100:.1f}")
    else:
        st.success(f"Kaza beklenmiyor. Olasılık: %{prob * 100:.1f}")