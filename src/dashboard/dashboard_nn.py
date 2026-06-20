import sys
from pathlib import Path

# (streamlit run src/dashboard/dashboard_nn.py)
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model

from src.common.artifacts import load_artifacts, prepare_input

RUN_NAME = "train_new_dataset_op"
MODEL_DIR = BASE_DIR / "models" / RUN_NAME

model = load_model(MODEL_DIR / f"neural_network_{RUN_NAME}.keras")
artifacts = load_artifacts(MODEL_DIR, RUN_NAME)
encoder = artifacts["encoder"]


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

# same prepare_input as the API
final_input = prepare_input(input_df, artifacts)


if st.button("Tahmini Gör"):
    prob = float(model.predict(final_input)[0][0])
    pred = int(prob > 0.5)

    st.markdown(f"### Saat: {hour} - Şehir: {city}")

    if pred == 1:
        st.error(f"1 saat içinde kaza olabilir. Olasılık: %{prob * 100:.1f}")
    else:
        st.success(f"Kaza beklenmiyor. Olasılık: %{prob * 100:.1f}")
