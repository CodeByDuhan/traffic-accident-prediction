import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Ayarlar
DATA_PATH = BASE_DIR / "data" / "processed" / "augmented_final_dataset_t+1.csv"
FEATURE = "is_saganak"
OUTPUT_PATH = BASE_DIR / "reports" / "results_op2" / f"nn_shap_dependence_{FEATURE}.png"

# Kategorik ve numerik kolonlar
cat_cols = ["city", "yağış_türü"]
num_cols_all = [
    "hour", "temperature_2m", "windspeed_10m", "precipitation",
    "is_night", "dangerous_temp", "high_wind", "risky_rain", "is_saganak"
]
num_cols_to_scale = num_cols_all.copy()
num_cols_to_scale.remove("is_saganak")

# Load data
df = pd.read_csv(DATA_PATH)
df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)
df["dangerous_temp"] = ((df["temperature_2m"] < 0) | (df["temperature_2m"] > 35)).astype(int)
df["high_wind"] = (df["windspeed_10m"] > 20).astype(int)
df["risky_rain"] = 0.0
df.loc[(df["precipitation"] > 0) & (df["precipitation"] <= 2), "risky_rain"] = 1.0
df.loc[(df["precipitation"] > 2) & (df["precipitation"] <= 5), "risky_rain"] = 0.5
df.loc[df["precipitation"] > 5, "risky_rain"] = 0.0
df["is_saganak"] = df["is_saganak"].astype(int)

# Sample
df = df.sample(1100, random_state=42)

# Encoder / Scaler / Order
encoder = joblib.load(BASE_DIR / "models" / "op2" / "encoder_op2.pkl")
scaler = joblib.load(BASE_DIR / "models" / "op2" / "scaler_op2.pkl")
feature_order = joblib.load(BASE_DIR / "models" / "op2" / "feature_order_op2.pkl")

# Encode kategorik
X_cat = pd.DataFrame(
    encoder.transform(df[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols),
    index=df.index
)

# Scale numeric (is_saganak hariç)
X_num_scaled = pd.DataFrame(
    scaler.transform(df[num_cols_to_scale]),
    columns=num_cols_to_scale,
    index=df.index
)

X_num_scaled["is_saganak"] = df["is_saganak"].values

# Final X
X = pd.concat([X_num_scaled, X_cat], axis=1)
X = X[feature_order]

# Model yükle
model = load_model(BASE_DIR / "models" / "op2" / "neural_network_op2.keras")
X_np = X.values.astype(np.float32)
background = X_np[:50]

# Predict fonksiyonu
def model_fn(x):
    return model.predict(x).flatten()

# SHAP hesapla
explainer = shap.KernelExplainer(model_fn, background)
shap_values = explainer.shap_values(X_np)

# SHAP hesaplandıktan sonra:
shap_array = np.array(shap_values)

# aldığım bir hata sonucu kontrol amaçlı shape değerleri aynı mı görmek istedim
print("SHAP shape:", shap_array.shape)
print("X shape:", X.shape)

# Eğer sadece tek çıktı varsa ve shap_array şekli (100, 1) ise reshape et:
if shap_array.ndim == 2 and shap_array.shape[1] == 1:
    shap_array = shap_array.reshape(-1)

# Dependence plot
shap.dependence_plot(
    FEATURE,
    shap_array,
    X,
    show=False,
    interaction_index=None  # interaction_index'i kapattık plot için
)
plt.title(f"Neural Network SHAP Dependence - {FEATURE}")
plt.tight_layout()
plt.savefig(OUTPUT_PATH)
plt.show()

