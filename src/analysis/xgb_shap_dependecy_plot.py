import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


DATA_PATH = BASE_DIR / "data" / "processed" / "augmented_final_dataset_t+1.csv"
FEATURE = "is_saganak"
OUTPUT_PATH = BASE_DIR / "reports" / "results_op2" / f"xgb_shap_dependence_{FEATURE}.png"

# Featurlar
cat_cols = ["city", "yağış_türü"]
num_cols_all = [
    "hour", "temperature_2m", "windspeed_10m", "precipitation",
    "is_night", "dangerous_temp", "high_wind", "risky_rain", "is_saganak"
]
target_col = "accident_happened_t+1"

# Veri + Feature Engineering
df = pd.read_csv(DATA_PATH)
df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)
df["dangerous_temp"] = ((df["temperature_2m"] < 0) | (df["temperature_2m"] > 35)).astype(int)
df["high_wind"] = (df["windspeed_10m"] > 20).astype(int)
df["risky_rain"] = 0.0
df.loc[(df["precipitation"] > 0) & (df["precipitation"] <= 2), "risky_rain"] = 1.0
df.loc[(df["precipitation"] > 2) & (df["precipitation"] <= 5), "risky_rain"] = 0.5
df.loc[df["precipitation"] > 5, "risky_rain"] = 0.0

# is_saganak tür dönüşümü
df["is_saganak"] = df["is_saganak"].astype(int)

# Sample küçültüyoruz yine
df = df.sample(50000, random_state=42)

# Encode + Scale
encoder = joblib.load(BASE_DIR / "models" / "op2" / "encoder_op2.pkl")
scaler = joblib.load(BASE_DIR / "models" / "op2" / "scaler_op2.pkl")
feature_order = joblib.load(BASE_DIR / "models" / "op2" / "feature_order_op2.pkl")

# normalize edilecek numerik sütunlar
num_cols_to_scale = [
    "hour", "temperature_2m", "windspeed_10m", "precipitation",
    "is_night", "dangerous_temp", "high_wind", "risky_rain"
]

# Encode kategorik
X_cat = pd.DataFrame(
    encoder.transform(df[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols),
    index=df.index
)

# Scale numeric is_saganak hariç
X_num_scaled = pd.DataFrame(
    scaler.transform(df[num_cols_to_scale]),
    columns=num_cols_to_scale,
    index=df.index
)

# is_saganak normalize edilmeden
X_num_scaled["is_saganak"] = df["is_saganak"].values

# X matrisi
X = pd.concat([X_num_scaled, X_cat], axis=1)
X = X[feature_order]

#  Model ve SHAP
model = joblib.load(BASE_DIR / "models" / "op2" / "xgboost_op2.pkl")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

#  Dependence plot
shap.dependence_plot(
    FEATURE,
    shap_values,
    X,
    show=False
)
plt.title(f"XGBoost SHAP Dependence - {FEATURE}")
plt.tight_layout()
plt.savefig(OUTPUT_PATH)
plt.show()
