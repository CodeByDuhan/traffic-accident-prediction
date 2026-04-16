import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


DATA_PATH = BASE_DIR / "data" / "processed" / "augmented_final_dataset_t+1.csv"
OUTPUT_PATH = BASE_DIR / "reports" / "results_op2" / "xgb_pdp.png"

# feature tanımı
cat_cols = ["city", "yağış_türü"]
num_cols = [
    "hour", "temperature_2m", "windspeed_10m", "precipitation",
    "is_night", "dangerous_temp", "high_wind", "risky_rain"
]
extra_feature = ["is_saganak"]  # scaler'a girmeyecek ama modele giriyor

target_col = "accident_happened_t+1"
pdp_features = num_cols + extra_feature  # PDP'de is_saganak'ı da göster

#  Veriyi küçültülmüş örnekle verdim çünkü 2 milyon satır zaman alıyor.
df = pd.read_csv(DATA_PATH)
df = df.sample(50000, random_state=42)

# özellik müh
df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)
df["dangerous_temp"] = ((df["temperature_2m"] < 0) | (df["temperature_2m"] > 35)).astype(int)
df["high_wind"] = (df["windspeed_10m"] > 20).astype(int)
df["risky_rain"] = 0.0
df.loc[(df["precipitation"] > 0) & (df["precipitation"] <= 2), "risky_rain"] = 1.0
df.loc[(df["precipitation"] > 2) & (df["precipitation"] <= 5), "risky_rain"] = 0.5
df.loc[df["precipitation"] > 5, "risky_rain"] = 0.0
df["is_saganak"] = (df["precipitation"] > 5).astype(int)

# Encode + Scale
encoder = joblib.load(BASE_DIR / "models" / "op2" / "encoder_op2.pkl")
scaler = joblib.load(BASE_DIR / "models" / "op2" / "scaler_op2.pkl")
feature_order = joblib.load(BASE_DIR / "models" / "op2" / "feature_order_op2.pkl")

X_raw = df[cat_cols + num_cols + extra_feature]
y = df[target_col]

X_cat = pd.DataFrame(
    encoder.transform(X_raw[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols),
    index=X_raw.index
)

X_num = pd.DataFrame(
    scaler.transform(X_raw[num_cols]),
    columns=num_cols,
    index=X_raw.index
)

# is_saganak'ı scaler'dan geçirmeden ekle
X_num["is_saganak"] = X_raw["is_saganak"].values

# Birleştir
X = pd.concat([X_num, X_cat], axis=1)
X = X[feature_order]

# Modeli yükle
model = joblib.load(BASE_DIR / "models" / "op2" / "xgboost_op2.pkl")

# PDP çiz
fig, ax = plt.subplots(figsize=(14, 10))
PartialDependenceDisplay.from_estimator(
    model, X, features=pdp_features, ax=ax
)
plt.suptitle("XGBoost PDP - Tüm Özellikler", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_PATH)
plt.show()
