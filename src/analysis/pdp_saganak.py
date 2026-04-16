import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


DATA_PATH = BASE_DIR / "data" / "processed" / "augmented_final_dataset_t+1.csv"
OUTPUT_PATH = BASE_DIR / "reports" / "results_op2" / "xgb_pdp_is_saganak.png"
FEATURE = "is_saganak" # bu yapıya göre tüm featureları tek satır değişerek görrebiliriz

# Feature
cat_cols = ["city", "yağış_türü"]
scaled_num_cols = [  # sadece scaler’a girenler
    "hour", "temperature_2m", "windspeed_10m", "precipitation",
    "is_night", "dangerous_temp", "high_wind", "risky_rain"
]
manual_cols = ["is_saganak"]  # scaler’a sokulmadan sonradan eklenecek
target_col = "accident_happened_t+1"


df = pd.read_csv(DATA_PATH)

# özellik müh
df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)
df["dangerous_temp"] = ((df["temperature_2m"] < 0) | (df["temperature_2m"] > 35)).astype(int)
df["high_wind"] = (df["windspeed_10m"] > 20).astype(int)
df["risky_rain"] = 0.0
df.loc[(df["precipitation"] > 0) & (df["precipitation"] <= 2), "risky_rain"] = 1.0
df.loc[(df["precipitation"] > 2) & (df["precipitation"] <= 5), "risky_rain"] = 0.5
df.loc[df["precipitation"] > 5, "risky_rain"] = 0.0
df["is_saganak"] = (df["precipitation"] > 5).astype(int)

# Dengeyi koruyarak örnekle
df_sampled = df.groupby("is_saganak", group_keys=False).apply(lambda x: x.sample(min(len(x), 3000), random_state=42))
df_sampled = df_sampled.sample(frac=1, random_state=42)

X_raw = df_sampled[cat_cols + scaled_num_cols + manual_cols]
y = df_sampled[target_col]

# Encode + scale
encoder = joblib.load(BASE_DIR / "models" / "op2" / "encoder_op2.pkl")
scaler = joblib.load(BASE_DIR / "models" / "op2" / "scaler_op2.pkl")
feature_order = joblib.load(BASE_DIR / "models" / "op2" / "feature_order_op2.pkl")

X_cat = pd.DataFrame(
    encoder.transform(X_raw[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols),
    index=X_raw.index
)
X_scaled_num = pd.DataFrame(
    scaler.transform(X_raw[scaled_num_cols]),
    columns=scaled_num_cols,
    index=X_raw.index
)
X_manual = X_raw[manual_cols]

# Birleştir
X = pd.concat([X_scaled_num, X_manual, X_cat], axis=1)
X = X[feature_order]

# Modeli yükle
model = joblib.load(BASE_DIR / "models" / "op2" / "xgboost_op2.pkl")

# PDP çiz
fig, ax = plt.subplots(figsize=(6, 4))
PartialDependenceDisplay.from_estimator(
    model, X, features=[FEATURE], ax=ax
)
plt.title("XGBoost PDP - is_saganak")
plt.tight_layout()
plt.savefig(OUTPUT_PATH)
plt.show()
