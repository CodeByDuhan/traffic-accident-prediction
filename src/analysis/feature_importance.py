import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

from sklearn.inspection import permutation_importance
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import load_model

# 1. Load data
df = pd.read_csv(BASE_DIR / "data" / "processed" / "augmented_final_dataset_t+1.csv")

# 2. Feature engineering
df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)
df["dangerous_temp"] = ((df["temperature_2m"] < 0) | (df["temperature_2m"] > 35)).astype(int)
df["high_wind"] = (df["windspeed_10m"] > 20).astype(int)

df["risky_rain"] = 0.0
df.loc[(df["precipitation"] > 0) & (df["precipitation"] <= 2), "risky_rain"] = 1.0
df.loc[(df["precipitation"] > 2) & (df["precipitation"] <= 5), "risky_rain"] = 0.5
df.loc[df["precipitation"] > 5, "risky_rain"] = 0.0

# 3. Load tools
encoder = joblib.load(BASE_DIR / "models" / "op2" / "encoder_op2.pkl")
scaler = joblib.load(BASE_DIR / "models" / "op2" / "scaler_op2.pkl")
feature_order = joblib.load(BASE_DIR / "models" / "op2" / "feature_order_op2.pkl")

# 4. Transform features
cat_cols = ["city", "yağış_türü"]
num_cols_scaled = [
    "hour", "temperature_2m", "windspeed_10m", "precipitation",
    "is_night", "dangerous_temp", "high_wind", "risky_rain"
]

X_cat_arr = encoder.transform(df[cat_cols])
X_cat_cols = encoder.get_feature_names_out(cat_cols)
X_cat = pd.DataFrame(X_cat_arr, columns=X_cat_cols, index=df.index)

X_num_arr = scaler.transform(df[num_cols_scaled])
X_num = pd.DataFrame(X_num_arr, columns=num_cols_scaled, index=df.index)

# is_saganak normalize edilmeden ekleniyor
X_num["is_saganak"] = df["is_saganak"].astype(int).values

# Final X
X = pd.concat([X_num, X_cat], axis=1)
X = X[feature_order]

# Y
y = df["accident_happened_t+1"]

# Sample
X_sample = X.sample(5000, random_state=42)
y_sample = y.loc[X_sample.index]

# 5. Load models
xgb_model = joblib.load(BASE_DIR / "models" / "op2" / "xgboost_op2.pkl")
rf_model = joblib.load(BASE_DIR / "models" / "op2" / "random_forest_op2.pkl")
nn_model = load_model(BASE_DIR / "models" / "op2" / "neural_network_op2.keras")

# -------------------------
# Feature Importance - XGBoost
xgb_importances = xgb_model.feature_importances_
indices = np.argsort(xgb_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("XGBoost Feature Importance")
plt.bar(range(len(feature_order)), xgb_importances[indices])
plt.xticks(range(len(feature_order)), np.array(feature_order)[indices], rotation=90)
plt.tight_layout()
plt.savefig(BASE_DIR / "reports" / "results_op2" / "xgb_feature_importance.png")
plt.close()

# -------------------------
# Feature Importance - Random Forest
rf_importances = rf_model.feature_importances_
indices = np.argsort(rf_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Random Forest Feature Importance")
plt.bar(range(len(feature_order)), rf_importances[indices])
plt.xticks(range(len(feature_order)), np.array(feature_order)[indices], rotation=90)
plt.tight_layout()
plt.savefig(BASE_DIR / "reports" / "results_op2" / "rf_feature_importance.png")
plt.close()

# -------------------------
# Permutation Importance - Neural Network
wrapped_model = KerasClassifier(model=nn_model, verbose=0)

# Dummy fit (gerekiyor yoksa predict edemez)
wrapped_model.fit(X_sample.to_numpy(), y_sample.to_numpy(), epochs=0)

perm_result = permutation_importance(
    estimator=wrapped_model,
    X=X_sample.to_numpy(),
    y=y_sample.to_numpy(),
    n_repeats=10,
    scoring="accuracy",
    random_state=42
)

sorted_idx = perm_result.importances_mean.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.title("Neural Network Permutation Importance")
plt.bar(range(len(feature_order)), perm_result.importances_mean[sorted_idx])
plt.xticks(range(len(feature_order)), np.array(feature_order)[sorted_idx], rotation=90)
plt.tight_layout()
plt.savefig(BASE_DIR / "reports" / "results_op2" / "nn_permutation_importance.png")
plt.close()
