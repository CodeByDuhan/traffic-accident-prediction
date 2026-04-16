import pandas as pd
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import load_model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


df = pd.read_csv(BASE_DIR / "data" / "processed" / "augmented_final_dataset_t+1.csv")


df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)
df["dangerous_temp"] = ((df["temperature_2m"] < 0) | (df["temperature_2m"] > 35)).astype(int)
df["high_wind"] = (df["windspeed_10m"] > 20).astype(int)

df["risky_rain"] = 0.0
df.loc[(df["precipitation"] > 0) & (df["precipitation"] <= 2), "risky_rain"] = 1.0
df.loc[(df["precipitation"] > 2) & (df["precipitation"] <= 5), "risky_rain"] = 0.5
df.loc[df["precipitation"] > 5, "risky_rain"] = 0.0

encoder = joblib.load(BASE_DIR / "models" / "op2" / "encoder_op2.pkl")
scaler = joblib.load(BASE_DIR / "models" / "op2" / "scaler_op2.pkl")
feature_order = joblib.load(BASE_DIR / "models" / "op2" / "feature_order_op2.pkl")


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

X = pd.concat([X_num, X_cat], axis=1)
X = X[feature_order]  # SHAP uyumlu


X_sample = X.sample(5000, random_state=42)


xgb_model = joblib.load(BASE_DIR / "models" / "op2" / "xgboost_op2.pkl")
rf_model = joblib.load(BASE_DIR / "models" / "op2" / "random_forest_op2.pkl")
nn_model = load_model(BASE_DIR / "models" / "op2" / "neural_network_op2.keras")

# -------------------------
#  SHAP - XGBoost
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values_xgb = explainer_xgb.shap_values(X_sample)

# Liste mi değil mi kontrolü (bazı sürümlerde shap_values[1] gerekir)
if isinstance(shap_values_xgb, list):
    shap_vals_xgb_plot = shap_values_xgb[1]  # Class 1
else:
    shap_vals_xgb_plot = shap_values_xgb     # Direct array

shap.summary_plot(
    shap_vals_xgb_plot,
    features=X_sample,
    feature_names=feature_order,
    plot_type="bar",
    show=False
)
plt.title("XGBoost SHAP Importance")
plt.savefig(BASE_DIR / "reports" / "results_op2" / "xgb_shap_genel_özet.png")
plt.close()



#  SHAP - Neural Network
X_nn_sample = X_sample.sample(100, random_state=42).copy()
X_nn_sample = X_nn_sample[feature_order]
X_nn_np = X_nn_sample.to_numpy().astype(np.float32)
background = X_nn_np[:50]

def model_fn(X_numpy):
    return nn_model.predict(X_numpy)

explainer_nn = shap.KernelExplainer(model_fn, background)
shap_values_nn = explainer_nn.shap_values(X_nn_np)

print("SHAP shape:", np.array(shap_values_nn).shape)
print("Data shape:", X_nn_sample.shape)

shap_values = shap_values_nn[0] if isinstance(shap_values_nn, list) else shap_values_nn

shap.summary_plot(
    shap_values,
    features=X_nn_sample,
    feature_names=feature_order,
    plot_type="bar",
    show=False
)
plt.title("Neural Network SHAP Importance")
plt.savefig(BASE_DIR / "reports" / "results_op2" / "nn_shap_genel_özet.png")
plt.close()



# SHAP - Random Forest
explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf = explainer_rf.shap_values(X_sample)

# class count kontrolü
if isinstance(shap_values_rf, list) and len(shap_values_rf) == 2:
    rf_shap_plot_vals = shap_values_rf[1]  # class 1
else:
    rf_shap_plot_vals = shap_values_rf    # tek sınıf varsa direkt array

print("RF SHAP shape:", np.array(rf_shap_plot_vals).shape)
print("X_sample shape:", X_sample.shape)

shap.summary_plot(
    rf_shap_plot_vals,
    features=X_sample,
    feature_names=feature_order,
    plot_type="bar",
    show=False
)
plt.title("Random Forest SHAP Importance")
plt.savefig(BASE_DIR / "reports" / "results_op2" / "rf_shap_genel_özet.png")
plt.close()

