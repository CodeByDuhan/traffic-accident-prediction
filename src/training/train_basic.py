import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


df = pd.read_csv(BASE_DIR / "data" / "processed" / "augmented_final_dataset_t+1.csv")

# özellik müh
df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)
df["dangerous_temp"] = ((df["temperature_2m"] < 0) | (df["temperature_2m"] > 35)).astype(int)
df["high_wind"] = (df["windspeed_10m"] > 20).astype(int)

# yağmur miktarına göre kaza riski
df["risky_rain"] = 0.0
df.loc[(df["precipitation"] > 0) & (df["precipitation"] <= 2), "risky_rain"] = 1
df.loc[(df["precipitation"] > 2) & (df["precipitation"] <= 5), "risky_rain"] = 0.5
df.loc[df["precipitation"] > 5, "risky_rain"] = 0.0


df["is_saganak"] = df["is_saganak"].astype(int)

# Hedef ve feature sütunları
target = "accident_happened_t+1"
cat_cols = ["city", "yağış_türü"]
num_cols = [
    "hour", "temperature_2m", "windspeed_10m", "precipitation",
    "is_night", "dangerous_temp", "high_wind", "risky_rain"

]# ️ is_saganak'ı normalize etmedik 1,0 0çok

# Encoding & Scaling
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = pd.DataFrame(encoder.fit_transform(df[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))

scaler = StandardScaler()
X_num = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

# (2 milyon 0, sadece 16 bin 1)
# Normalize edilirse SHAP değerleri sapıtıyor
X_num["is_saganak"] = df["is_saganak"].values

# final feature matrisi
X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
y = df[target]

# Train/Test bölme %20 olarak ayırdım
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# klasör
os.makedirs(BASE_DIR / "models" / "basic", exist_ok=True)
os.makedirs(BASE_DIR / "reports" / "results_basic", exist_ok=True)

results = []

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train)
joblib.dump(xgb, BASE_DIR / "models" / "basic" / "xgboost.pkl")
results.append({
    "model": "XGBoost",
    "accuracy": accuracy_score(y_test, xgb.predict(X_test)),
    "precision": precision_score(y_test, xgb.predict(X_test)),
    "recall": recall_score(y_test, xgb.predict(X_test)),
    "confusion_matrix": confusion_matrix(y_test, xgb.predict(X_test)).tolist()
})

# random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, BASE_DIR / "models" / "basic" / "random_forest.pkl")
results.append({
    "model": "Random Forest",
    "accuracy": accuracy_score(y_test, rf.predict(X_test)),
    "precision": precision_score(y_test, rf.predict(X_test)),
    "recall": recall_score(y_test, rf.predict(X_test)),
    "confusion_matrix": confusion_matrix(y_test, rf.predict(X_test)).tolist()
})

# neural Network
nn = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])


nn.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    verbose=1
)

nn.save(BASE_DIR / "models" / "basic" / "neural_network.keras")
y_pred_nn = (nn.predict(X_test) > 0.5).astype(int)

results.append({
    "model": "Neural Network",
    "accuracy": accuracy_score(y_test, y_pred_nn),
    "precision": precision_score(y_test, y_pred_nn),
    "recall": recall_score(y_test, y_pred_nn),
    "confusion_matrix": confusion_matrix(y_test, y_pred_nn).tolist()
})

# sonuçları kaydet
results_df = pd.DataFrame(results)
results_df.to_csv(BASE_DIR / "reports" / "results_basic" / "basic_model_results.csv", index=False)

# encoder, scaler ve feature sıralaması
joblib.dump(encoder, BASE_DIR / "models" / "basic" / "encoder.pkl")
joblib.dump(scaler, BASE_DIR / "models" / "basic" / "scaler.pkl")
joblib.dump(X.columns.tolist(), BASE_DIR / "models" / "basic" / "feature_order.pkl")
