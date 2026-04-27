import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
TARGET = "accident_happened_t+1"

RUN_NAME = "train_new_dataset_basic"


# ============================================================
# PATH SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_corrected_augmented_dataset_t+1.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test_corrected_augmented_dataset_t+1.csv"

MODEL_DIR = BASE_DIR / "models" / RUN_NAME
REPORT_DIR = BASE_DIR / "reports" / f"results_{RUN_NAME}"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

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


# ============================================================
# REPORT HELPERS
# ============================================================

def print_precipitation_distribution(df: pd.DataFrame, title: str) -> None:
    temp_df = df.copy()

    temp_df["precip_bin"] = pd.cut(
        temp_df["precipitation"],
        bins=[-0.101, 2, 5, 10, 100],
        labels=["0–2mm", "2–5mm", "5–10mm", ">10mm"]
    )

    report = (
        temp_df.groupby("precip_bin", observed=False)[TARGET]
        .agg(
            total="count",
            positive="sum",
            positive_ratio="mean"
        )
        .reset_index()
    )

    print(f"\n=== {title} ===")
    print(report)


def evaluate_global_metrics(model_name: str, y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "run_name": RUN_NAME,
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = None

    return metrics


def evaluate_precipitation_bins(model_name: str, test_df: pd.DataFrame, y_true, y_pred, y_proba=None) -> pd.DataFrame:
    eval_df = test_df.copy()
    eval_df["y_true"] = np.asarray(y_true)
    eval_df["y_pred"] = np.asarray(y_pred)

    if y_proba is not None:
        eval_df["y_proba"] = np.asarray(y_proba)

    eval_df["precip_bin"] = pd.cut(
        eval_df["precipitation"],
        bins=[-0.101, 2, 5, 10, 100],
        labels=["0–2mm", "2–5mm", "5–10mm", ">10mm"]
    )

    rows = []

    for precip_bin, group in eval_df.groupby("precip_bin", observed=False):
        if group.empty:
            continue

        row = {
            "run_name": RUN_NAME,
            "model": model_name,
            "precip_bin": str(precip_bin),
            "total": len(group),
            "positive": int(group["y_true"].sum()),
            "true_positive_ratio": group["y_true"].mean(),
            "predicted_positive": int(group["y_pred"].sum()),
            "predicted_positive_ratio": group["y_pred"].mean(),
            "accuracy": accuracy_score(group["y_true"], group["y_pred"]),
            "precision": precision_score(group["y_true"], group["y_pred"], zero_division=0),
            "recall": recall_score(group["y_true"], group["y_pred"], zero_division=0),
            "f1": f1_score(group["y_true"], group["y_pred"], zero_division=0),
            "confusion_matrix": confusion_matrix(group["y_true"], group["y_pred"]).tolist()
        }

        if y_proba is not None:
            row["mean_predicted_probability"] = group["y_proba"].mean()

        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_context_bins(model_name: str, test_df: pd.DataFrame, y_true, y_pred, y_proba=None) -> pd.DataFrame:
    eval_df = test_df.copy()
    eval_df["y_true"] = np.asarray(y_true)
    eval_df["y_pred"] = np.asarray(y_pred)

    if y_proba is not None:
        eval_df["y_proba"] = np.asarray(y_proba)

    context_cols = [
        "heavy_rain_safe_context",
        "heavy_rain_risky_context",
        "heavy_rain_high_wind",
        "heavy_rain_night"
    ]

    rows = []

    for col in context_cols:
        for value in [0, 1]:
            group = eval_df[eval_df[col] == value]

            if group.empty:
                continue

            row = {
                "run_name": RUN_NAME,
                "model": model_name,
                "context_feature": col,
                "context_value": value,
                "total": len(group),
                "positive": int(group["y_true"].sum()),
                "true_positive_ratio": group["y_true"].mean(),
                "predicted_positive": int(group["y_pred"].sum()),
                "predicted_positive_ratio": group["y_pred"].mean(),
                "accuracy": accuracy_score(group["y_true"], group["y_pred"]),
                "precision": precision_score(group["y_true"], group["y_pred"], zero_division=0),
                "recall": recall_score(group["y_true"], group["y_pred"], zero_division=0),
                "f1": f1_score(group["y_true"], group["y_pred"], zero_division=0),
                "confusion_matrix": confusion_matrix(group["y_true"], group["y_pred"]).tolist()
            }

            if y_proba is not None:
                row["mean_predicted_probability"] = group["y_proba"].mean()

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df = add_features(train_df)
test_df = add_features(test_df)

print("\nDataset loaded.")
print(f"Train shape: {train_df.shape}")
print(f"Test shape : {test_df.shape}")

print_precipitation_distribution(train_df, "TRAIN PRECIPITATION DISTRIBUTION")
print_precipitation_distribution(test_df, "TEST PRECIPITATION DISTRIBUTION")


# ============================================================
# FEATURE COLUMNS
# ============================================================

cat_cols = [
    "city",
    "yağış_türü"
]

num_cols = [
    "hour",
    "temperature_2m",
    "windspeed_10m",
    "precipitation"
]

binary_cols = [
    "is_night",
    "dangerous_temp",
    "high_wind",
    "is_saganak",

    "rain_0_2",
    "rain_2_5",
    "rain_5_10",
    "rain_gt_10",

    "heavy_rain_safe_context",
    "heavy_rain_risky_context",
    "heavy_rain_high_wind",
    "heavy_rain_night"
]

feature_cols = cat_cols + num_cols + binary_cols

metadata_cols = [
    "source_id",
    "is_augmented",
    "augmentation_type"
]

for col in metadata_cols:
    if col in feature_cols:
        raise ValueError(f"Metadata column feature listesine girmiş: {col}")

bad_feature_cols = [
    "source_id",
    "is_augmented",
    "augmentation_type",
    "time",
    "weathercode",
    "year",
    TARGET
]

for bad_col in bad_feature_cols:
    if bad_col in feature_cols:
        raise ValueError(f"Leakage riski: {bad_col} feature listesine girmiş.")

missing_train_cols = [col for col in feature_cols + [TARGET] if col not in train_df.columns]
missing_test_cols = [col for col in feature_cols + [TARGET] if col not in test_df.columns]

if missing_train_cols:
    raise ValueError(f"Train dataset içinde eksik kolonlar var: {missing_train_cols}")

if missing_test_cols:
    raise ValueError(f"Test dataset içinde eksik kolonlar var: {missing_test_cols}")


X_train_raw = train_df[feature_cols]
y_train = train_df[TARGET].astype(int)

X_test_raw = test_df[feature_cols]
y_test = test_df[TARGET].astype(int)


# ============================================================
# ENCODING + SCALING
# ============================================================

encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

scaler = StandardScaler()

X_train_cat = encoder.fit_transform(X_train_raw[cat_cols])
X_test_cat = encoder.transform(X_test_raw[cat_cols])

X_train_num = scaler.fit_transform(X_train_raw[num_cols])
X_test_num = scaler.transform(X_test_raw[num_cols])

X_train_bin = X_train_raw[binary_cols].astype(int).values
X_test_bin = X_test_raw[binary_cols].astype(int).values

X_train = np.hstack([
    X_train_num,
    X_train_bin,
    X_train_cat
])

X_test = np.hstack([
    X_test_num,
    X_test_bin,
    X_test_cat
])

feature_order = (
    num_cols +
    binary_cols +
    list(encoder.get_feature_names_out(cat_cols))
)

print("\nFeature matrix created.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"Feature count: {len(feature_order)}")

if X_train.shape[1] != len(feature_order):
    raise ValueError("X_train feature sayısı ile feature_order uzunluğu eşleşmiyor.")

if X_test.shape[1] != len(feature_order):
    raise ValueError("X_test feature sayısı ile feature_order uzunluğu eşleşmiyor.")


# ============================================================
# SAVE PREPROCESSORS
# ============================================================

joblib.dump(encoder, MODEL_DIR / f"encoder_{RUN_NAME}.pkl")
joblib.dump(scaler, MODEL_DIR / f"scaler_{RUN_NAME}.pkl")
joblib.dump(feature_order, MODEL_DIR / f"feature_order_{RUN_NAME}.pkl")
joblib.dump(cat_cols, MODEL_DIR / f"cat_cols_{RUN_NAME}.pkl")
joblib.dump(num_cols, MODEL_DIR / f"num_cols_{RUN_NAME}.pkl")
joblib.dump(binary_cols, MODEL_DIR / f"binary_cols_{RUN_NAME}.pkl")

with open(REPORT_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "run_name": RUN_NAME,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "binary_cols": binary_cols,
            "feature_order": feature_order,
            "excluded_metadata_cols": metadata_cols
        },
        f,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# MODEL TRAINING
# ============================================================

results = []
precip_bin_reports = []
context_reports = []


# ------------------------------------------------------------
# 1) XGBoost - BASIC PARAMETERS
# ------------------------------------------------------------

xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=RANDOM_STATE
)

xgb.fit(X_train, y_train)

joblib.dump(xgb, MODEL_DIR / f"xgboost_{RUN_NAME}.pkl")

xgb_pred = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]

results.append(
    evaluate_global_metrics(
        model_name="XGBoost",
        y_true=y_test,
        y_pred=xgb_pred,
        y_proba=xgb_proba
    )
)

precip_bin_reports.append(
    evaluate_precipitation_bins(
        model_name="XGBoost",
        test_df=test_df,
        y_true=y_test,
        y_pred=xgb_pred,
        y_proba=xgb_proba
    )
)

context_reports.append(
    evaluate_context_bins(
        model_name="XGBoost",
        test_df=test_df,
        y_true=y_test,
        y_pred=xgb_pred,
        y_proba=xgb_proba
    )
)


# ------------------------------------------------------------
# 2) Random Forest - BASIC PARAMETERS
# ------------------------------------------------------------

rf = RandomForestClassifier(
    random_state=RANDOM_STATE
)

rf.fit(X_train, y_train)

joblib.dump(rf, MODEL_DIR / f"random_forest_{RUN_NAME}.pkl")

rf_pred = rf.predict(X_test)

if hasattr(rf, "predict_proba"):
    rf_proba = rf.predict_proba(X_test)[:, 1]
else:
    rf_proba = None

results.append(
    evaluate_global_metrics(
        model_name="Random Forest",
        y_true=y_test,
        y_pred=rf_pred,
        y_proba=rf_proba
    )
)

precip_bin_reports.append(
    evaluate_precipitation_bins(
        model_name="Random Forest",
        test_df=test_df,
        y_true=y_test,
        y_pred=rf_pred,
        y_proba=rf_proba
    )
)

context_reports.append(
    evaluate_context_bins(
        model_name="Random Forest",
        test_df=test_df,
        y_true=y_test,
        y_pred=rf_pred,
        y_proba=rf_proba
    )
)


# ------------------------------------------------------------
# 3) Neural Network - BASIC PARAMETERS
# ------------------------------------------------------------

nn = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

nn.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

nn.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    verbose=1
)

nn.save(MODEL_DIR / f"neural_network_{RUN_NAME}.keras")

nn_proba = nn.predict(X_test).reshape(-1)
nn_pred = (nn_proba > 0.5).astype(int)

results.append(
    evaluate_global_metrics(
        model_name="Neural Network",
        y_true=y_test,
        y_pred=nn_pred,
        y_proba=nn_proba
    )
)

precip_bin_reports.append(
    evaluate_precipitation_bins(
        model_name="Neural Network",
        test_df=test_df,
        y_true=y_test,
        y_pred=nn_pred,
        y_proba=nn_proba
    )
)

context_reports.append(
    evaluate_context_bins(
        model_name="Neural Network",
        test_df=test_df,
        y_true=y_test,
        y_pred=nn_pred,
        y_proba=nn_proba
    )
)


# ============================================================
# SAVE REPORTS
# ============================================================

results_df = pd.DataFrame(results)
results_df.to_csv(
    REPORT_DIR / f"{RUN_NAME}_global_model_results.csv",
    index=False
)

precip_bin_report_df = pd.concat(precip_bin_reports, ignore_index=True)
precip_bin_report_df.to_csv(
    REPORT_DIR / f"{RUN_NAME}_precipitation_bin_evaluation.csv",
    index=False
)

context_report_df = pd.concat(context_reports, ignore_index=True)
context_report_df.to_csv(
    REPORT_DIR / f"{RUN_NAME}_heavy_rain_context_evaluation.csv",
    index=False
)

classification_reports = {
    "XGBoost": classification_report(
        y_test,
        xgb_pred,
        zero_division=0,
        output_dict=True
    ),
    "Random Forest": classification_report(
        y_test,
        rf_pred,
        zero_division=0,
        output_dict=True
    ),
    "Neural Network": classification_report(
        y_test,
        nn_pred,
        zero_division=0,
        output_dict=True
    )
}

with open(REPORT_DIR / f"{RUN_NAME}_classification_reports.json", "w", encoding="utf-8") as f:
    json.dump(classification_reports, f, ensure_ascii=False, indent=4)

run_config = {
    "run_name": RUN_NAME,
    "weighted_training": False,
    "sample_weight_used_for_tree_models": False,
    "xgboost_params": {
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE
    },
    "random_forest_params": {
        "random_state": RANDOM_STATE
    },
    "neural_network_params": {
        "layers": [
            "Dense(32, relu)",
            "Dense(16, relu)",
            "Dense(1, sigmoid)"
        ],
        "learning_rate": 0.001,
        "validation_split": 0.2,
        "epochs": 10,
        "batch_size": 32
    }
}

with open(REPORT_DIR / f"{RUN_NAME}_run_config.json", "w", encoding="utf-8") as f:
    json.dump(run_config, f, ensure_ascii=False, indent=4)

print("\n=== GLOBAL RESULTS ===")
print(results_df)

print("\n=== PRECIPITATION BIN EVALUATION ===")
print(precip_bin_report_df)

print("\n=== HEAVY RAIN CONTEXT EVALUATION ===")
print(context_report_df)

print(f"\nModels saved to: {MODEL_DIR}")
print(f"Reports saved to: {REPORT_DIR}")