import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# make the repo root importable so scripts can be run directly
# (python src/training/train_new_dataset_op.py)
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.common.features import (
    add_features,
    TARGET,
    CAT_COLS,
    NUM_COLS,
    BINARY_COLS,
    FEATURE_COLS,
    METADATA_COLS
)
from src.common.evaluation import (
    print_precipitation_distribution,
    evaluate_global_metrics,
    evaluate_precipitation_bins,
    evaluate_context_bins
)


"""
Optimized training run for the traffic accident prediction project.

Feature engineering and evaluation live in src/common so the API, dashboards
and analysis scripts share the exact same definitions. This script only owns
what is training-specific: fitting the preprocessors, the three models, and
dumping artifacts + reports.
"""


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
RUN_NAME = "train_new_dataset_op"


# ============================================================
# PATH SETTINGS
# ============================================================

# splits come from prepare_augmented_data.py -> GroupShuffleSplit on source_id,
# so a row and its jittered copies are never separated across train/test
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_corrected_augmented_dataset_t+1.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test_corrected_augmented_dataset_t+1.csv"

MODEL_DIR = BASE_DIR / "models" / RUN_NAME
REPORT_DIR = BASE_DIR / "reports" / f"results_{RUN_NAME}"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


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
# FEATURE COLUMN GUARDS
# ============================================================

for col in METADATA_COLS:
    if col in FEATURE_COLS:
        raise ValueError(f"Metadata column found in feature list: {col}")

# fail fast if a leakage-prone column ever sneaks into the feature list
bad_feature_cols = METADATA_COLS + ["time", "weathercode", "year", TARGET]
for bad_col in bad_feature_cols:
    if bad_col in FEATURE_COLS:
        raise ValueError(f"Leakage risk: {bad_col} found in feature list.")

missing_train_cols = [col for col in FEATURE_COLS + [TARGET] if col not in train_df.columns]
missing_test_cols = [col for col in FEATURE_COLS + [TARGET] if col not in test_df.columns]

if missing_train_cols:
    raise ValueError(f"Missing columns in train dataset: {missing_train_cols}")

if missing_test_cols:
    raise ValueError(f"Missing columns in test dataset: {missing_test_cols}")


X_train_raw = train_df[FEATURE_COLS]
y_train = train_df[TARGET].astype(int)

X_test_raw = test_df[FEATURE_COLS]
y_test = test_df[TARGET].astype(int)


# ============================================================
# ENCODING + SCALING
# ============================================================

# handle_unknown="ignore" so an unseen category coming through the API
# doesn't crash the request
encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

scaler = StandardScaler()

# fit on train only, test just gets transform -> no stats leakage
X_train_cat = encoder.fit_transform(X_train_raw[CAT_COLS])
X_test_cat = encoder.transform(X_test_raw[CAT_COLS])

X_train_num = scaler.fit_transform(X_train_raw[NUM_COLS])
X_test_num = scaler.transform(X_test_raw[NUM_COLS])

# binary flags stay unscaled, they're already 0/1
X_train_bin = X_train_raw[BINARY_COLS].astype(int).values
X_test_bin = X_test_raw[BINARY_COLS].astype(int).values

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

# saved with the model artifacts, the API/dashboards rebuild inputs in this
# exact order so serving can't silently misalign features
feature_order = (
    NUM_COLS +
    BINARY_COLS +
    list(encoder.get_feature_names_out(CAT_COLS))
)

print("\nFeature matrix created.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"Feature count: {len(feature_order)}")

if X_train.shape[1] != len(feature_order):
    raise ValueError("X_train feature count does not match feature_order length.")

if X_test.shape[1] != len(feature_order):
    raise ValueError("X_test feature count does not match feature_order length.")


# ============================================================
# SAVE PREPROCESSORS
# ============================================================

joblib.dump(encoder, MODEL_DIR / f"encoder_{RUN_NAME}.pkl")
joblib.dump(scaler, MODEL_DIR / f"scaler_{RUN_NAME}.pkl")
joblib.dump(feature_order, MODEL_DIR / f"feature_order_{RUN_NAME}.pkl")
joblib.dump(CAT_COLS, MODEL_DIR / f"cat_cols_{RUN_NAME}.pkl")
joblib.dump(NUM_COLS, MODEL_DIR / f"num_cols_{RUN_NAME}.pkl")
joblib.dump(BINARY_COLS, MODEL_DIR / f"binary_cols_{RUN_NAME}.pkl")

with open(REPORT_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "run_name": RUN_NAME,
            "cat_cols": CAT_COLS,
            "num_cols": NUM_COLS,
            "binary_cols": BINARY_COLS,
            "feature_order": feature_order,
            "excluded_metadata_cols": METADATA_COLS
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


def evaluate_model(model_name, y_pred, y_proba):
    # every model goes through the same three report layers
    results.append(
        evaluate_global_metrics(RUN_NAME, model_name, y_test, y_pred, y_proba)
    )
    precip_bin_reports.append(
        evaluate_precipitation_bins(RUN_NAME, model_name, test_df, y_test, y_pred, y_proba)
    )
    context_reports.append(
        evaluate_context_bins(RUN_NAME, model_name, test_df, y_test, y_pred, y_proba)
    )


# ------------------------------------------------------------
# 1) XGBoost
# ------------------------------------------------------------

# dataset contains near-duplicate jittered rows by design, so going for
# moderate depth + subsampling + some L1/L2 instead of deeper trees
xgb = XGBClassifier(
    learning_rate=0.1,
    n_estimators=350,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.8,
    reg_lambda=1.2,
    eval_metric="logloss",
    verbosity=0,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

xgb.fit(X_train, y_train)

joblib.dump(xgb, MODEL_DIR / f"xgboost_{RUN_NAME}.pkl")

xgb_pred = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]

evaluate_model("XGBoost", xgb_pred, xgb_proba)


# ------------------------------------------------------------
# 2) Random Forest
# ------------------------------------------------------------

# capped depth at 12 to keep the forest from just memorizing the
# augmented copies
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight=None
)

rf.fit(X_train, y_train)

joblib.dump(rf, MODEL_DIR / f"random_forest_{RUN_NAME}.pkl")

rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1] if hasattr(rf, "predict_proba") else None

evaluate_model("Random Forest", rf_pred, rf_proba)


# ------------------------------------------------------------
# 3) Neural Network
# ------------------------------------------------------------

# keras validation_split just slices the tail without shuffling, no guarantee
# the class ratio is preserved -> doing the validation split manually with
# stratify instead. this + dropout + early stopping + LR scheduling gave the
# biggest jump over the basic run (acc 0.749 -> 0.771)
X_nn_train, X_nn_val, y_nn_train, y_nn_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=RANDOM_STATE
)

nn = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dropout(0.1),
    Dense(1, activation="sigmoid")
])

# lower LR + bigger batches trained much more stable on ~1.7M rows
nn.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# restore_best_weights -> keep the best val epoch, not the last one
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    min_delta=0.0005,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_delta=0.0005,
    verbose=1
)

nn.fit(
    X_nn_train,
    y_nn_train,
    validation_data=(X_nn_val, y_nn_val),
    epochs=50,
    batch_size=256,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

nn.save(MODEL_DIR / f"neural_network_{RUN_NAME}.keras")

nn_proba = nn.predict(X_test).reshape(-1)
nn_pred = (nn_proba > 0.5).astype(int)

evaluate_model("Neural Network", nn_pred, nn_proba)


# ============================================================
# SAVE REPORTS
# ============================================================

# three report layers: global -> rain bins -> context flags.
# final model pick (XGBoost) came from the last two, NN was slightly
# ahead on global scores but over-predicted in heavy rain
results_df = pd.DataFrame(results)
results_df.to_csv(REPORT_DIR / f"{RUN_NAME}_global_model_results.csv", index=False)

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
    "XGBoost": classification_report(y_test, xgb_pred, zero_division=0, output_dict=True),
    "Random Forest": classification_report(y_test, rf_pred, zero_division=0, output_dict=True),
    "Neural Network": classification_report(y_test, nn_pred, zero_division=0, output_dict=True)
}

with open(REPORT_DIR / f"{RUN_NAME}_classification_reports.json", "w", encoding="utf-8") as f:
    json.dump(classification_reports, f, ensure_ascii=False, indent=4)

run_config = {
    "run_name": RUN_NAME,
    "xgboost_params": {
        "learning_rate": 0.1,
        "n_estimators": 350,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.8,
        "reg_lambda": 1.2
    },
    "random_forest_params": {
        "n_estimators": 200,
        "max_depth": 12,
        "class_weight": None
    },
    "neural_network_changes": {
        "manual_stratified_validation": True,
        "validation_split_removed": True,
        "learning_rate": 0.0005,
        "early_stopping_patience": 6,
        "reduce_lr_patience": 3,
        "dropout_1": 0.2,
        "dropout_2": 0.1,
        "epochs": 50,
        "batch_size": 256
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
