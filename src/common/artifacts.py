"""Model artifact loading + serving-side input preparation.

Used by the Flask API, the Streamlit dashboards and the analysis scripts.
Everything reads the artifacts saved at training time (encoder, scaler,
feature order), so serving applies exactly the same transform as training.
"""

import joblib
import pandas as pd

from .features import add_features


def load_artifacts(model_dir, run_name):
    return {
        "encoder": joblib.load(model_dir / f"encoder_{run_name}.pkl"),
        "scaler": joblib.load(model_dir / f"scaler_{run_name}.pkl"),
        "feature_order": joblib.load(model_dir / f"feature_order_{run_name}.pkl"),
        "cat_cols": joblib.load(model_dir / f"cat_cols_{run_name}.pkl"),
        "num_cols": joblib.load(model_dir / f"num_cols_{run_name}.pkl"),
        "binary_cols": joblib.load(model_dir / f"binary_cols_{run_name}.pkl"),
    }


def build_model_matrix(df, artifacts):
    # transform-only, never fit: encoder/scaler were fit on the training split
    encoder = artifacts["encoder"]
    scaler = artifacts["scaler"]

    X_cat = pd.DataFrame(
        encoder.transform(df[artifacts["cat_cols"]]),
        columns=encoder.get_feature_names_out(artifacts["cat_cols"]),
        index=df.index
    )

    X_num = pd.DataFrame(
        scaler.transform(df[artifacts["num_cols"]]),
        columns=artifacts["num_cols"],
        index=df.index
    )

    X_bin = df[artifacts["binary_cols"]].astype(int)

    X = pd.concat([X_num, X_bin, X_cat], axis=1)

    # reorder to the exact column order saved at training time, so serving
    # can't silently misalign features
    return X[artifacts["feature_order"]]


def prepare_input(input_df, artifacts):
    # validate the raw request BEFORE add_features: binary cols are derived
    # there, and add_features itself crashes on a missing raw column, so
    # checking afterwards (as the old version did) could never actually fire
    required = artifacts["cat_cols"] + artifacts["num_cols"]
    missing_cols = [col for col in required if col not in input_df.columns]

    if missing_cols:
        raise ValueError(f"Eksik input kolonları: {missing_cols}")

    # raw request -> engineered features -> model matrix
    input_df = add_features(input_df)

    return build_model_matrix(input_df, artifacts)
