import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# PATH SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data" / "processed" / "final_dataset_t+1.csv"

CORRECTED_OUTPUT_PATH = (
    BASE_DIR / "data" / "processed" / "02_corrected_augmented_dataset_t+1.csv"
)

TRAIN_OUTPUT_PATH = (
    BASE_DIR / "data" / "processed" / "train_corrected_augmented_dataset_t+1.csv"
)

TEST_OUTPUT_PATH = (
    BASE_DIR / "data" / "processed" / "test_corrected_augmented_dataset_t+1.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
TARGET = "accident_happened_t+1"

TEST_SIZE = 0.20

# Safe sağanak negatif augmentation
SAFE_5_10_NEG_N = 12000
SAFE_GT_10_NEG_N = 2000

# Riskli sağanak pozitif augmentation
RISKY_5_10_POS_N = 3000
RISKY_GT_10_POS_N = 700


# ============================================================
# FEATURE / FLAG FUNCTIONS
# ============================================================

def add_basic_flags(df):
    """
    Augmentation kaynaklarını seçebilmek için gerekli temel flag'leri üretir.
    """

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

    return df


# ============================================================
# AUGMENTATION FUNCTIONS
# ============================================================

def create_augmented_samples(
    source_df: pd.DataFrame,
    n_samples: int,
    target_bin_name: str,
    target_label: int,
    augmentation_type: str,
    random_state: int,
    base_columns: list
) -> pd.DataFrame:
    """
    Belirli bir kaynak dataframe üzerinden jitter ile sentetik örnek üretir.

    Kritik:
        source_id korunur.
        Böylece aynı orijinal satırdan türeyen örnekler group split sırasında birlikte kalır.
    """

    if n_samples <= 0:
        return pd.DataFrame(columns=base_columns)

    if source_df.empty:
        print(
            f"UYARI: target_bin={target_bin_name}, target_label={target_label}, "
            f"augmentation_type={augmentation_type} için kaynak veri yok. Atlandı."
        )
        return pd.DataFrame(columns=base_columns)

    rng = np.random.default_rng(random_state)

    sampled = source_df.sample(
        n=n_samples,
        replace=True,
        random_state=random_state
    ).copy()

    sampled["temperature_2m"] = sampled["temperature_2m"] + rng.normal(
        loc=0,
        scale=0.5,
        size=n_samples
    )

    sampled["windspeed_10m"] = sampled["windspeed_10m"] + rng.normal(
        loc=0,
        scale=0.5,
        size=n_samples
    )

    sampled["precipitation"] = sampled["precipitation"] + rng.normal(
        loc=0,
        scale=0.5,
        size=n_samples
    )

    sampled["windspeed_10m"] = sampled["windspeed_10m"].clip(lower=0)
    sampled["precipitation"] = sampled["precipitation"].clip(lower=0)

    if target_bin_name == "5_10":
        sampled["precipitation"] = sampled["precipitation"].clip(
            lower=5.01,
            upper=10.0
        )

    elif target_bin_name == "gt_10":
        sampled["precipitation"] = sampled["precipitation"].clip(
            lower=10.01,
            upper=100.0
        )

    else:
        raise ValueError("target_bin_name sadece '5_10' veya 'gt_10' olabilir.")

    sampled[TARGET] = target_label

    sampled["is_augmented"] = 1
    sampled["augmentation_type"] = augmentation_type

    if "source_id" not in sampled.columns:
        raise ValueError("source_df içinde source_id yok. Group split için source_id zorunlu.")

    sampled = add_basic_flags(sampled)

    sampled = sampled[base_columns]

    return sampled


def create_corrected_augmented_dataset(df):
    """
    Full corrected augmented dataset üretir.
    Daha sonra bu dataset source_id bazlı group split ile train/test'e ayrılır.
    """

    df = df.copy()

    df = df.reset_index(drop=True)
    df["source_id"] = df.index.astype(int)
    df["is_augmented"] = 0
    df["augmentation_type"] = "original"

    df = add_basic_flags(df)

    base_columns = df.columns.tolist()

    # --------------------------------------------------------
    # SAFE SAGANAK NEGATIVE SOURCES
    # --------------------------------------------------------

    safe_saganak = df[
        (df["precipitation"] > 5) &
        (df["temperature_2m"].between(10, 25)) &
        (df["windspeed_10m"] < 10) &
        (df["hour"].between(9, 18)) &
        (df[TARGET] == 0)
    ].copy()

    safe_5_10 = safe_saganak[
        (safe_saganak["precipitation"] > 5) &
        (safe_saganak["precipitation"] <= 10)
    ].copy()

    safe_gt_10 = safe_saganak[
        safe_saganak["precipitation"] > 10
    ].copy()

    # --------------------------------------------------------
    # RISKY SAGANAK POSITIVE SOURCES
    # --------------------------------------------------------

    risky_saganak = df[
        (df["precipitation"] > 5) &
        (
            (df["is_night"] == 1) |
            (df["high_wind"] == 1) |
            (df["dangerous_temp"] == 1)
        ) &
        (df[TARGET] == 1)
    ].copy()

    risky_5_10 = risky_saganak[
        (risky_saganak["precipitation"] > 5) &
        (risky_saganak["precipitation"] <= 10)
    ].copy()

    risky_gt_10 = risky_saganak[
        risky_saganak["precipitation"] > 10
    ].copy()

    print("\n=== AUGMENTATION KAYNAK SAYILARI ===")
    print(f"Safe 5–10mm negatif kaynak : {len(safe_5_10)}")
    print(f"Safe >10mm negatif kaynak  : {len(safe_gt_10)}")
    print(f"Risky 5–10mm pozitif kaynak: {len(risky_5_10)}")
    print(f"Risky >10mm pozitif kaynak : {len(risky_gt_10)}")

    # --------------------------------------------------------
    # CREATE AUGMENTED ROWS
    # --------------------------------------------------------

    aug_safe_5_10_neg = create_augmented_samples(
        source_df=safe_5_10,
        n_samples=SAFE_5_10_NEG_N,
        target_bin_name="5_10",
        target_label=0,
        augmentation_type="safe_5_10_neg",
        random_state=RANDOM_STATE,
        base_columns=base_columns
    )

    aug_safe_gt_10_neg = create_augmented_samples(
        source_df=safe_gt_10,
        n_samples=SAFE_GT_10_NEG_N,
        target_bin_name="gt_10",
        target_label=0,
        augmentation_type="safe_gt_10_neg",
        random_state=RANDOM_STATE + 1,
        base_columns=base_columns
    )

    aug_risky_5_10_pos = create_augmented_samples(
        source_df=risky_5_10,
        n_samples=RISKY_5_10_POS_N,
        target_bin_name="5_10",
        target_label=1,
        augmentation_type="risky_5_10_pos",
        random_state=RANDOM_STATE + 2,
        base_columns=base_columns
    )

    aug_risky_gt_10_pos = create_augmented_samples(
        source_df=risky_gt_10,
        n_samples=RISKY_GT_10_POS_N,
        target_bin_name="gt_10",
        target_label=1,
        augmentation_type="risky_gt_10_pos",
        random_state=RANDOM_STATE + 3,
        base_columns=base_columns
    )

    corrected_df = pd.concat(
        [
            df,
            aug_safe_5_10_neg,
            aug_safe_gt_10_neg,
            aug_risky_5_10_pos,
            aug_risky_gt_10_pos
        ],
        ignore_index=True
    )

    corrected_df = add_basic_flags(corrected_df)

    print("\n=== EKLENEN ÖRNEK SAYILARI ===")
    print(f"Safe 5–10mm negatif eklendi : {len(aug_safe_5_10_neg)}")
    print(f"Safe >10mm negatif eklendi  : {len(aug_safe_gt_10_neg)}")
    print(f"Risky 5–10mm pozitif eklendi: {len(aug_risky_5_10_pos)}")
    print(f"Risky >10mm pozitif eklendi : {len(aug_risky_gt_10_pos)}")
    print(f"Toplam eklenen örnek        : {len(corrected_df) - len(df)}")

    return corrected_df


# ============================================================
# GROUP SPLIT FUNCTIONS
# ============================================================

def group_split_dataset(corrected_df):
    """
    source_id bazlı GroupShuffleSplit uygular.

    Aynı source_id'ye sahip orijinal + sentetik satırlar
    ya train tarafında kalır ya test tarafında kalır.
    """

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    groups = corrected_df["source_id"]
    y = corrected_df[TARGET]

    train_idx, test_idx = next(
        splitter.split(
            corrected_df,
            y=y,
            groups=groups
        )
    )

    train_df = corrected_df.iloc[train_idx].copy()
    test_df = corrected_df.iloc[test_idx].copy()

    return train_df, test_df


# ============================================================
# MAIN
# ============================================================

def main():
    df = pd.read_csv(INPUT_PATH)

    corrected_df = create_corrected_augmented_dataset(df)

    CORRECTED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    corrected_df.to_csv(CORRECTED_OUTPUT_PATH, index=False)

    print(f"\nCorrected full dataset kaydedildi:")
    print(CORRECTED_OUTPUT_PATH)

    train_df, test_df = group_split_dataset(corrected_df)

    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)

    print(f"\nTrain dataset kaydedildi:")
    print(TRAIN_OUTPUT_PATH)

    print(f"\nTest dataset kaydedildi:")
    print(TEST_OUTPUT_PATH)


if __name__ == "__main__":
    main()
