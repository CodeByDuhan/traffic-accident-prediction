import pandas as pd
from pathlib import Path


# ============================================================
# PATH SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

ORIGINAL_PATH = BASE_DIR / "data" / "processed" / "final_dataset_t+1.csv"

CORRECTED_PATH = (
    BASE_DIR / "data" / "processed" / "02_corrected_augmented_dataset_t+1.csv"
)

TRAIN_PATH = (
    BASE_DIR / "data" / "processed" / "train_corrected_augmented_dataset_t+1.csv"
)

TEST_PATH = (
    BASE_DIR / "data" / "processed" / "test_corrected_augmented_dataset_t+1.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "accident_happened_t+1"


# ============================================================
# REPORT FUNCTIONS
# ============================================================

def print_dataset_shape_report(
    original_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    print("\n=== DATASET SATIR SAYISI KARŞILAŞTIRMASI ===")
    print(f"Original rows : {len(original_df)}")
    print(f"Corrected rows: {len(corrected_df)}")
    print(f"Train rows    : {len(train_df)}")
    print(f"Test rows     : {len(test_df)}")
    print(f"Added rows    : {len(corrected_df) - len(original_df)}")

    if len(corrected_df) > 0:
        print(f"Train ratio   : {len(train_df) / len(corrected_df):.4f}")
        print(f"Test ratio    : {len(test_df) / len(corrected_df):.4f}")

    if len(train_df) + len(test_df) == len(corrected_df):
        print("Train + Test toplamı corrected dataset ile eşleşiyor.")
    else:
        print("UYARI: Train + Test toplamı corrected dataset ile eşleşmiyor.")


def print_target_distribution_report(df: pd.DataFrame, title: str) -> None:
    report = (
        df[TARGET]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("target")
        .reset_index(name="count")
    )

    report["ratio"] = report["count"] / len(df)

    print(f"\n=== {title} TARGET DAĞILIMI ===")
    print(report)


def print_precipitation_report(df: pd.DataFrame, title: str) -> None:
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

    print(f"\n=== {title} YAĞIŞ DAĞILIMI ===")
    print(report)


def print_feature_bin_report(df: pd.DataFrame, title: str) -> None:
    temp_df = df.copy()

    feature_bins = {
        "precipitation": [-0.101, 2, 5, 10, 100],
        "windspeed_10m": [-0.101, 10, 20, 100],
        "temperature_2m": [-100.001, 0, 35, 100],
        "hour": [-0.101, 6, 19, 24],
    }

    reports = []

    for feature, bins in feature_bins.items():
        temp_df["feature_bin"] = pd.cut(temp_df[feature], bins=bins)

        report = (
            temp_df.groupby("feature_bin", observed=False)[TARGET]
            .agg(
                total="count",
                positive="sum",
                positive_ratio="mean"
            )
            .reset_index()
        )

        report.insert(0, "feature", feature)
        reports.append(report)

    final_report = pd.concat(reports, ignore_index=True)

    print(f"\n=== {title} FEATURE BIN DAĞILIMI ===")
    print(final_report)


def print_augmentation_type_report(df: pd.DataFrame, title: str) -> None:
    if "augmentation_type" not in df.columns:
        print(f"\n=== {title} AUGMENTATION TYPE DAĞILIMI ===")
        print("augmentation_type kolonu yok.")
        return

    report = (
        df.groupby("augmentation_type", observed=False)[TARGET]
        .agg(
            total="count",
            positive="sum",
            positive_ratio="mean"
        )
        .reset_index()
    )

    print(f"\n=== {title} AUGMENTATION TYPE DAĞILIMI ===")
    print(report)


def check_source_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    if "source_id" not in train_df.columns or "source_id" not in test_df.columns:
        print("\n=== SOURCE_ID LEAKAGE CHECK ===")
        print("source_id kolonu train veya test dataset içinde yok.")
        return

    train_sources = set(train_df["source_id"].unique())
    test_sources = set(test_df["source_id"].unique())

    overlap = train_sources.intersection(test_sources)

    print("\n=== SOURCE_ID LEAKAGE CHECK ===")
    print(f"Train unique source_id: {len(train_sources)}")
    print(f"Test unique source_id : {len(test_sources)}")
    print(f"Overlap source_id     : {len(overlap)}")

    if len(overlap) == 0:
        print("Leakage yok: Aynı source_id train ve test arasında bölünmemiş.")
    else:
        print("UYARI: Leakage var. Bazı source_id'ler hem train hem test içinde.")


def compare_original_and_corrected_columns(
    original_df: pd.DataFrame,
    corrected_df: pd.DataFrame
) -> None:
    original_cols = set(original_df.columns)
    corrected_cols = set(corrected_df.columns)

    added_cols = sorted(corrected_cols - original_cols)
    removed_cols = sorted(original_cols - corrected_cols)

    print("\n=== ORIGINAL VS CORRECTED KOLON KARŞILAŞTIRMASI ===")
    print(f"Original column count : {len(original_df.columns)}")
    print(f"Corrected column count: {len(corrected_df.columns)}")
    print(f"Added columns         : {added_cols}")
    print(f"Removed columns       : {removed_cols}")


def compare_original_and_corrected_precip_bins(
    original_df: pd.DataFrame,
    corrected_df: pd.DataFrame
) -> None:
    original_temp = original_df.copy()
    corrected_temp = corrected_df.copy()

    original_temp["precip_bin"] = pd.cut(
        original_temp["precipitation"],
        bins=[-0.101, 2, 5, 10, 100],
        labels=["0–2mm", "2–5mm", "5–10mm", ">10mm"]
    )

    corrected_temp["precip_bin"] = pd.cut(
        corrected_temp["precipitation"],
        bins=[-0.101, 2, 5, 10, 100],
        labels=["0–2mm", "2–5mm", "5–10mm", ">10mm"]
    )

    original_report = (
        original_temp.groupby("precip_bin", observed=False)[TARGET]
        .agg(
            original_total="count",
            original_positive="sum",
            original_positive_ratio="mean"
        )
        .reset_index()
    )

    corrected_report = (
        corrected_temp.groupby("precip_bin", observed=False)[TARGET]
        .agg(
            corrected_total="count",
            corrected_positive="sum",
            corrected_positive_ratio="mean"
        )
        .reset_index()
    )

    comparison = original_report.merge(
        corrected_report,
        on="precip_bin",
        how="outer"
    )

    comparison["added_total"] = (
        comparison["corrected_total"] - comparison["original_total"]
    )

    comparison["added_positive"] = (
        comparison["corrected_positive"] - comparison["original_positive"]
    )

    print("\n=== ORIGINAL VS CORRECTED YAĞIŞ BIN KARŞILAŞTIRMASI ===")
    print(comparison)


# ============================================================
# MAIN
# ============================================================

def main():
    original_df = pd.read_csv(ORIGINAL_PATH)
    corrected_df = pd.read_csv(CORRECTED_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print_dataset_shape_report(
        original_df=original_df,
        corrected_df=corrected_df,
        train_df=train_df,
        test_df=test_df
    )

    compare_original_and_corrected_columns(
        original_df=original_df,
        corrected_df=corrected_df
    )

    compare_original_and_corrected_precip_bins(
        original_df=original_df,
        corrected_df=corrected_df
    )

    print_target_distribution_report(original_df, "ORIGINAL")
    print_target_distribution_report(corrected_df, "CORRECTED")
    print_target_distribution_report(train_df, "TRAIN")
    print_target_distribution_report(test_df, "TEST")

    print_precipitation_report(original_df, "ORIGINAL")
    print_precipitation_report(corrected_df, "CORRECTED")
    print_precipitation_report(train_df, "TRAIN")
    print_precipitation_report(test_df, "TEST")

    print_feature_bin_report(corrected_df, "CORRECTED")
    print_feature_bin_report(train_df, "TRAIN")
    print_feature_bin_report(test_df, "TEST")

    print_augmentation_type_report(corrected_df, "CORRECTED")
    print_augmentation_type_report(train_df, "TRAIN")
    print_augmentation_type_report(test_df, "TEST")

    check_source_leakage(train_df, test_df)


if __name__ == "__main__":
    main()