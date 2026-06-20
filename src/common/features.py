"""Single source of truth for feature engineering and model column lists.

Training, analysis, the API and the dashboards all import from here, so a
threshold change (e.g. the 5mm heavy-rain cutoff) propagates everywhere at
once instead of having to be edited in 7 different files.
"""

TARGET = "accident_happened_t+1"

CAT_COLS = [
    "city",
    "yağış_türü"
]

NUM_COLS = [
    "hour",
    "temperature_2m",
    "windspeed_10m",
    "precipitation"
]

BINARY_COLS = [
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

FEATURE_COLS = CAT_COLS + NUM_COLS + BINARY_COLS

# bookkeeping columns from the augmentation pipeline, must never reach the
# model: is_augmented / augmentation_type basically encode the label by
# construction, source_id would let the model memorize augmentation groups
METADATA_COLS = [
    "source_id",
    "is_augmented",
    "augmentation_type"
]


def add_features(df):
    df = df.copy()

    # >5mm/h = heavy rain ("saganak"), this threshold drives everything below
    df["is_saganak"] = (df["precipitation"] > 5).astype(int)

    df["is_night"] = (
        (df["hour"] >= 22) |
        (df["hour"] <= 6)
    ).astype(int)

    # <0 icing, >35 extreme heat
    df["dangerous_temp"] = (
        (df["temperature_2m"] < 0) |
        (df["temperature_2m"] > 35)
    ).astype(int)

    df["high_wind"] = (df["windspeed_10m"] > 20).astype(int)

    # explicit rain bins because the rain/risk relation is NOT monotonic here,
    # don't want the model assuming more rain = more accidents
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

    # main idea of this whole project: heavy rain alone is not a risk signal.
    # daytime + mild temp + low wind = drivers slow down or stay home
    df["heavy_rain_safe_context"] = (
        (df["precipitation"] > 5) &
        (df["temperature_2m"].between(10, 25)) &
        (df["windspeed_10m"] < 10) &
        (df["hour"].between(9, 18))
    ).astype(int)

    # opposite case: heavy rain + night/wind/extreme temp -> actually risky.
    # these two flags mirror the augmentation contexts in prepare_augmented_data.py
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
