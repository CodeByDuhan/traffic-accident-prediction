import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]



df = pd.read_csv(BASE_DIR / "data" / "processed" / "augmented_final_dataset_t+1.csv")

# Feature analiz aralıkları
bins_config = {
    "precipitation": [-0.1, 2, 5, 10, 100],
    "windspeed_10m": [-0.1, 10, 20, 100],
    "temperature_2m": [-100, 0, 35, 100],
    "hour": [-0.1, 6, 19, 24]
}


# sonuçları tutacağımız liste
summary_list = []

# her feature için analiz
for feature, bins in bins_config.items():
    df["feature_bin"] = pd.cut(df[feature], bins=bins, include_lowest=True)

    grouped = df.groupby("feature_bin")["accident_happened_t+1"].agg(
        total="count",
        positive="sum"
    ).reset_index()

    grouped["feature"] = feature
    grouped["positive_ratio"] = grouped["positive"] / grouped["total"]
    summary_list.append(grouped)

# birleştir
summary_df = pd.concat(summary_list, ignore_index=True)[
    ["feature", "feature_bin", "total", "positive", "positive_ratio"]
]
print(summary_df)
print("analizi tamamlandı")
