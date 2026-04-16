import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


df = pd.read_csv(BASE_DIR / "data" / "processed" / "augmented_final_dataset_t+1.csv")

# Sağanak mm aralıklarını oluştur
bins = [-0.1, 2, 5, 10, 100]
labels = ["0–2mm", "2–5mm", "5–10mm", ">10mm"]
df["precip_bin"] = pd.cut(df["precipitation"], bins=bins, labels=labels)

# Her aralık için kaza istatistikleri
summary = (
    df.groupby("precip_bin")["accident_happened_t+1"]
    .agg(total="count", positive="sum")
    .reset_index()
)

summary["positive_ratio"] = summary["positive"] / summary["total"]


print(summary)
