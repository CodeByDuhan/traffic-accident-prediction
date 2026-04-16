import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


df = pd.read_csv(BASE_DIR / "data" / "processed" / "final_dataset_t+1.csv")

#  is_saganak flag'i oluştur (ana veri için)
df["is_saganak"] = (df["precipitation"] > 5).astype(int)

# güvenli sağanak saatlerini kazasız, düşük rüzgar, orta sıcaklık, gündüz ile filtreledim
safe_saganak = df[
    (df["precipitation"] > 5) &
    (df["temperature_2m"].between(10, 25)) &
    (df["windspeed_10m"] < 10) &
    (df["hour"].between(9, 18)) &
    (df["accident_happened_t+1"] == 0)
].copy()

#  Augmentasyon yeri, jitter (küçük rastgele değişiklikler ile ) ekleyerek 15.000 örnek ürettim
# çünkü model asla sağanakta risk azalır non lineariteyi veri azlığından öğrenemiyordu
N = 15000
augmented = safe_saganak.sample(n=N, replace=True).assign(
    temperature_2m=lambda x: x["temperature_2m"] + np.random.normal(0, 0.5, size=N),
    windspeed_10m=lambda x: x["windspeed_10m"] + np.random.normal(0, 0.5, size=N),
    precipitation=lambda x: x["precipitation"] + np.random.normal(0, 0.5, size=N),
)

augmented["accident_happened_t+1"] = 0

# sabit 1 yerine her döngüden sonra tekrar hesaplama:
augmented["is_saganak"] = (augmented["precipitation"] > 5).astype(int)

# birleştir
full_df = pd.concat([df, augmented], ignore_index=True)

# kaydet
full_df.to_csv(BASE_DIR / "data" / "processed" / "augmented_final_dataset_t+1.csv", index=False)
print("kaydedildi")
