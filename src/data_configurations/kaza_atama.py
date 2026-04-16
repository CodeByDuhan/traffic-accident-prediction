import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# maks %60 oranda kaza dağıtımı yapıcazz
MAX_POSITIVE_RATE = 0.60

# veriyi oku
df = pd.read_csv(BASE_DIR / "data" / "interim" / "weather_all_cleaned.csv")
df["time"] = pd.to_datetime(df["time"])
df["year"] = df["time"].dt.year
df["hour"] = df["time"].dt.hour

# başlangıç risk_score
risk_score = pd.Series(0.01, index=df.index)

#  sıcaklık etkisi
risk_score += (df["temperature_2m"] < 0) * 0.15 # buzlanma
risk_score += (df["temperature_2m"] > 35) * 0.05

#  rüzgar etkisi
risk_score += (df["windspeed_10m"] > 20) * 0.15
risk_score += ((df["windspeed_10m"] > 10) & (df["windspeed_10m"] <= 20)) * 0.08

risk_score += ((df["precipitation"] > 0) & (df["precipitation"] <= 2)) * 0.15
risk_score += ((df["precipitation"] > 2) & (df["precipitation"] <= 5)) * 0.08
risk_score += ((df["precipitation"] > 5) & (df["precipitation"] <= 10)) * (-0.03)
risk_score += ((df["precipitation"] > 10)) * (-0.05)


#  yağış türü etkisi
risk_score += (df["yağış_türü"] == "kar") * 0.15
risk_score += (df["yağış_türü"] == "yağmur") * 0.10

#  aaat etkisi
risk_score += df["hour"].isin([0,1,2,3,4,5,6]) * 0.15
risk_score += df["hour"].isin([19,20,21,22,23]) * 0.08

#  kaza datasını oku
acc_df = pd.read_csv(BASE_DIR / "data" / "raw" / "accident_counts_2022_2024.csv")
acc_df.columns = [col.strip().lower() for col in acc_df.columns]
df["accident_happened"] = 0

#  kaza dağıtımı şehir + yıl bazında
for _, row in acc_df.iterrows():
    city, year, count = row["province"], row["year"], row["total_accidents"]
    mask = (df["city"].str.lower() == city.lower()) & (df["year"] == year)
    sub_df = df[mask]

    if sub_df.empty:
        print(f"{city} {year} için veri yok")
        continue

    max_allowed = int(len(sub_df) * MAX_POSITIVE_RATE)
    allowed = min(count, max_allowed)

    sub_scores = risk_score[mask]
    weights = sub_scores if sub_scores.sum() > 0 else np.ones(len(sub_df))
    weights /= weights.sum()

    selected_indices = np.random.choice(sub_df.index, size=allowed, replace=False, p=weights.values)
    df.loc[selected_indices, "accident_happened"] = 1

    print(f"{city}-{year}: {allowed} saat etiketlendi") # kaza oranı tekrrar kontrol

#  T+1 hedef sütunu oluşturcaz çünkü model +1 saati tahmin edicek
df = df.sort_values(by=["city", "time"])
df["accident_happened_t+1"] = df.groupby("city")["accident_happened"].shift(-1)
df.dropna(subset=["accident_happened_t+1"], inplace=True)
df["accident_happened_t+1"] = df["accident_happened_t+1"].astype(int)

df.drop(columns=["accident_happened"]).to_csv(
    BASE_DIR / "data" / "processed" / "final_dataset_t+1.csv",
    index=False
)
print(" final_dataset_t+1.csv başarıyla oluştu.")

positive_rate = df["accident_happened"].mean()
print(f"Toplam pozitif oran: {positive_rate:.2%}")

print(df["accident_happened_t+1"].value_counts(normalize=True))


