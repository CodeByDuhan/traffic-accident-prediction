import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]



df = pd.read_csv(BASE_DIR / "data" / "interim" / "weather_all_cleaned.csv")  

df["is_saganak"] = (df["precipitation"] > 5).astype(int)

# şehir bazında toplam saat sayısı ve sağanak saat sayısı
summary = df.groupby("city").agg(
    total_hours=("is_saganak", "count"),
    saganak_hours=("is_saganak", "sum")
)

# yüzdelik oran
summary["saganak_oran"] = (summary["saganak_hours"] / summary["total_hours"]) * 100

# en çoktan aza sırala
summary = summary.sort_values("saganak_oran", ascending=False)

summary.to_csv(BASE_DIR / "reports" / "results_op2" / "saganak_oran_by_city.csv")
print("✅ Şehir bazlı sağanak oranı dosyaya yazıldı: saganak_oran_by_city.csv")
