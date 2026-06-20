import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Dosya yolları
input_file = "/Users/duhanaydin/DataBoss/DataBoss2/data/final_dataset_t+1.csv"
output_folder = "data_graphs"
os.makedirs(output_folder, exist_ok=True)

# 2. Veriyi oku
df = pd.read_csv(input_file)

# 3. Grafik ayarları (görsel kalite)
plt.style.use("seaborn-v0_8-whitegrid")

# -------------------------------
# Kaza Sayılarını Yağış Türüne Göre
# -------------------------------
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x="yağış_türü", hue="accident_happened_t+1", palette="Set2")
plt.title("Accidents by Precipitation Type")
plt.xlabel("Precipitation Type")
plt.ylabel("Count")
plt.legend(title="Accident")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "accidents_by_weather_type.png"))
plt.close()

# -------------------------------
# Şehir Bazlı Kaza Yoğunluğu
# -------------------------------
accident_by_city = df.groupby("city")["accident_happened_t+1"].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
accident_by_city.plot(kind="bar", color="salmon")
plt.title("Total Accidents by City")
plt.ylabel("Number of Accidents")
plt.xlabel("City")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "accidents_by_city.png"))
plt.close()

# -------------------------------
# Korelasyon Matrisi (Sayısal Değişkenler)
# -------------------------------
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
corr = df[num_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix (Numerical Features)")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "correlation_matrix.png"))
plt.close()

print(f"✅ EDA tamamlandı. Grafikler '{output_folder}/' klasörüne kaydedildi.")
