## Trafik Kazası Tahmin Sistemi (Staj Projesi – DataBoss, ODTÜ Teknokent)

Merhaba, bu proje stajım süresince geliştirdiğim bir trafik kazası tahmin sistemidir.  
Amacım, şehir bazlı saatlik hava durumu verilerinden yola çıkarak bir sonraki saat için trafik kazası olup olmayacağını öngörmekti.

Bu repo, geliştirme sürecimde kullandığım tüm veri işleme, model eğitimi, görselleştirme, API ve arayüz dosyalarını içeriyor.  
Projenin son halini hem Flask API olarak servis ettim ve de Streamlit ile görsel dashboardlar hazırladım.

---

##  Neyi Amaçladım?

Benim için bu proje sadece bir tahmin modeli kurmak değildi.  
Aynı zamanda aşağıdaki hedeflere ulaşmak istedim:

- Farklı modelleri karşılaştırarak en doğru tahmini yapabilen yapıyı bulmak
- Veriyle uğraşmak, etiketlemek, veri mühendisliği yapmak
- Bir modeli kullanıcıya API olarak sunmak
- Modelin nasıl düşündüğünü SHAP ve PDP ile analiz etmek
- Kullanıcılar için dashboard oluşturmak

---

---

##  Veri Toplama ve Hazırlama Sürecim

> Bu proje için veriyi sıfırdan ve anlamlı bir şekilde hazırlamam gerekiyordu.

- Saatlik hava durumu verilerini Open-Meteo API üzerinden çektim.
- Open-Meteo koordinatlarla çalıştığı için önce Türkiye’deki illerin enlem-boylam bilgilerini Turkiye_il_koordinatlar.csv dosyasında topladım.
- Bu koordinatları kullanarak 2022–2024 yılları arasında 81 şehir için saatlik hava verisini çekip birleştirdim
- Ardından TÜİK’ten aldığım şehir bazlı yıllık toplam trafik kazası sayılarını, hava durumu verileriyle eşleştirdim(accident_counts_2022_2024.csv)
- Böylece her şehir için saatlik gözlem + yıllık kaza sayısı bilgisine sahip veri setini elde ettim.

-(Veri setleri 100 mb tan büyük olduğundan data klasöründe ana başlıklarını bıraktım sadece.)
---

## Yağış Türü Kategorileştirmesi (weathercode)

Open-Meteo’dan gelen weathercode sütunu çok sayıda farklı hava durumu kodu içerdiği için bunu basitleştirdim.  
Her gözlem için hava durumunu şu üç gruptan birine çevirdim:

- yağmur
- kar
- yok

Bu sayede modelin yağış türünü yorumlaması kolaylaştı.

---

## Veri Augmentasyonu: is_saganak

Model ilk aşamada is_saganak (sağanak yağış var mı) değişkenini öğrenmekte zorlandı.  
Bunun nedeni, veri setinde bu durumun nadir görülmesiydi. Bu yüzden:

- is_saganak = 1 olan satırları filtreledim
- Bu satırları uygun oranla veri setine çoğaltma işlemi uygulayarak yeniden ekledim (15 bin veri, genel orana göre %10 un biraz üzerine çıkardım.)

> Böylece model, sağanak yağışın olduğu durumları daha iyi görebildi ve bu özelliği daha anlamlı şekilde öğrenmeye başladı.
Böylelikle aşırı sağanak olduğunda sürücülerin ekstra yavaş ve dikkatli gideceğini varsayarak kaza tahmin oranını düşürmeyi başardım.

##  Model Geliştirme Sürecim

Model geliştirmeyi 3 aşamalı olarak planladım:

1. basic: Hiçbir optimizasyon yapmadan modelleri test ettim  
2. op: Daha iyi performans için feature engineering ve hyperparametre tuning yaptım  
3. op2: Son olarak özellikleri genişletip önemli binary feature’ları doğrudan ekledim

>  Bu yaklaşımı daha önceki CNN projemden ilham alarak yaptım.  
> O projede de kademe kademe ilerleyerek en iyi sonucu aramıştım.  
> Ama bu kez görüntü değil sadece sayısal veriler kullandığım için sonuçlar arası fark daha sınırlı kaldı.

---

### Logistic Regression

Başta Logistic Regression da eklemiştim ama:

- Doğrusal olduğu için non-linear ilişkileri yakalayamadı
- Performansı oldukça düşüktü

Bu yüzden bu repo dosyalarında yer almadı, koddan da çıkardım.

---

## Model Sonuçları

| Model          | Versiyon     | Accuracy | Precision | Recall |
|----------------|--------------|----------|-----------|--------|
| XGBoost        | basic        | 0.7714   | 0.7324    | 0.7897 |
| XGBoost        | optimized    | 0.7713   | 0.7319    | 0.7904 |
| **XGBoost**    | optimized2   | **0.7951** | **0.7434**  | **0.8490** |
| Neural Network | optimized2   | 0.8013   | 0.7634    | 0.8253 |

### Final deploy ettiğim model: XGBoost (op2) 
### Alternatif dashboard modeli: Neural Network (op2)

---

##  Etiketleme ve %60 Sınırı

Veride şehir-yıl bazlı toplam kaza sayısı vardı. Bunları saatlik veriyle doğrudan eşleştirmek regresyon gibi çalışırdı.  
Ama ben sınıflandırma modeli kurduğum için, kazaları risk_score oranlarına göre dağıttım (probabilistic sampling).  
Yani aslında:

> Saatlik hava durumu özelliklerine göre kaza olma ihtimali üzerinden kazaları dağıttım.

Ama bu dağılımı da kontrollü tuttum çünkü çok fazla 1 verirsem model overfit olurdu.  
Bu yüzden:  
> Ya şehir-yıldaki toplam kaza sayısı kadar (3 yıldaki toplam saat sayısının %60 kadarını geçmediyse) 
> Ya da toplam saatlerin maksimum %60'ı kadar kaza etiketi verdim

---

### Uygulanan Kod Mantığı (veri etiketi sayısını belirleme)

Kazaları dağıtırken, sadece TÜİK’ten gelen sayı yeterli değildi.  
Ben modelin öğrenmesini dengede tutmak için her şehir-yıl kombinasyonunda şu satırı kullandım:


max_allowed = int(len(sub_df) * MAX_POSITIVE_RATE)
allowed = min(count, max_allowed)


Burada:
-  sub_df: O şehir-yıla ait saatlik veri (örneğin 1 yıl * 8760 saat)
-  count: O yıl için TÜİK’ten gelen toplam kaza sayısı
-  MAX_POSITIVE_RATE: %60 olarak sabit tanımlandı
-  allowed: O şehir-yıl için en fazla kaç tane "kaza oldu" etiketi verebileceğimi hesapladı

> Bu şekilde hem gerçek kaza sayılarına bağlı kaldım, hem de toplam saat sayısının %60’ını geçmemeyi garanti ederek modeli dengesiz veriyle eğitmedim.

##  Flask API (XGBoost-op2 Modeli)
	python3 flask_dep.py


### POST /predict
  json
{
  "city": "Ankara",
  "yağış_türü": "yağmur",
  "hour": 17,
  "temperature_2m": 24.5,
  "windspeed_10m": 12.3,
  "precipitation": 3.2
}


  json
{
  "prediction": 1,
  "probability": 0.7821
}


---

##  Dashboardlar

- `dashboard_xgb.py` → XGBoost için tahmin arayüzü  
- `dashboard_nn.py` → Neural Network için ayrı arayüz

Her iki dashboard da:
- Şehir ve hava durumu girince modelden tahmin alıyor
- Model içindeki `is_saganak`, `risky_rain`, `dangerous_temp` gibi feature’ları backend’de otomatik hesaplıyor

---

##  Görselleştirme (SHAP & PDP)

Modelin nasıl düşündüğünü anlamak için SHAP, PDP ve feature importance görselleri oluşturdum.
Genel yapıda, az veriye sahip feature lar verideki yaygın featurelara göre altta kaldığını gördüm. fakat ayrı olarak
Baktığımda hepsinin model öğrenmesinde bir etken olduğunu gördüm

Özellikle şu özellikleri görselleştirdim:

- is_saganak
- dangerous_temp
- risky_rain
- is_night

Görseller: results_op2_MAIN/ klasöründe



## Geliştirici Bilgisi

- İsim: Duhan Aydın  
- Staj Yeri: DataBoss, ODTÜ Teknokent  
- Proje Türü: Gerçek veriyle tahmin sistemi (binary classification)  
 
