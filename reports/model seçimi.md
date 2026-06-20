# Model Karşılaştırması ve Seçim Gerekçesi


---

## 1. Riskli Bağlamlar (context = 1) — üç model de başarılı


| Bağlam (=1) | Model | Recall | Precision | F1 |
|---|---|---:|---:|---:|
| **risky_context** | XGBoost | 0.991 | 0.964 | 0.978 |
| | Random Forest | 1.000 | 0.961 | 0.980 |
| | Neural Network | 0.990 | 0.953 | 0.976 |
| **high_wind** | XGBoost | 0.988 | 0.973 | 0.980 |
| | Random Forest | 1.000 | 0.966 | 0.982 |
| | Neural Network | 0.992 | 0.958 | 0.979 |
| **night** | XGBoost | 0.994 | 0.965 | 0.979 |
| | Random Forest | 1.000 | 0.963 | 0.981 |
| | Neural Network | 0.991 | 0.956 | 0.977 |



---

## 2. Genel Performans (safe_context=0 satırı, yani veri çoğunluğu)

| Model | Accuracy | Recall | Precision | F1 |
|---|---:|---:|---:|---:|
| XGBoost | 0.770 | 0.790 | 0.733 | 0.760 |
| Random Forest | 0.730 | 0.675 | 0.723 | 0.698 |
| Neural Network | 0.770 | 0.784 | 0.737 | 0.759 |

**Sonuç:** XGBoost ve NN genel olarak benzer; Random Forest biraz geride.

---

#  SEÇİMİ BELİRLEYEN KISIM

## Güvenli Bağlam (safe_context = 1) 

Bu satırlarda **gerçek kaza oranı sadece %2.3**. Yani model "burada kaza YOK" demeli.
 *güvenli koşullarda yağış bir risk sinyali değildir. bu yüzden model az yanlış alarm vermeli*

| Model | Toplam satır | Yanlış Pozitif (yanlış alarm) | Gerçek Pozitif (yakalanan) | Yorum |
|---|---:|---:|---:|---|
| **XGBoost** | 2735 | **0** | 4 / 63 | Hiç yanlış alarm yok |
| Random Forest | 2735 | 0 | 0 / 63 |  Alarm yok ama hiç kaza da yakalayamadı |
| Neural Network | 2735 | **119** | 4 / 63 |  119 gereksiz alarm |

### Confusion matrix (safe_context=1)

```
                    [[Doğru Negatif, Yanlış Pozitif], [Yanlış Negatif, Doğru Pozitif]]

XGBoost:        [[2672,   0], [59,  4]]   ->   0 yanlış alarm
Random Forest:  [[2672,   0], [63,  0]]   ->   0 yanlış alarm, ama 0 gerçek yakalama
Neural Network: [[2553, 119], [59,  4]]   -> 119 yanlış alarm
```



**Karar:** Neural Network güvenli bağlamda 119 gereksiz alarm üretti — bu, projenin
"güvenli koşulda yağıştan korkma" hedefine aykırı. Random Forest hiç yanlış alarm
vermedi ama güvenli bağlamdaki gerçek kazaları da tamamen kaçırdı. XGBoost ise hem
güvenli bağlamı doğru tanıdı (0 yanlış alarm), hem riskli bağlamda yüksek recall'u
korudu. Bağlama göre en dengeli karar veren model olduğu için **XGBoost seçildi.**