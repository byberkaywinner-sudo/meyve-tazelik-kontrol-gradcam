<div align="center">

# 🍎🍌🍊 Yapay Zeka ve XAI (Grad-CAM) Destekli Akıllı Meyve Tazelik Kontrol Sistemi

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-red?logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Arayüz-FF4B4B?logo=streamlit&logoColor=white)
![XAI](https://img.shields.io/badge/XAI-Grad--CAM-9cf)
![Durum](https://img.shields.io/badge/Durum-Tamamlandı-success)

</div>

---

## 📋 Proje Kapağı

| Alan | Bilgi |
|---|---|
| **Proje Adı** | Yapay Zeka ve XAI (Grad-CAM) Destekli Akıllı Meyve Tazelik Kontrol Sistemi |
| **Dersin Öğretim Üyesi** | Hüseyin Yanık |
| **Hazırlayanlar** | İbrahim Berkay Tunçdemir — Hüseyin Emre Şeker |
| **Öğrenci No** | 23430070018 — 23430070073 |
| **Üniversite** | Mersin Üniversitesi |
| **Fakülte / Yüksekokul** | Erdemli Uygulamalı Teknoloji ve İşletmecilik Yüksekokulu |
| **Bölüm** | Bilişim Sistemleri ve Teknolojileri |

---

## 📑 İçindekiler

1. [Projenin Amacı ve Kapsamı](#1--projenin-amacı-ve-kapsamı)
2. [Veri Seti ve Veri Artırma (Data Augmentation) Stratejileri](#2--veri-seti-ve-veri-artırma-data-augmentation-stratejileri)
3. [Model Mimarisi](#3--model-mimarisi)
4. [Eğitim Süreci ve Optimizasyon](#4-️-eğitim-süreci-ve-optimizasyon)
5. [Açıklanabilir Yapay Zeka (XAI) ve Grad-CAM Entegrasyonu](#5--açıklanabilir-yapay-zeka-xai-ve-grad-cam-entegrasyonu)
6. [Kurulum ve Çalıştırma Talimatları](#6--kurulum-ve-çalıştırma-talimatları)
7. [Sonuç ve Değerlendirme](#7--sonuç-ve-değerlendirme)

---

## 1. 🎯 Projenin Amacı ve Kapsamı

Gıda paketleme ve dağıtım tesislerinde kalite kontrol süreci, geleneksel olarak insan gözüyle yürütülen, yorucu, subjektif ve ölçeklenmesi zor bir iştir. Bir taşıma bandı üzerinde saniyeler içinde akıp giden meyvelerin her birinin tazelik durumunu tutarlı şekilde değerlendirmek, insan operatör için pratikte sürdürülemez bir yük oluşturur. Bu proje, bu probleme **görüntü tabanlı, otomatik ve nesnel** bir çözüm sunmayı amaçlamaktadır.

**Senaryo:** Bir meyve paketleme hattında, taşıma bandı üzerinden tekil olarak geçen elma, muz ve portakallar; bant üzerine konumlandırılmış bir kamera tarafından görüntülenir ve sistem her bir meyveyi **gerçek zamanlı olarak** "Taze" veya "Çürük" sınıflarına ayırır.

Bu endüstriyel senaryo, projenin en kritik mühendislik zorluğunu da ortaya çıkarmıştır: **bant üzerindeki meyveler, sabit ve temiz bir stüdyo arka planında değil; metal/plastik bant dokusu, kasa kenarları, değişken paketleme hattı ışığı ve farklı kamera açıları gibi "gürültülü" bir ortamda** görüntülenmektedir. Yapılan ilk deneylerde modelin, meyvenin kendisi değil **arka plandaki doku ve renk örüntülerini** öğrenerek yanlış sınıflandırmalar (False Positive) yaptığı gözlemlenmiştir — örneğin bir sepetin gölgeli dokusu, çürük lekesiyle karıştırılabilmektedir. Bu gözlem, **Bölüm 2**'de detaylandırılan güçlü veri artırma stratejisinin doğrudan motivasyonunu oluşturmuştur.

**Kapsam:** Sistem, 3 meyve türü × 2 tazelik durumu olmak üzere toplam **6 sınıflı** bir görüntü sınıflandırma problemi olarak tasarlanmıştır:

| Meyve | Taze | Çürük |
|---|---|---|
| Elma 🍎 | Taze Elma 🍏 | Çürük Elma 🍎 |
| Muz 🍌 | Taze Muz 🍌 | Çürük Muz 🍌 |
| Portakal 🍊 | Taze Portakal 🍊 | Çürük Portakal 🍊 |

Projenin nihai çıktısı sadece bir sınıflandırma etiketi değildir; aynı zamanda **kararın nedenini görselleştiren** bir Açıklanabilir Yapay Zeka (XAI) katmanı içeren, Streamlit tabanlı interaktif bir web arayüzüdür.

---

## 2. 🗂️ Veri Seti ve Veri Artırma (Data Augmentation) Stratejileri

### 2.1. Veri Seti Yapısı

Veri seti, `dataset/dataset/train` ve `dataset/dataset/val` olmak üzere önceden ayrılmış eğitim/doğrulama klasörlerinden oluşmaktadır. Her sınıf kendi alt klasöründe tutulmakta ve Keras'ın `image_dataset_from_directory` fonksiyonu, klasör adlarını otomatik olarak sınıf etiketine dönüştürmektedir. Eğitim ve doğrulama setlerinin **aynı sınıf sırasına** sahip olmasını garanti etmek için sabit bir `seed=42` değeri kullanılmıştır; bu, sonuçların tekrar üretilebilirliğini (reproducibility) doğrudan etkileyen kritik bir mühendislik detayıdır.

### 2.2. Veri Artırma (Data Augmentation) Stratejisi — Endüstriyel Gerekçelendirme

Bölüm 1'de açıklanan "arka plan gürültüsü" problemini çözmek amacıyla, modelin mimarisine **eğitim sırasında aktif, çıkarım (inference) sırasında pasif** olan bir veri artırma katmanı eklenmiştir. Bu katman, her epoch'ta modele meyvenin biraz farklı bir versiyonunu göstererek, modelin "meyveyi" öğrenmesini, "arka planı ve konumu" ezberlemesini engellemeyi hedefler:

| Katman | Parametre | Endüstriyel Gerekçe |
|---|---|---|
| `RandomFlip` | `horizontal` | Meyvenin bant üzerinde sağa veya sola dönük olması sonucu değiştirmemelidir. |
| `RandomRotation` | `0.15` | Meyveler bant üzerinde her zaman dik durmaz; hafif dönüklük senaryoları simüle edilir. |
| `RandomZoom` | `0.2` | Kameranın bant üzerindeki nesneye olan mesafesi (ve dolayısıyla görünür meyve boyutu) sabit değildir. |
| `RandomContrast` | `0.2` | Paketleme hattındaki değişken aydınlatma (gölgeli/parlak bölgeler) koşullarına dayanıklılık sağlar. |
| `RandomTranslation` | `0.1 / 0.1` | Meyvenin kamera karesinin merkezine tam hizalanmamış, kenarda kalmış görüntülerini simüle eder. |

Bu katman, modelin bir parçası olarak (`.keras` formatında) kaydedilir; böylece eğitim mantığı dağıtım (deployment) aşamasına da taşınır ve ayrı bir ön-işleme adımına gerek kalmaz.

### 2.3. Performans Optimizasyonu

Veri setleri, `.cache()` ve `.prefetch(buffer_size=tf.data.AUTOTUNE)` ile sarmalanmıştır. Bu, diskten tekrar tekrar okuma maliyetini ortadan kaldırır ve bir sonraki veri grubunun (batch), GPU/CPU hesaplama yaparken arka planda hazırlanmasını sağlayarak eğitim hızını artırır — modern TensorFlow `tf.data` performans standardıdır.

---

## 3. 🧠 Model Mimarisi

Model, Keras'ın **Functional API**'si kullanılarak (`keras.Model(inputs=..., outputs=...)`) inşa edilmiştir. Bu mimari tercih, sadece bir kodlama stili değil; **Bölüm 5**'te detaylandırılan Grad-CAM entegrasyonunun sağlıklı çalışabilmesi için bilinçli bir mühendislik kararıdır.

```
Girdi (224×224×3)
      │
Veri Artırma Katmanı (sadece eğitimde aktif)
      │
Rescaling (1/255)
      │
┌─────────────── 1. Evrişim Bloğu ───────────────┐
│  Conv2D(32, 3×3) → BatchNormalization → MaxPool │
└──────────────────────────────────────────────────┘
      │
┌─────────────── 2. Evrişim Bloğu ───────────────┐
│  Conv2D(64, 3×3) → BatchNormalization → MaxPool │
└──────────────────────────────────────────────────┘
      │
┌──────── 3. Evrişim Bloğu (son_conv_katmani) ────┐
│ Conv2D(128, 3×3) → BatchNormalization → MaxPool │
└──────────────────────────────────────────────────┘
      │
   Flatten
      │
Dense(128) → BatchNormalization → Dropout(0.5)
      │
Dense(6, Softmax)  →  Çıkış
```

**Mimari Tasarım Kararları:**

- **BatchNormalization:** Her evrişim ve yoğun (Dense) bloğunun sonrasına eklenmiştir. Bu katman, bir önceki katmanın çıkış dağılımını normalize ederek (1) eğitimi hızlandırır, (2) daha yüksek öğrenme oranlarının kararlı şekilde kullanılabilmesini sağlar ve (3) modelin arka plandaki rastgele doku/renk varyasyonlarına karşı daha az hassas, daha genellenebilir özellikler öğrenmesine katkıda bulunur.
- **Dropout (%50):** Yalnızca yoğun katman bloğunda, sınıflandırma kararının verildiği noktada uygulanmıştır. Eğitim sırasında nöronların yarısını rastgele devre dışı bırakarak, modelin belirli nöron kombinasyonlarına aşırı bağımlı hale gelmesini (overfitting) engeller.
- **`son_conv_katmani` İsimlendirmesi:** Üçüncü evrişim bloğunun çıkış katmanı bilinçli olarak isimlendirilmiştir; bu isim, Grad-CAM ısı haritası hesaplamasında doğrudan referans olarak kullanılmaktadır (bkz. Bölüm 5).
- **Kayıp Fonksiyonu Uyumluluğu:** Etiketler tam sayı (integer) formatında tutulduğu için çıkış katmanı `softmax` aktivasyonu ile 6 nörona sahiptir ve `sparse_categorical_crossentropy` kayıp fonksiyonuyla eşleştirilmiştir; bu, etiketleri one-hot encode etme adımını gereksiz kılan, bellek açısından verimli bir tercihtir.

---

## 4. ⚙️ Eğitim Süreci ve Optimizasyon

Eğitim süreci, sabit bir epoch sayısı dayatmak yerine, **modelin kendi öğrenme dinamiğine göre otomatik olarak optimize edilmesini** sağlayan üç callback mekanizmasıyla yönetilmiştir. Bu, "epoch sayısı keyfi seçildi" eleştirisinin önüne geçen, veriye dayalı bir yaklaşımdır.

### 4.1. Kullanılan Callback Mekanizmaları

| Callback | İzlenen Metrik | Yapılandırma | Görevi |
|---|---|---|---|
| `EarlyStopping` | `val_loss` | `patience=8`, `restore_best_weights=True` | Doğrulama kaybı 8 epoch boyunca iyileşmezse eğitimi durdurur ve **en iyi ağırlıkları geri yükler**. |
| `ModelCheckpoint` | `val_accuracy` | `save_best_only=True` | En yüksek doğrulama doğruluğuna sahip model anlık görüntüsünü diske kalıcı olarak kaydeder. |
| `ReduceLROnPlateau` | `val_loss` | `factor=0.5`, `patience=4`, `min_lr=1e-6` | Doğrulama kaybı platoya girdiğinde öğrenme oranını yarıya indirir; modelin minimuma daha hassas adımlarla yaklaşmasını sağlar. |

### 4.2. Gerçekleşen Eğitim Süreci ve Sonuçlar

Model, **maksimum 50 epoch** sınırıyla eğitime başlamış ve aşağıdaki şekilde sonuçlanmıştır:

| Olay | Epoch | Değer |
|---|---|---|
| 🏆 En yüksek doğrulama doğruluğu (`val_accuracy`) | **30** | **%99.77** (0.9977) |
| 🛑 `EarlyStopping` tetiklendi | **44** | Eğitim sonlandırıldı |
| ♻️ Geri yüklenen en iyi ağırlıklar (`val_loss` bazlı) | **36** | `restore_best_weights=True` |

Bu üç rakam birlikte okunduğunda, modelin öğrenme dinamiği hakkında önemli bir mühendislik gözlemi ortaya çıkmaktadır: **`ModelCheckpoint` ve `EarlyStopping` farklı metrikleri izlemektedir.** `ModelCheckpoint`, en yüksek `val_accuracy`'yi epoch 30'da yakalamış ve bu anlık görüntüyü `en_iyi_meyve_modeli.keras` olarak kalıcı hale getirmiştir. Ancak `EarlyStopping`, doğruluk yerine `val_loss`'u izlemektedir; model epoch 30'dan sonra da doğruluğunu koruyarak `val_loss`'unu epoch 36'ya kadar iyileştirmeyi sürdürmüş, bu noktadan sonra 8 epoch (`patience=8`) boyunca daha fazla iyileşme gözlenmediği için sistem epoch 44'te eğitimi durdurmuş ve **epoch 36'daki en düşük kayıp değerine ait ağırlıkları** geri yüklemiştir.

Bu davranış; modelin sadece "doğru tahmin etmeyi" değil, aynı zamanda **tahminlerinde daha az kararsız (daha düşük kayıp/daha yüksek güven) hale gelmeyi** epoch 30'dan 36'ya kadar sürdürdüğünü; ardından öğrenmenin doğal olarak durağanlaştığını göstermektedir — bu da `EarlyStopping` ve `ReduceLROnPlateau` mekanizmalarının **tam olarak tasarlandıkları gibi** çalıştığının kanıtıdır.

### 4.3. Optimizasyon Ayarları

- **Optimizer:** Adam, başlangıç öğrenme oranı `learning_rate=1e-3`
- **Kayıp Fonksiyonu:** `sparse_categorical_crossentropy`
- **Metrik:** `accuracy`
- **Toplu İşlem Boyutu (Batch Size):** 32
- **Rastgelelik Tohumu (Seed):** 42 (tekrar üretilebilirlik için sabitlenmiştir)

---

## 5. 🔍 Açıklanabilir Yapay Zeka (XAI) ve Grad-CAM Entegrasyonu

### 5.1. Neden XAI?

%99.77 doğruluk gibi yüksek bir performans değeri, kalite kontrol gibi kritik bir endüstriyel uygulamada **tek başına yeterli güven oluşturmaz**. Model, "Çürük" kararını meyvenin gerçekten çürük bölgesine bakarak mı veriyor, yoksa arka plandaki bir gürültüye mi takılıyor? Bu sorunun yanıtlanması, sistemin üretim hattında güvenle kullanılabilmesi için zorunludur. Bu nedenle projeye, modelin kararını **görsel olarak doğrulayan** bir Grad-CAM (Gradient-weighted Class Activation Mapping) katmanı entegre edilmiştir.

### 5.2. Karşılaşılan Mühendislik Sorunu ve Profesyonel Çözümü

Grad-CAM uygulamalarında, özellikle Keras 3 ile birlikte yaygın olarak rastlanan bir hata olan **`"Sequential layer has no defined output"`** hatası, genellikle modelin katmanlarının `input_shape` belirtilmeden tek tek bir `Sequential` yapıya eklenmesinden kaynaklanır; bu durumda model grafiği, ilk gerçek çağrıya kadar tam olarak inşa edilmemiş olur ve ara katmanların `.output` özelliği erişilemez hale gelir.

Bu projede bu sorun **kök nedeninden** çözülmüştür: model, Bölüm 3'te belirtildiği gibi en baştan **Functional API** ile (`keras.Model(inputs=girdi, outputs=cikis)`) inşa edilmiştir. Functional modellerde hesaplama grafiği tanım anında tam olarak oluşturulduğu için her katmanın çıkışı (`.output`) doğrudan ve güvenli şekilde erişilebilir durumdadır.

Bu sayede Grad-CAM modeli, katmanları tek tek elle yeniden bağlamak gibi kırılgan bir yönteme **gerek kalmadan**, Keras'ın resmî Grad-CAM uygulamasında izlenen standart yaklaşımla inşa edilmiştir:

```python
grad_model = keras.Model(
    inputs=model.input,
    outputs=[model.get_layer("son_conv_katmani").output, model.output],
)
```

Bu tek satır, orijinal modelin ağırlıklarını kopyalamadan, mevcut hesaplama grafiğini yeniden kullanarak hem hedef evrişim katmanının aktivasyon haritasını hem de nihai tahmini **aynı ileri besleme (forward pass) içinde** döndüren yeni bir model oluşturur.

### 5.3. Isı Haritası Hesaplama Adımları

1. `tf.GradientTape` içinde, tahmin edilen sınıfın skoru hesaplanır.
2. Bu skorun, `son_conv_katmani`'nin çıkışına göre gradyanı alınır.
3. Gradyanlar mekansal eksenler boyunca ortalanarak (Global Average Pooling), her bir kanal için bir "önem ağırlığı" elde edilir.
4. Evrişim çıkışı bu ağırlıklarla ağırlıklandırılıp toplanır; negatif değerler ReLU ile elenir ve sonuç `[0, 1]` aralığına normalize edilir.
5. Elde edilen tek kanallı ısı haritası, OpenCV'nin `applyColorMap` (JET renk paleti) fonksiyonuyla renklendirilir ve `addWeighted` ile orijinal fotoğrafın **tam çözünürlüğü** üzerine yarı saydam olarak bindirilir.

Sonuç, Streamlit arayüzünde kullanıcıya **kırmızı/sarı bölgelerin modelin odaklandığı alanları temsil ettiği** bir görsel olarak sunulur — böylece model kararı, kara kutu (black-box) olmaktan çıkıp doğrulanabilir hale gelir.

---

## 6. 🚀 Kurulum ve Çalıştırma Talimatları

### 6.1. Gereksinimler

```bash
pip install tensorflow streamlit opencv-python pillow numpy
```

### 6.2. Klasör Yapısı

```
proje-klasoru/
├── dataset/
│   └── dataset/
│       ├── train/   # Her sınıf için bir alt klasör
│       └── val/
├── cnn_egitim.py        # Model eğitim betiği
├── cnn_arayuz.py         # Streamlit XAI arayüzü
├── en_iyi_meyve_modeli.keras       # Eğitim sonucu otomatik oluşur
└── README.md
```

### 6.3. Adım Adım Çalıştırma

**1. Modeli eğitin:**

```bash
python cnn_egitim.py
```

Eğitim tamamlandığında en iyi doğrulama performansına sahip model `en_iyi_meyve_modeli.keras` adıyla otomatik olarak kaydedilir.

**2. Web arayüzünü başlatın:**

```bash
streamlit run cnn_arayuz.py
```

**3. Tarayıcınızda açılan adrese gidin** (varsayılan olarak `http://localhost:8501`), bir meyve fotoğrafı yükleyin ve sonucu Grad-CAM ısı haritasıyla birlikte görüntüleyin.

---

## 7. 📌 Sonuç ve Değerlendirme

Bu proje kapsamında, sadece yüksek doğruluklu (%99.77 doğrulama doğruluğu) bir sınıflandırma modeli değil; aynı zamanda **endüstriyel gerçekçilik** (arka plan gürültüsüne dayanıklılık), **mühendislik disiplini** (otomatik, metriğe dayalı eğitim durdurma mekanizmaları) ve **şeffaflık** (Grad-CAM ile açıklanabilirlik) ilkelerini bir araya getiren uçtan uca bir sistem geliştirilmiştir. `EarlyStopping`, `ModelCheckpoint` ve `ReduceLROnPlateau` mekanizmalarının epoch 30-36-44 aralığında gösterdiği etkileşim, modelin keyfi bir epoch sayısıyla değil, **kendi öğrenme eğrisinin veriye dayalı analiziyle** sonlandırıldığını kanıtlamaktadır.

---

## 👥 Katkıda Bulunanlar

| Ad Soyad | Öğrenci No |
|---|---|
| İbrahim Berkay Tunçdemir | 23430070018 |
| Hüseyin Emre Şeker | 23430070073 |

**Ders Sorumlusu:** Hüseyin Yanık
**Bölüm:** Bilişim Sistemleri ve Teknolojileri
**Yüksekokul:** Erdemli Uygulamalı Teknoloji ve İşletmecilik Yüksekokulu
**Üniversite:** Mersin Üniversitesi
