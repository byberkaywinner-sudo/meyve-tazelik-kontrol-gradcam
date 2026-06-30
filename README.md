<div align="center">

# 🍎🍌🍊 Yapay Zeka ve XAI (Grad-CAM) Destekli Akıllı Meyve Tazelik Kontrol Sistemi

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-red?logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Arayüz-FF4B4B?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Güvenlik%20Katmanı-5C3EE8?logo=opencv&logoColor=white)
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
6. [Endüstriyel Güvenlik Katmanı (Açık-Küme Güvenlik Ağı)](#6--endüstriyel-güvenlik-katmanı-açık-küme-güvenlik-ağı)
7. [Kurulum ve Çalıştırma Talimatları](#7--kurulum-ve-çalıştırma-talimatları)
8. [Sonuç ve Değerlendirme](#8--sonuç-ve-değerlendirme)

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

Projenin nihai çıktısı sadece bir sınıflandırma etiketi değildir; aynı zamanda **kararın nedenini görselleştiren** bir Açıklanabilir Yapay Zeka (XAI) katmanı ve gerçek dünya koşullarındaki hatalı girdilere karşı sistemi koruyan bir **Endüstriyel Güvenlik Katmanı** (bkz. Bölüm 6) içeren, Streamlit tabanlı interaktif bir web arayüzüdür.

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
| `RandomTranslation` | `0.1 / 0.1` | Meyvenin kamera karesinin merkezine tam hizalanmamış, kenarda kalan görüntülerini simüle eder. |

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

## 6. 🛡️ Endüstriyel Güvenlik Katmanı (Açık-Küme Güvenlik Ağı)

> **Bu bölüm, projeyi bir "ders ödevi prototipi"nden, gerçek dünya koşullarına dayanıklı bir "endüstriyel sistem"e dönüştüren katmanı anlatır.**

### 6.1. Çözülen Temel Problem: Kapalı-Küme (Closed-World) Tuzağı

Eğitilen model, 6 sınıflık **kapalı bir dünya (closed-world)** varsayımıyla çalışır; yani "bunların hiçbiri değil" diyebileceği bir çıkış nöronu **yoktur**. Softmax fonksiyonunun matematiksel doğası gereği, sisteme bir meyve değil de **dağılım-dışı (Out-of-Distribution / OOD)** bir görüntü — örneğin bir insan yüzü, el veya bir kurum logosu — verildiğinde dahi, olasılıkların toplamı 1'e normalize edilir ve model, gördüğü nesneyi **mecburen** en yakın bildiği dokuya yüksek bir güvenle yuvarlar. Bu, modelin bir "eğitim hatası" değil; tüm kapalı-küme sınıflandırıcıların **yapısal bir sınırlamasıdır**.

Bu sınırlama, canlı bir kalite kontrol hattında ciddi bir güvenilirlik riski oluşturur: kameraya yanlışlıkla bir operatörün eli geçtiğinde, sistemin bunu "%100 Çürük Muz" olarak raporlaması kabul edilemez. Bu projede söz konusu risk, modeli yeniden eğitmeden, modelin çıktısını **bağımsız olarak denetleyen** çok katmanlı bir güvenlik ağı ile yönetilmiştir.

### 6.2. Çok Katmanlı Savunma Mimarisi

Güvenlik ağı, "**gerçek bir meyve asla yanlışlıkla reddedilmemeli, ancak meyve olmayan hiçbir nesne de sınıflandırılmamalı**" tasarım önceliğiyle, dört bağımsız kontrolden oluşur. Bu kontrollerin hiçbiri modelin ağırlıklarını değiştirmez; tamamı modelin çıktısını ve ham görüntüyü ek bir denetim katmanından geçirir.

| # | Katman | Teknik | Görevi |
|---|---|---|---|
| 1 | **Ana Savunma** | OpenCV Haar Cascade (yapısal yüz tespiti) | Görüntüde insan yüzü varsa, modelin tahminine **hiç bakmadan** işlemi durdurur. |
| 2 | **İkincil Yedek** | YCrCb cilt tonu filtresi (yüksek eşik) | Yüz dedektörünün kaçırdığı aşırı durumlarda devreye giren, renk tabanlı bir yedek. |
| 3 | **Güven Kilidi** | Confidence Threshold (%65) | Modelin en yüksek güveni düşükse, "kararsız" durumunu yakalar. |
| 4 | **Marj Kilidi** | Top-1 / Top-2 Margin (%15) | İki en olası sınıf birbirine çok yakınsa, modelin salınımını yakalar. |

#### 🥇 Katman 1 — Ana Savunma: Yapısal Yüz Tespiti (Haar Cascade)

Güvenlik ağının birincil ve en güçlü hattı, OpenCV ile birlikte hazır gelen klasik **Haar Cascade** yüz dedektörüdür. Bu katmanın seçilmesinin ardındaki mühendislik gerekçesi kritik öneme sahiptir: **renk tabanlı yöntemler, sıcak insan teni tonu ile sarı/kahverengi (örn. çürük muz) meyve tonlarını birbirinden güvenilir şekilde ayıramaz.** Buna karşılık Haar Cascade, görüntünün rengine değil, **yapısal desenine** (göz-burun-ağız geometrisine) bakar.

Bu yaklaşımın doğrudan sonucu şudur: dedektör yalnızca **gerçekten bir yüz yapısı** gördüğünde tetiklenir ve bir meyveyi — rengi ne olursa olsun — **asla "yüz" sanmaz**. Bu özellik, "gerçek meyveyi asla yanlışlıkla reddetme" önceliğiyle birebir örtüşür ve bu katmanı, yanlış pozitif riski neredeyse sıfır olan en güvenli OOD filtresi yapar. Sistem; hem önden bakan yüzleri (`frontalface`) hem de hafif yan profilleri (`profileface`, ayna simetrisiyle birlikte) tarar.

#### 🥈 Katman 2 — İkincil Yedek: YCrCb Cilt Tonu Filtresi

Yüz dedektörünün zor açılarda (aşırı yakın, kısmen görünen yüz veya yalnızca el) bir yüzü kaçırma ihtimaline karşı, modelden bağımsız ikinci bir güvenlik ağı eklenmiştir. Bu filtre, görüntüyü **YCrCb renk uzayına** çevirir (cilt tonu için HSV'den daha dar ve güvenilir bir aralık sunar) ve görüntüdeki en büyük **bitişik (connected component)** cilt-tonu bölgesinin oranını hesaplar.

"Gerçek meyveyi asla reddetme" önceliği gereği bu filtrenin eşiği **bilerek çok yüksek (%60)** tutulmuştur: yalnızca kare neredeyse tamamen tek parça bir cilt tonuyla kaplandığında devreye girer. Bu sayede sarı muz, turuncu veya kahverengi çürük portakal gibi gerçek meyveler bu yedek filtre tarafından **yanlışlıkla reddedilmez**. Tek piksellik gürültüyü temizlemek için ayrıca morfolojik açma (opening) işlemi uygulanır.

#### 🥉 Katman 3 ve 4 — Güven ve Marj Kilitleri

Görüntü bir yüz/cilt içermese bile, model gerçek bir meyve karşısında dahi kararsız kalabilir (örneğin renk yanlılığı nedeniyle bir sınıf ile diğeri arasında bölünmesi). Bu durumları yakalamak için iki istatistiksel kilit eklenmiştir:

- **Güven Kilidi (%65):** Modelin en yüksek tahmin olasılığı bu eşiğin altındaysa, sonuç gösterilmez. %50 civarı bir skor, sınıflar arası neredeyse "yazı-tura" anlamına gelir.
- **Marj Kilidi (%15):** En yüksek iki olasılık arasındaki fark bu eşiğin altındaysa (örn. %42'ye karşı %38), model iki sınıf arasında salınıyor demektir; bu da düşük güvenilirlik sinyali olarak değerlendirilir.

### 6.3. Şeffaflık: Hata Ayıklama Paneli

Bir görüntü herhangi bir güvenlik kontrolünden geçemediğinde, kullanıcıya kırmızı bir teknik hata yerine **nazik ve eğitici bir uyarı** gösterilir (örn. "Görüntüde bir insan yüzü tespit edildi; lütfen kameraya yalnızca meyveyi gösterin"). Bu uyarının altında, açılabilir bir **teknik detay paneli** yer alır; bu panel, tespit edilen yüz durumunu, modelin ham güven skorunu, top-1/top-2 marjını ve cilt tonu oranını şeffaf biçimde gösterir. Bu, sistemin kararlarını gizlemek yerine **açıklanabilir kılan**, XAI felsefesiyle tutarlı bir tasarım tercihidir.

---

## 7. 🚀 Kurulum ve Çalıştırma Talimatları

### 7.1. Gereksinimler

```bash
pip install tensorflow streamlit opencv-python pillow numpy
```

> **Not:** Güvenlik katmanındaki Haar Cascade yüz dedektörü için gerekli model dosyaları (`haarcascade_frontalface_default.xml`, `haarcascade_profileface.xml`) `opencv-python` paketiyle birlikte **otomatik olarak** gelir; ayrı bir indirme gerektirmez.

### 7.2. Klasör Yapısı

```
proje-klasoru/
├── dataset/
│   └── dataset/
│       ├── train/   # Her sınıf için bir alt klasör
│       └── val/
├── cnn_egitim.py             # Model eğitim betiği
├── cnn_arayuz.py              # Streamlit XAI + Güvenlik Katmanı arayüzü
├── en_iyi_meyve_modeli.keras  # Eğitim sonucu otomatik oluşur
└── README.md
```

### 7.3. Adım Adım Çalıştırma

**1. Modeli eğitin:**

```bash
python cnn_egitim.py
```

Eğitim tamamlandığında en iyi doğrulama performansına sahip model `en_iyi_meyve_modeli.keras` adıyla otomatik olarak kaydedilir.

**2. Web arayüzünü başlatın:**

```bash
streamlit run cnn_arayuz.py
```

**3. Tarayıcınızda açılan adrese gidin** (varsayılan olarak `http://localhost:8501`). Arayüz iki çalışma modu sunar:
- **📁 Fotoğraf Yükle:** Diskten bir meyve fotoğrafı seçin.
- **📷 Kameradan Çek:** Canlı kamera ile anlık fotoğraf çekin.

Her iki modda da sistem; meyveyi sınıflandırır, Grad-CAM ısı haritasını üretir ve sonucu güvenlik katmanından geçirerek gösterir.

---

## 8. 📌 Sonuç ve Değerlendirme

Bu proje kapsamında, sadece yüksek doğruluklu (%99.77 doğrulama doğruluğu) bir sınıflandırma modeli değil; aynı zamanda **endüstriyel gerçekçilik** (arka plan gürültüsüne dayanıklılık), **mühendislik disiplini** (otomatik, metriğe dayalı eğitim durdurma mekanizmaları), **şeffaflık** (Grad-CAM ile açıklanabilirlik) ve **gerçek dünya dayanıklılığı** (çok katmanlı açık-küme güvenlik ağı) ilkelerini bir araya getiren uçtan uca bir sistem geliştirilmiştir.

`EarlyStopping`, `ModelCheckpoint` ve `ReduceLROnPlateau` mekanizmalarının epoch 30-36-44 aralığında gösterdiği etkileşim, modelin keyfi bir epoch sayısıyla değil, **kendi öğrenme eğrisinin veriye dayalı analiziyle** sonlandırıldığını kanıtlamaktadır. Bölüm 6'da anlatılan güvenlik katmanı ise, modelin kapalı-küme yapısal sınırlamasının farkında olunduğunu ve bu sınırlamanın sistem seviyesinde profesyonelce yönetildiğini göstermektedir — ki bu, akademik bir prototipi gerçek bir mühendislik ürününden ayıran temel farktır.

### 8.1. Bilinen Sınırlamalar ve Gelecek Çalışmalar

Sistem, tekil ve net görüntülerde yüksek başarı sergilemekle birlikte, mevcut modelin **renk yanlılığı (color bias)** taşıdığı gözlemlenmiştir; bu, modelin "şekil" yerine "renk" özelliklerine ağırlık verdiği eğitim verisinin bir sonucudur. Gelecek çalışmalarda; (1) sınıflar arası renk dengesi gözetilmiş, daha büyük ve çeşitli bir veri seti, (2) transfer öğrenme (MobileNet/EfficientNet gibi önceden eğitilmiş omurgalar) ve (3) sahnedeki meyveleri önce tespit edip kırpan bir YOLO ön-aşaması ile beslenen hibrit bir mimari hedeflenmektedir. Bu iyileştirmeler, mevcut güçlü güvenlik altyapısı korunarak modelin sınıflandırma çekirdeğini güçlendirecektir.

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
