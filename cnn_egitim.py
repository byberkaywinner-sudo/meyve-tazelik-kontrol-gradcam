"""
Yapay Zeka Destekli Akıllı Meyve Tazelik Kontrol Sistemi
==========================================================

Bu betik, 6 sınıflı (Taze/Çürük: Elma, Muz, Portakal) bir meyve tazelik
sınıflandırma modelini eğitmek için tasarlanmıştır.

Tasarım Kararları (Akademik Gerekçelendirme):
----------------------------------------------
1. Önceki denemelerde modelin sepet gibi karmaşık arka planlardaki doku ve
   renklere aldanarak hatalı sınıflandırma (False Positive) yapması, modelin
   "arka plan" ile "nesne" arasındaki ilişkiyi ezberlemesinden (overfitting)
   kaynaklanmaktadır. Bu sorunu çözmek için güçlü bir Data Augmentation
   katmanı eklenmiştir; bu katman modeli farklı açı, zoom ve kontrast
   senaryolarına zorlayarak gerçek dünya koşullarına karşı dayanıklı hale
   getirir.
2. BatchNormalization katmanları, her katmandaki aktivasyon dağılımını
   normalize ederek eğitimi hızlandırır ve modelin daha kararlı, genellenebilir
   özellikler öğrenmesini sağlar.
3. Kod, TensorFlow 2.16+ ile birlikte gelen Keras 3 standartlarına göre
   yazılmıştır (örn. .h5 yerine modern .keras formatı kullanılmıştır).

Yazar: [Öğrenci Adı]
Ders: [Ders Adı / Dönem Projesi]
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------------
# 1. SABİTLERİN TANIMLANMASI
# ----------------------------------------------------------------------------
# Kod tekrarını önlemek ve okunabilirliği artırmak için tüm "sihirli sayılar"
# (magic numbers) burada merkezi olarak tanımlanmıştır. PEP8 standardına göre
# sabitler büyük harfle yazılır.

VERI_SETI_YOLU_EGITIM = "dataset/dataset/train"
VERI_SETI_YOLU_DOGRULAMA = "dataset/dataset/val"
GORUNTU_BOYUTU = (224, 224)
GORUNTU_SEKLI = (224, 224, 3)
TOPLU_ISLEM_BOYUTU = 32
MAKSIMUM_EPOCH = 50  # EarlyStopping sayesinde gerçek eğitim daha kısa sürebilir.
RASTGELELIK_TOHUMU = 42  # Sonuçların tekrar üretilebilir (reproducible) olması için.
EN_IYI_MODEL_DOSYA_ADI = "en_iyi_meyve_modeli.keras"


def veri_setlerini_hazirla():
    """
    Eğitim ve doğrulama veri setlerini diskten yükler.

    Görüntüler klasör yapısından otomatik olarak etiketlenir (her alt klasör
    bir sınıf kabul edilir). Veri setlerinin her iki yüklemede de aynı
    sınıf sırasına sahip olması için sabit bir 'seed' (tohum) değeri
    kullanılmıştır.

    Returns:
        tuple: (egitim_veri_seti, dogrulama_veri_seti, sinif_isimleri)
    """
    egitim_veri_seti = keras.utils.image_dataset_from_directory(
        VERI_SETI_YOLU_EGITIM,
        image_size=GORUNTU_BOYUTU,
        batch_size=TOPLU_ISLEM_BOYUTU,
        seed=RASTGELELIK_TOHUMU,
    )

    dogrulama_veri_seti = keras.utils.image_dataset_from_directory(
        VERI_SETI_YOLU_DOGRULAMA,
        image_size=GORUNTU_BOYUTU,
        batch_size=TOPLU_ISLEM_BOYUTU,
        seed=RASTGELELIK_TOHUMU,
    )

    sinif_isimleri = egitim_veri_seti.class_names
    print(f"Tespit Edilen Sınıflar ({len(sinif_isimleri)} adet): {sinif_isimleri}")

    # Eğitim hızını artırmak için veri setini önbelleğe alıyoruz (caching) ve
    # bir sonraki batch'i GPU/CPU işlem yaparken arka planda hazırlıyoruz
    # (prefetching). Bu, modern TensorFlow performans standardıdır.
    egitim_veri_seti = egitim_veri_seti.cache().prefetch(
        buffer_size=tf.data.AUTOTUNE
    )
    dogrulama_veri_seti = dogrulama_veri_seti.cache().prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    return egitim_veri_seti, dogrulama_veri_seti, sinif_isimleri


def veri_artirma_katmanini_olustur():
    """
    Modelin arka plan gürültüsüne (sepet, masa, farklı zeminler vb.) karşı
    dayanıklı hale gelmesini sağlayacak Data Augmentation katmanını oluşturur.

    Bu katman SADECE eğitim sırasında aktif olur; doğrulama ve test
    aşamasında devre dışı kalır (Keras bunu otomatik yönetir). Amaç, modele
    her epoch'ta meyvenin biraz farklı bir versiyonunu göstererek "meyveyi"
    öğrenmesini, "arka planı" ezberlemesini engellemektir.

    Returns:
        keras.Sequential: Veri artırma adımlarını içeren katman grubu.
    """
    veri_artirma = keras.Sequential(
        [
            # Görüntüyü yatayda rastgele çevirir (meyvenin sepetin sağında
            # veya solunda olması sonucu değiştirmemeli).
            layers.RandomFlip("horizontal"),
            # Hafif döndürme: meyvenin her zaman dik durmayacağı senaryoları
            # simüle eder.
            layers.RandomRotation(0.15),
            # Yakınlaştırma/uzaklaştırma: farklı çekim mesafelerini taklit eder.
            layers.RandomZoom(0.2),
            # Kontrast değişimi: farklı ışık koşullarındaki (gölgeli sepet,
            # parlak ortam ışığı vb.) görüntülere karşı dayanıklılık sağlar.
            layers.RandomContrast(0.2),
            # Öteleme: meyvenin kareye tam ortalanmamış olabileceği,
            # kenarda kalan görüntüleri simüle eder.
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        ],
        name="veri_artirma_katmani",
    )
    return veri_artirma


def cnn_modelini_olustur(sinif_sayisi):
    """
    Tazelik tespiti için Evrişimli Sinir Ağı (CNN) mimarisini kurar.

    Mimari, her evrişim bloğunda Conv2D -> BatchNormalization -> MaxPooling
    sırasını izler. BatchNormalization, aktivasyon değerlerini normalize
    ederek hem eğitimi hızlandırır hem de modelin arka plandaki rastgele
    doku/renk varyasyonlarına karşı daha az hassas, daha genellenebilir
    özellikler öğrenmesine yardımcı olur.

    Args:
        sinif_sayisi (int): Çıkış katmanındaki nöron sayısı (toplam sınıf adedi).

    Returns:
        keras.Model: Derlenmeye hazır, inşa edilmiş CNN modeli.
    """
    girdi = keras.Input(shape=GORUNTU_SEKLI, name="girdi_katmani")

    # Veri artırma katmanı, modelin bir parçası olarak eklenir. Bu sayede
    # model kaydedildiğinde (.keras formatında) augmentation mantığı da
    # modelle birlikte taşınır.
    x = veri_artirma_katmanini_olustur()(girdi)

    # Piksel değerlerini [0, 255] aralığından [0, 1] aralığına ölçekleriz.
    # Bu, optimizasyon algoritmasının (Adam) daha kararlı çalışmasını sağlar.
    x = layers.Rescaling(1.0 / 255)(x)

    # --- 1. Evrişim Bloğu ---
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # --- 2. Evrişim Bloğu ---
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # --- 3. Evrişim Bloğu ---
    # 'son_conv_katmani' ismi korunmuştur: Grad-CAM ısı haritası analizinde
    # bu katmanın çıkışı referans alınacaktır.
    x = layers.Conv2D(
        128, (3, 3), padding="same", activation="relu", name="son_conv_katmani"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # --- Sınıflandırma (Karar Verme) Bloğu ---
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    # Dropout oranı %50: Nöronların yarısını rastgele "kapatarak" modelin
    # belirli nöron kombinasyonlarına aşırı bağımlı (overfit) olmasını engeller.
    x = layers.Dropout(0.5)(x)

    cikis = layers.Dense(sinif_sayisi, activation="softmax", name="cikis_katmani")(x)

    model = keras.Model(inputs=girdi, outputs=cikis, name="meyve_tazelik_modeli")
    return model


def callback_listesini_olustur():
    """
    Eğitim sürecini izleyip otomatik olarak optimize edecek callback'leri
    (geri çağırım fonksiyonları) oluşturur.

    - EarlyStopping: Doğrulama kaybı (val_loss) belirli bir sabır (patience)
      süresi boyunca iyileşmezse eğitimi otomatik olarak durdurur. Bu, hem
      zaman kazandırır hem de modelin ezberlemeye başladığı noktadan sonra
      eğitime devam etmesini önler.
    - ModelCheckpoint: Eğitim boyunca elde edilen EN İYİ doğrulama
      performansına sahip model ağırlıklarını diske kaydeder. Böylece son
      epoch'ta model performansı düşmüş olsa bile, en başarılı versiyon
      elimizde kalır.

    Returns:
        list: Keras callback nesnelerinin listesi.
    """
    erken_durdurma = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    )

    model_kontrol_noktasi = keras.callbacks.ModelCheckpoint(
        filepath=EN_IYI_MODEL_DOSYA_ADI,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    # Öğrenme oranını, doğrulama kaybı plato yaptığında otomatik düşürür.
    # Bu, modelin minimum noktaya daha hassas adımlarla yaklaşmasını sağlar.
    ogrenme_orani_azaltma = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    )

    return [erken_durdurma, model_kontrol_noktasi, ogrenme_orani_azaltma]


def main():
    """Eğitim sürecinin tüm adımlarını sırayla yürüten ana fonksiyon."""
    # Tekrar üretilebilirlik (reproducibility) için rastgelelik tohumunu sabitliyoruz.
    keras.utils.set_random_seed(RASTGELELIK_TOHUMU)

    # 1. Adım: Veri setlerini hazırlama.
    egitim_veri_seti, dogrulama_veri_seti, sinif_isimleri = veri_setlerini_hazirla()
    sinif_sayisi = len(sinif_isimleri)

    # 2. Adım: Modeli inşa etme.
    model = cnn_modelini_olustur(sinif_sayisi)
    model.summary()

    # 3. Adım: Modeli derleme (compile).
    # Etiketler tam sayı (integer) formatında olduğu için
    # 'sparse_categorical_crossentropy' kayıp fonksiyonu kullanılır.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 4. Adım: Callback'leri hazırlama.
    callback_listesi = callback_listesini_olustur()

    # 5. Adım: Eğitimi başlatma.
    print("-" * 60)
    print("Eğitim süreci başlatılıyor...")
    print("-" * 60)
    gecmis = model.fit(
        egitim_veri_seti,
        validation_data=dogrulama_veri_seti,
        epochs=MAKSIMUM_EPOCH,
        callbacks=callback_listesi,
    )

    # 6. Adım: Nihai modeli kaydetme.
    # Not: ModelCheckpoint zaten en iyi modeli kaydetmiştir; bu satır ise
    # eğitimin son durumundaki modeli yedek olarak modern .keras formatında
    # saklar (Keras 3 standardı; eski .h5 formatı yerine önerilir).
    model.save("son_epoch_meyve_modeli.keras")

    print("-" * 60)
    print(f"Eğitim tamamlandı. En iyi model '{EN_IYI_MODEL_DOSYA_ADI}' "
          "olarak kaydedildi.")
    print(f"Eğitilen sınıflar: {sinif_isimleri}")
    print("-" * 60)

    return gecmis


if __name__ == "__main__":
    main()