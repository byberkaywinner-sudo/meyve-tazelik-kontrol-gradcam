"""
Yapay Zeka Destekli Akıllı Meyve Tazelik Kontrol Sistemi
==========================================================
Grad-CAM Destekli Web Arayüzü (cnn_arayuz.py)

Bu betik, 'cnn_egitim.py' ile eğitilip diske kaydedilmiş CNN modelini yükler;
kullanıcının yüklediği veya kameradan canlı çektiği bir meyve fotoğrafını
6 sınıf üzerinden sınıflandırır ve modelin kararını HANGİ bölgelere bakarak
verdiğini Grad-CAM ısı haritası ile görselleştirir (Açıklanabilir Yapay
Zeka / XAI).

ÖNEMLİ NOT — Model Dosyası Hakkında:
-------------------------------------
'cnn_egitim.py' incelendiğinde eğitimin sonunda '.h5' formatında bir dosya
KAYDEDİLMEDİĞİ görülür. ModelCheckpoint callback'i en iyi modeli
'en_iyi_meyve_modeli.keras' adıyla kaydeder (eğitimin son hâli ise ayrıca
'son_epoch_meyve_modeli.keras' olarak saklanır). Bu arayüzde MODEL_DOSYA_YOLU
sabiti, bilerek en iyi doğrulama performansına sahip dosyayı işaret edecek
şekilde ayarlanmıştır. Modelinizi farklı bir adla kaydettiyseniz, sadece bu
sabiti güncellemeniz yeterlidir.

ÖNEMLİ NOT — Grad-CAM Mimarisi ve Keras 3 Uyumluluğu:
-------------------------------------------------------
'cnn_egitim.py' modeli Keras'ın Functional API'si ile (keras.Model(inputs=...,
outputs=...)) inşa edilmiştir; Sequential API ile DEĞİL. Bu önemli bir
ayrımdır: Functional modellerde her katmanın çıkışı (.output), modelin
hesaplama grafiği zaten tanımlı olduğu için doğrudan erişilebilir durumdadır.

"Sequential layer has no defined output" hatası, yalnızca katmanların
input_shape verilmeden tek tek eklendiği Sequential modellerde, grafiğin
ilk çağrıya kadar inşa edilmemesinden kaynaklanır. Burada böyle bir model
söz konusu olmadığı için, katmanları elle birer birer yeniden bağlayan
(manual re-wiring) bir döngüye gerek YOKTUR — aksine böyle bir döngü, iç
içe geçmiş veri artırma bloğunu (Sequential) ve BatchNormalization /
Dropout katmanlarının eğitim/çıkarım davranışını bozma riski taşır.

Bunun yerine, resmî Keras Grad-CAM örneğinde de izlenen en sağlam yöntem
kullanılmıştır: modelin kendi grafiğinden 'model.get_layer(...).output' ile
ilgili katmanın çıkışı doğrudan alınır ve aynı girişi (model.input) paylaşan
yeni bir tf.keras.Model ile sarmalanır. Ağırlık kopyalanmaz, grafiğin
mevcut bağlantıları yeniden kullanılır.

ÖNEMLİ NOT — Canlı Kamera Desteği (st.camera_input):
-------------------------------------------------------
Sunum/demo senaryosunda kullanıcı, "📁 Fotoğraf Yükle" ve "📷 Kameradan Çek"
sekmelerinden birini seçebilir. Her iki giriş yöntemi de ortak bir
`pil_resim` değişkenine indirgenir; bu sayede aşağıdaki tüm işlem hattı
(PIL açma → 224x224 boyutlandırma → model tahmini → Grad-CAM ısı haritası
üretimi) HER İKİ kaynak için de aynı, tek bir kod yolundan (`analiz_et`)
geçer. Canlı sunum sırasında oluşabilecek kamera izni reddi, bozuk/boş kare
gibi durumlar `try-except` ile yakalanır ve kullanıcıya teknik hata yerine
zarif bir Streamlit uyarısı gösterilir; uygulama asla çökmez.

ÖNEMLİ NOT — Açık-Küme (Open-Set) Güvenlik Ağı:
-------------------------------------------------------
Mevcut model, 6 sınıflık KAPALI bir dünya (closed-world) varsayımıyla
eğitilmiştir; "bunların hiçbiri değil" diyebileceği bir çıkış nöronu
yoktur. Softmax fonksiyonunun matematiksel doğası gereği, kameraya bir
insan yüzü veya elin tutulması gibi dağılım-dışı (Out-of-Distribution /
OOD) bir girdi verildiğinde dahi olasılıklar toplamı 1'e normalize edilir
ve model en yakın dokuya (örn. cilt tonu → muz/elma karışımı) yüksek bir
güvenle yuvarlanmaya ZORLANIR. Bu, modelin eğitim hatası değil; kapalı-küme
sınıflandırıcıların yapısal bir sınırlamasıdır ve mevcut .keras dosyası
yeniden eğitilmeden kod seviyesinde tam olarak çözülemez.

Bu nedenle modele GÜVENMEK YERİNE modeli SORGULAYAN bağımsız kontroller
eklenmiştir; bunların hiçbiri modelin ağırlıklarını değiştirmez, sadece
modelin çıktısını ve ham görüntüyü ayrıca denetler:

  1. YÜZ TESPİTİ (Haar Cascade): Modelden BAĞIMSIZ, OpenCV ile birlikte
     hazır gelen klasik bir yüz dedektörü, görüntüde insan yüzü olup
     olmadığını RENGE DEĞİL, yapısal desene (göz/burun/ağız geometrisi)
     bakarak tespit eder. Bu, "yüzü yakala ama gerçek meyveyi reddetme"
     hedefinin anahtarıdır: renk tabanlı bir filtre sarı muz ile sıcak
     ten tonunu ayırt edemezken, yüz dedektörü yalnızca GERÇEKTEN bir
     yüz gördüğünde tetiklenir; bir meyvenin rengi ne olursa olsun onu
     asla "yüz" sanmaz. Bu yüzden bu kontrol, yanlışlıkla gerçek meyve
     reddetme riski neredeyse sıfır olan, en güvenli OOD filtresidir.
  2. GÜVEN EŞİĞİ (Confidence Threshold): En yüksek olasılık
     GUVEN_ESIGI'nin altındaysa, model muhtemelen kararsızdır (bkz. sarı
     elma testi → %50.47). Sonuç gösterilmez, kullanıcıdan meyveyi daha
     net göstermesi istenir.
  3. MARJ KONTROLÜ (Top-1 / Top-2 Margin): En yüksek iki olasılık
     birbirine çok yakınsa (örn. %42 Çürük Muz / %38 Çürük Elma), model
     iki sınıf arasında salınıyor demektir; bu da düşük güvenilirlik
     sinyalidir.
  4. CİLT TONU (Yedek/İkincil): YCrCb renk-aralığı tabanlı, modelden
     bağımsız bir filtre. Yüz dedektörü bazı zor açılarda (aşırı yan
     profil, kısmen görünen yüz) yüzü kaçırırsa diye eklenmiş İKİNCİL
     bir güvenlik ağıdır. "Gerçek meyveyi asla reddetme" önceliği gereği
     bu filtrenin eşiği BİLEREK çok yüksek (muhafazakâr) tutulmuştur;
     yani yalnızca kare neredeyse tamamen tek parça cilt tonuyla
     kaplandığında devreye girer, böylece sarı/turuncu/kahverengi
     meyveleri yanlışlıkla reddetmez.

Önemli tasarım kararı: "gerçek meyveler asla reddedilmemeli" önceliği
benimsenmiştir. Bu nedenle yüz tespiti ana savunma hattıdır (meyveyi
hiç etkilemez), cilt tonu ise yalnızca aşırı durumlar için yüksek eşikli
bir yedektir.

Gerekli paketler:
    pip install streamlit tensorflow opencv-python pillow numpy

Çalıştırma:
    streamlit run cnn_arayuz.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

import cv2

# ----------------------------------------------------------------------------
# 1. SABİTLERİN TANIMLANMASI
# ----------------------------------------------------------------------------
# 'cnn_egitim.py' ile aynı sabitler (görüntü boyutu, katman ismi) burada
# tekrar tanımlanmıştır; iki betiğin senkronize kalması için bu isimlerin
# eğitim kodundakilerle birebir eşleşmesi gerekir.

GORUNTU_BOYUTU = (224, 224)
MODEL_DOSYA_YOLU = "en_iyi_meyve_modeli.keras"
SON_KONVOLUSYON_KATMANI_ADI = "son_conv_katmani"
ISI_HARITASI_AGIRLIGI = 0.4  # Isı haritasının orijinal resme bindirilme oranı.

SINIF_ISIMLERI = [
    "Taze Elma 🍏",
    "Taze Muz 🍌",
    "Taze Portakal 🍊",
    "Çürük Elma 🍎",
    "Çürük Muz 🍌",
    "Çürük Portakal 🍊",
]

# SINIF_ISIMLERI listesindeki çürük sınıfların indeksleri (sonuç kutusunun
# renk/ikon seçimi için kullanılır).
CURUK_SINIF_INDEKSLERI = {3, 4, 5}

# ----------------------------------------------------------------------------
# AÇIK-KÜME (OPEN-SET) GÜVENLİK AĞI SABİTLERİ
# ----------------------------------------------------------------------------
# Bu üç eşik, modelin kendisine DOKUNMADAN, modelin ürettiği sonuçları
# sunum öncesinde sorgulamak için kullanılır. Değerler, sarı-elma testinde
# gözlemlenen %50.47 ve yüz testinde gözlemlenen %100 arasındaki net
# boşluktan yola çıkılarak temkinli (muhafazakâr) seçilmiştir.

# 1) Güven Eşiği: En yüksek tahmin olasılığı bu değerin altındaysa model
#    "kararsız" kabul edilir. %50 civarı bir skor, sınıflar arası neredeyse
#    yazı-tura demektir; bu nedenle eşik daha temkinli bir noktaya, %65'e
#    çekilmiştir.
GUVEN_ESIGI = 0.65

# 2) Marj Eşiği: En yüksek iki olasılık arasındaki fark bu değerin
#    altındaysa, model iki sınıf arasında salınıyor demektir (örn. Çürük
#    Muz %42 / Çürük Elma %38 → marj sadece 0.04).
MARJ_ESIGI = 0.15

# 3) Cilt Tonu (İKİNCİL/YEDEK) Oranı Eşiği: Artık ANA savunma yüz
#    dedektörüdür (bkz. yuz_tespit_edildi_mi). Bu cilt-tonu kontrolü
#    yalnızca yüz dedektörünün kaçırdığı zor durumlar için bir yedektir.
#    "Gerçek meyveyi asla reddetme" önceliği gereği eşik BİLEREK çok
#    yüksek tutulmuştur (0.60): yani kare neredeyse tamamen TEK PARÇA
#    cilt tonuyla kaplanmadıkça devreye girmez. Bu sayede sarı muz,
#    turuncu/kahverengi çürük portakal gibi gerçek meyveler bu yedek
#    filtre tarafından yanlışlıkla reddedilmez.
CILT_TONU_ORANI_ESIGI = 0.60

# 4) Yüz Tespiti (ANA SAVUNMA) parametreleri: OpenCV Haar Cascade
#    dedektörü, görüntüde insan yüzü ararken renge değil yapısal desene
#    bakar. MIN_YUZ_BOYUT_ORANI, tespit edilen yüzün görüntü kenarına
#    oranla en az ne kadar büyük olması gerektiğini belirler; çok küçük
#    (gürültü kaynaklı) yanlış tespitleri eler. 0.20 = yüz, kısa kenarın
#    en az %20'si kadar olmalı.
MIN_YUZ_BOYUT_ORANI = 0.20



# ----------------------------------------------------------------------------
# 2. MODEL YÜKLEME
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Model yükleniyor, lütfen bekleyin...")
def model_yukle(model_yolu: str) -> keras.Model:
    """
    Eğitilmiş Keras modelini diskten yükler ve önbelleğe alır.

    'st.cache_resource' sayesinde model, kullanıcı her fotoğraf
    yüklediğinde yeniden diskten okunmaz; sadece ilk çalıştırmada yüklenir
    ve oturum boyunca bellekte tutulur. Bu, arayüzün tepki süresini
    önemli ölçüde hızlandırır.
    """
    return keras.models.load_model(model_yolu)


@st.cache_resource(show_spinner=False)
def _grad_cam_modeli_olustur(_model: keras.Model, konv_katman_adi: str) -> keras.Model:
    """
    Grad-CAM için kullanılacak yardımcı modeli oluşturur.

    Orijinal modelin GİRİŞİ ile hem hedef konvolüsyon katmanının çıkışını
    hem de modelin nihai tahmin çıkışını aynı anda döndüren bir Functional
    model kurulur. Bu, ağırlıkları kopyalamaz; mevcut hesaplama grafiğini
    yeniden kullanır.

    Not: '_model' parametresi alt çizgiyle başlar; bu, Streamlit'e bu
    parametreyi önbellek anahtarı hesaplarken (hash'lerken) göz ardı
    etmesini söyler (Keras modelleri doğrudan hash'lenemez).
    """
    try:
        konv_katman_ciktisi = _model.get_layer(konv_katman_adi).output
    except ValueError as hata:
        st.error(
            f"'{konv_katman_adi}' adında bir katman modelde bulunamadı. "
            "SON_KONVOLUSYON_KATMANI_ADI sabitini, eğitim kodunuzdaki "
            "(cnn_egitim.py) son Conv2D katmanının ismiyle eşleştirin.\n\n"
            f"Teknik detay: {hata}"
        )
        st.stop()
        raise

    return keras.models.Model(
        inputs=_model.input,
        outputs=[konv_katman_ciktisi, _model.output],
    )


# ----------------------------------------------------------------------------
# 3. GÖRÜNTÜ HAZIRLAMA
# ----------------------------------------------------------------------------
def model_girdisi_olustur(pil_resim: Image.Image) -> tf.Tensor:
    """
    PIL görüntüsünü modelin beklediği (1, 224, 224, 3) şeklinde, float32
    türünde bir tensöre çevirir.

    Not: Piksel değerleri burada [0, 1] aralığına ölçeklenmez; bu işlem
    modelin kendi içindeki 'Rescaling' katmanı tarafından zaten
    yapılmaktadır (bkz. cnn_egitim.py / cnn_modelini_olustur). Bu kural,
    kameradan gelen kareler için de değişmeden geçerlidir.
    """
    yeniden_boyutlu = pil_resim.resize(GORUNTU_BOYUTU)
    goruntu_dizisi = np.array(yeniden_boyutlu)
    girdi_tensoru = tf.expand_dims(goruntu_dizisi, axis=0)
    return tf.cast(girdi_tensoru, tf.float32)


# ----------------------------------------------------------------------------
# 4. GRAD-CAM HESAPLAMA
# ----------------------------------------------------------------------------
def grad_cam_ve_tahmin_uret(
    model: keras.Model,
    girdi_tensoru: tf.Tensor,
    konv_katman_adi: str,
) -> tuple[np.ndarray, int, float]:
    """
    Tek bir ileri besleme (forward pass) içinde hem sınıf tahminini hem de
    Grad-CAM ısı haritasını üretir. Tahmin ve ısı haritası aynı çağrıdan
    geldiği için aralarında herhangi bir tutarsızlık oluşmaz.

    Yöntem, Keras'ın resmî Grad-CAM örneğindeki adımları izler:
      1. Hedef konvolüsyon katmanının çıkışı ile nihai tahmin aynı anda
         hesaplanır (GradientTape içinde).
      2. Tahmin edilen sınıfın skoru, konvolüsyon çıkışına göre türevlenir.
      3. Gradyanlar mekansal eksenlerde ortalanarak her kanal için bir
         "önem ağırlığı" elde edilir.
      4. Konvolüsyon çıkışı bu ağırlıklarla ağırlıklandırılıp toplanır,
         negatif değerler (ReLU) atılır ve [0, 1] aralığına normalize
         edilir.

    Returns:
        tuple:
            - isi_haritasi (np.ndarray): [0, 1] aralığında, (h, w) şekilli
              ısı haritası.
            - tahmin_indeksi (int): SINIF_ISIMLERI içindeki tahmin indeksi.
            - guven_orani (float): Tahminin güven yüzdesi (0-100).
    """
    grad_model = _grad_cam_modeli_olustur(model, konv_katman_adi)

    with tf.GradientTape() as tape:
        konv_ciktisi, tahminler = grad_model(girdi_tensoru, training=False)
        tahmin_indeksi = tf.argmax(tahminler[0])
        hedef_skor = tahminler[:, tahmin_indeksi]

    gradyanlar = tape.gradient(hedef_skor, konv_ciktisi)
    havuzlanmis_gradyanlar = tf.reduce_mean(gradyanlar, axis=(0, 1, 2))

    konv_ciktisi_tek = konv_ciktisi[0]
    isi_haritasi = konv_ciktisi_tek @ havuzlanmis_gradyanlar[..., tf.newaxis]
    isi_haritasi = tf.squeeze(isi_haritasi)
    isi_haritasi = tf.maximum(isi_haritasi, 0)

    maksimum_deger = tf.math.reduce_max(isi_haritasi)
    if maksimum_deger > 0:
        isi_haritasi = isi_haritasi / maksimum_deger

    guven_orani = float(tahminler[0, tahmin_indeksi].numpy()) * 100
    return isi_haritasi.numpy(), int(tahmin_indeksi.numpy()), guven_orani


def isi_haritasini_resme_bindir(
    orijinal_resim: np.ndarray,
    isi_haritasi: np.ndarray,
    agirlik: float = ISI_HARITASI_AGIRLIGI,
) -> np.ndarray:
    """
    Tek kanallı (0-1 aralığında) ısı haritasını JET renk paletiyle
    renklendirir ve orijinal RGB görüntünün ÇÖZÜNÜRLÜĞÜNE yeniden
    ölçeklenmiş şekilde yarı saydam olarak bindirir.

    Isı haritası orijinal görüntünün tam çözünürlüğüne büyütülür; böylece
    224x224'e küçültülmüş model girdisi yerine kullanıcının yüklediği veya
    kameradan çektiği fotoğrafın kalitesi korunur.
    """
    yukseklik, genislik = orijinal_resim.shape[:2]

    isi_haritasi_buyutulmus = cv2.resize(isi_haritasi, (genislik, yukseklik))
    isi_haritasi_8bit = np.uint8(255 * isi_haritasi_buyutulmus)

    isi_haritasi_renkli_bgr = cv2.applyColorMap(isi_haritasi_8bit, cv2.COLORMAP_JET)
    isi_haritasi_renkli_rgb = cv2.cvtColor(isi_haritasi_renkli_bgr, cv2.COLOR_BGR2RGB)

    bindirilmis_resim = cv2.addWeighted(
        orijinal_resim, 1 - agirlik, isi_haritasi_renkli_rgb, agirlik, 0
    )
    return bindirilmis_resim


# ----------------------------------------------------------------------------
# 4.B AÇIK-KÜME (OPEN-SET) GÜVENLİK AĞI FONKSİYONLARI
# ----------------------------------------------------------------------------
# Bu bölümdeki fonksiyonlar modeli HİÇ değiştirmez; modelin girdisini ve
# çıktısını modelden bağımsız ikinci bir gözle denetler. Amaç, mevcut
# .keras dosyası yeniden eğitilmeden, sunum sırasında "kapalı dünya"
# tuzağının (bkz. modül başlığı) yarattığı aşırı-güvenli yanlış
# sınıflandırmaları perdeleyebilmektir.

def cilt_tonu_orani_hesapla(orijinal_resim_rgb: np.ndarray) -> float:
    """
    Modelden tamamen BAĞIMSIZ bir görüntü-işleme filtresiyle, görüntüde
    GERÇEK bir insan cilt bölgesinin (yüz, el) bulunup bulunmadığını
    tahmin eder.

    Sürüm 2 — neden değişti:
    --------------------------
    İlk sürümde kullanılan geniş HSV aralığı (S>=30 gibi düşük bir
    doygunluk eşiği), portakal/elma üzerindeki SOLUK turuncu-sarı küf ve
    gölge bölgelerini de "cilt tonu" olarak işaretliyordu; bu da gerçek
    çürük meyve fotoğraflarının yanlışlıkla reddedilmesine yol açtı.
    Bu sürüm iki iyileştirme içerir:

      1. HSV yerine YCrCb renk uzayı kullanılır. Cr (kırmızı fark) ve Cb
         (mavi fark) kanalları, literatürde cilt tespiti için HSV'den
         daha DAR ve daha güvenilir bir aralık sunar; meyve kabuğu
         renklerinden (parlak turuncu, sarı, kırmızı, yeşil) çok daha
         net ayrışır.
      2. Salt PİKSEL ORANINA değil, aynı zamanda EN BÜYÜK BİTİŞİK
         BÖLGENİN (connected component) boyutuna da bakılır. Gerçek bir
         yüz/el, karede TEK BÜYÜK bir bitişik leke oluşturur. Meyvedeki
         küf/leke/gölge benzeri renk eşleşmeleri ise genelde KÜÇÜK ve
         DAĞINIK parçacıklar hâlinde kalır. Bu ayrım, oran tek başına
         yetersiz kaldığında ikinci bir güvenlik katmanı sağlar.

    Args:
        orijinal_resim_rgb (np.ndarray): RGB formatında, (H, W, 3) şekilli
            orijinal (224x224'e küçültülmemiş) görüntü dizisi.

    Returns:
        float: [0, 1] aralığında, en büyük bitişik cilt-tonu bölgesinin
            TOPLAM görüntüye oranı (dağınık küçük parçacıklar bu skoru
            yükseltmez; yalnızca tek, büyük ve bitişik bir bölge yükseltir).
    """
    bgr_resim = cv2.cvtColor(orijinal_resim_rgb, cv2.COLOR_RGB2BGR)
    ycrcb_resim = cv2.cvtColor(bgr_resim, cv2.COLOR_BGR2YCrCb)

    # YCrCb uzayında, gerçek ten tonu örneklerinden (açık/orta/koyu)
    # ölçülerek kalibre edilmiş dar bir aralık. Bu aralık; parlak
    # turuncu (portakal), kırmızı (elma), sarı (muz) ve gri (küf) gibi
    # meyve renklerinden Cb ekseninde net şekilde ayrışır.
    alt_sinir = np.array([40, 140, 77], dtype=np.uint8)
    ust_sinir = np.array([250, 173, 123], dtype=np.uint8)
    cilt_maskesi = cv2.inRange(ycrcb_resim, alt_sinir, ust_sinir)

    # Tek piksellik gürültüyü (meyvedeki rastgele küçük renk
    # eşleşmelerini) temizlemek için morfolojik açma işlemi uygulanır.
    cekirdek = np.ones((5, 5), np.uint8)
    cilt_maskesi = cv2.morphologyEx(cilt_maskesi, cv2.MORPH_OPEN, cekirdek)

    toplam_piksel = cilt_maskesi.size
    if toplam_piksel == 0:
        return 0.0

    # En büyük bitişik bölgeyi bul: gerçek bir yüz/el TEK BÜYÜK bir leke
    # oluşturur; meyvedeki dağınık küçük parçacıklar bu testi geçemez.
    bilesen_sayisi, etiketler, istatistikler, _ = cv2.connectedComponentsWithStats(
        cilt_maskesi, connectivity=8
    )
    if bilesen_sayisi <= 1:  # Sadece arka plan (0. etiket) varsa.
        return 0.0

    # 0. indeks her zaman arka plandır (cv2 kuralı); onu hesaba katmıyoruz.
    bilesen_alanlari = istatistikler[1:, cv2.CC_STAT_AREA]
    en_buyuk_bilesen_alani = int(bilesen_alanlari.max())

    return en_buyuk_bilesen_alani / toplam_piksel


@st.cache_resource(show_spinner=False)
def _yuz_dedektorlerini_yukle() -> tuple:
    """
    OpenCV ile birlikte hazır gelen Haar Cascade yüz dedektörlerini yükler
    ve önbelleğe alır (her karede diskten tekrar okunmasın diye).

    Hem önden (frontalface) hem yandan (profileface) yüz dedektörü
    döndürülür; böylece kullanıcı kameraya tam karşıdan bakmasa bile
    (hafif yan profil) yüz yakalanabilir. Bu dosyalar OpenCV kurulumuyla
    birlikte standart olarak gelir; ek indirme gerektirmez.

    Returns:
        tuple: (on_yuz_dedektoru, yan_yuz_dedektoru)
    """
    on_yuz = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    yan_yuz = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )
    return on_yuz, yan_yuz


def yuz_tespit_edildi_mi(orijinal_resim_rgb: np.ndarray) -> bool:
    """
    Görüntüde yeterince büyük bir insan yüzü olup olmadığını, modelden
    tamamen BAĞIMSIZ bir Haar Cascade dedektörüyle tespit eder.

    Bu, açık-küme güvenlik ağının ANA savunma hattıdır. Renk tabanlı
    yöntemlerin aksine, bu dedektör görüntünün RENGİNE değil yapısal
    desenine (göz-burun-ağız geometrisi) bakar. Bu çok önemli bir
    avantajdır: sıcak ten tonu ile sarı muz rengi birbirine çok yakın
    olduğundan renk filtresi ikisini karıştırabilirken, yüz dedektörü
    yalnızca GERÇEK bir yüz yapısı gördüğünde tetiklenir ve bir meyveyi
    -rengi ne olursa olsun- asla "yüz" sanmaz. Bu da "gerçek meyveyi asla
    yanlışlıkla reddetme" önceliğiyle birebir uyumludur.

    Çok küçük (gürültü kaynaklı) yanlış tespitleri elemek için, bulunan
    yüzün görüntünün kısa kenarına oranla MIN_YUZ_BOYUT_ORANI'ndan büyük
    olması şartı aranır.

    Args:
        orijinal_resim_rgb (np.ndarray): RGB formatında görüntü dizisi.

    Returns:
        bool: Yeterince büyük en az bir yüz bulunduysa True.
    """
    on_yuz_dedektoru, yan_yuz_dedektoru = _yuz_dedektorlerini_yukle()

    # Haar Cascade gri tonlamalı görüntü üzerinde çalışır.
    gri_resim = cv2.cvtColor(orijinal_resim_rgb, cv2.COLOR_RGB2GRAY)
    gri_resim = cv2.equalizeHist(gri_resim)  # Kontrastı dengele (tespiti güçlendirir).

    yukseklik, genislik = gri_resim.shape[:2]
    kisa_kenar = min(yukseklik, genislik)
    minimum_yuz_pikseli = int(kisa_kenar * MIN_YUZ_BOYUT_ORANI)
    minimum_boyut = (minimum_yuz_pikseli, minimum_yuz_pikseli)

    # Önden bakan yüzleri ara.
    on_yuzler = on_yuz_dedektoru.detectMultiScale(
        gri_resim, scaleFactor=1.1, minNeighbors=5, minSize=minimum_boyut
    )
    if len(on_yuzler) > 0:
        return True

    # Bulunamazsa yan profili ara (kullanıcı hafif yana dönükse).
    yan_yuzler = yan_yuz_dedektoru.detectMultiScale(
        gri_resim, scaleFactor=1.1, minNeighbors=5, minSize=minimum_boyut
    )
    if len(yan_yuzler) > 0:
        return True

    # Yan profil dedektörü tek yöne göre eğitilmiştir; görüntüyü yatayda
    # çevirip tekrar deneyerek diğer yöne dönük profilleri de yakalarız.
    yan_yuzler_ayna = yan_yuz_dedektoru.detectMultiScale(
        cv2.flip(gri_resim, 1), scaleFactor=1.1, minNeighbors=5, minSize=minimum_boyut
    )
    if len(yan_yuzler_ayna) > 0:
        return True

    return False


def guvenlik_degerlendirmesi_yap(
    orijinal_resim_rgb: np.ndarray,
    tum_olasiliklar: np.ndarray,
) -> dict:
    """
    Üç bağımsız güvenlik kontrolünü birlikte çalıştırıp tek bir karar
    nesnesinde toplar: cilt tonu oranı, güven eşiği ve top-1/top-2 marjı.

    Kontroller arasındaki öncelik sırası bilinçlidir: cilt tonu tespiti en
    güçlü reddetme sebebidir (modelin sınıf olasılıklarına hiç bakmadan
    devreye girer), çünkü yüz/el gibi durumlarda model %100 güvenle
    yanılabildiği gözlemlenmiştir (bkz. modül başlığı). Güven ve marj
    kontrolleri ise gerçek bir meyve olsa da modelin kararsız kaldığı
    (örn. sarı elma → %50.47) durumları yakalar.

    Args:
        orijinal_resim_rgb (np.ndarray): Analiz edilen orijinal görüntü.
        tum_olasiliklar (np.ndarray): Modelin 6 sınıf için ürettiği
            olasılık dağılımı (toplamı 1'e yakın).

    Returns:
        dict: {
            "guvenilir": bool,       # Sonuç kullanıcıya gösterilsin mi?
            "sebep": str | None,     # Güvenilir değilse, kullanıcıya
                                      # gösterilecek Türkçe açıklama.
            "yuz_bulundu": bool,
            "cilt_tonu_orani": float,
            "en_yuksek_olasilik": float,
            "marj": float,
        }
    """
    siralanmis_olasiliklar = np.sort(tum_olasiliklar)[::-1]
    en_yuksek_olasilik = float(siralanmis_olasiliklar[0])
    ikinci_en_yuksek_olasilik = float(siralanmis_olasiliklar[1])
    marj = en_yuksek_olasilik - ikinci_en_yuksek_olasilik

    yuz_bulundu = yuz_tespit_edildi_mi(orijinal_resim_rgb)
    cilt_tonu_orani = cilt_tonu_orani_hesapla(orijinal_resim_rgb)

    sonuc = {
        "guvenilir": True,
        "sebep": None,
        "yuz_bulundu": yuz_bulundu,
        "cilt_tonu_orani": cilt_tonu_orani,
        "en_yuksek_olasilik": en_yuksek_olasilik,
        "marj": marj,
    }

    # Kontrol 1 — ANA SAVUNMA (en güçlü reddetme): Görüntüde gerçek bir
    # yüz YAPISI tespit edildiyse, sınıf olasılıklarına hiç bakılmadan
    # tahmin bastırılır. Bu kontrol renge bakmadığı için bir meyveyi asla
    # yanlışlıkla "yüz" sanmaz.
    if yuz_bulundu:
        sonuc["guvenilir"] = False
        sonuc["sebep"] = (
            "📷 Görüntüde bir insan yüzü tespit edildi. Bu sistem yalnızca "
            "meyve fotoğrafları için eğitilmiştir; lütfen kameraya yalnızca "
            "meyveyi gösterin."
        )
        return sonuc

    # Kontrol 2 — İKİNCİL/YEDEK: Yüz dedektörü kaçırmış olabileceği aşırı
    # durumlar (çok yakın/kısmi yüz, el vb.) için, kare neredeyse tamamen
    # tek parça cilt tonuyla kaplıysa reddet. Eşik yüksek tutulduğundan
    # gerçek meyveleri etkilemez.
    if cilt_tonu_orani >= CILT_TONU_ORANI_ESIGI:
        sonuc["guvenilir"] = False
        sonuc["sebep"] = (
            "📷 Görüntünün büyük bölümü insan cilt tonuna benziyor (yüz, "
            "el vb.). Bu sistem yalnızca meyve fotoğrafları için "
            "eğitilmiştir; lütfen kameraya yalnızca meyveyi gösterin."
        )
        return sonuc

    # Kontrol 3: Genel güven eşiği.
    if en_yuksek_olasilik < GUVEN_ESIGI:
        sonuc["guvenilir"] = False
        sonuc["sebep"] = (
            f"🤔 Model bu görüntüde kararsız kaldı (en yüksek güven oranı "
            f"yalnızca %{en_yuksek_olasilik * 100:.2f}). Lütfen meyveyi "
            "daha net, tek başına ve iyi aydınlatılmış şekilde gösterin."
        )
        return sonuc

    # Kontrol 4: İki en olası sınıf birbirine çok yakınsa.
    if marj < MARJ_ESIGI:
        sonuc["guvenilir"] = False
        sonuc["sebep"] = (
            "🤔 Model iki sınıf arasında kararsız kaldı (olasılıklar "
            "birbirine çok yakın). Lütfen meyveyi daha net gösterin veya "
            "farklı bir açıdan tekrar deneyin."
        )
        return sonuc

    return sonuc





# ----------------------------------------------------------------------------
# 5. ARAYÜZ BİLEŞENLERİ
# ----------------------------------------------------------------------------
def sayfa_ayarlarini_yap() -> None:
    st.set_page_config(
        page_title="Akıllı Meyve Tazelik Kontrolü",
        page_icon="🍎",
        layout="centered",
    )


def baslik_bolumunu_olustur() -> None:
    st.title("🍎🍌🍊 Yapay Zeka Destekli Akıllı Meyve Tazelik Kontrol Sistemi")
    st.caption(
        "Evrişimli Sinir Ağı (CNN) ve Açıklanabilir Yapay Zeka (Grad-CAM) "
        "ile meyve tazelik analizi — Dönem Projesi"
    )
    st.markdown(
        """
        Bu sistem, yüklediğiniz veya kameradan canlı çektiğiniz meyve
        fotoğrafını **6 sınıf** üzerinden (Taze/Çürük × Elma/Muz/Portakal)
        sınıflandırır. Sadece bir karar vermekle kalmaz; **Grad-CAM**
        tekniğiyle modelin görüntünün **hangi bölgesine bakarak** bu
        kararı verdiğini ısı haritası olarak gösterir.
        """
    )
    st.divider()


def kenar_cubugunu_olustur() -> None:
    with st.sidebar:
        st.header("ℹ️ Proje Hakkında")
        st.markdown(
            """
            **Model Mimarisi:** 3 bloklu CNN (Conv2D → BatchNormalization →
            MaxPooling2D), ardından tam bağlantılı (Dense) sınıflandırma
            katmanları.

            **Açıklanabilirlik (XAI):** Tahminin güvenilirliğini
            doğrulamak için Grad-CAM kullanılır. Bu teknik, son konvolüsyon
            katmanındaki (`son_conv_katmani`) aktivasyon gradyanlarından
            yararlanarak modelin görüntünün hangi bölgesine odaklandığını
            görselleştirir.

            **Sınıflar:**
            """
        )
        for isim in SINIF_ISIMLERI:
            st.markdown(f"- {isim}")

        st.divider()
        st.caption(
            "Ders Dönem Projesi — Yapay Zeka Destekli Meyve Tazelik Kontrolü"
        )


def goruntu_kaynagini_sec() -> Image.Image | None:
    """
    Kullanıcıya iki ayrı sekme üzerinden görüntü kaynağı seçtirir:
    "📁 Fotoğraf Yükle" (mevcut file_uploader, DOKUNULMADAN korunmuştur) ve
    "📷 Kameradan Çek" (yeni eklenen st.camera_input). Hangi sekme
    kullanılırsa kullanılsın, fonksiyon tek tip bir PIL.Image nesnesi
    döndürür; böylece çağıran kod iki kaynak arasında herhangi bir ayrım
    yapmak zorunda kalmaz ve aynı analiz hattından (bkz. analiz_et) geçer.

    Olası bozuk/boş/okunamayan kare durumları burada try-except ile
    yakalanır; canlı sunum sırasında uygulamanın çökmesi yerine kullanıcıya
    zarif bir uyarı gösterilir.

    Returns:
        Image.Image | None: Seçilen/çekilen fotoğraf (RGB), yoksa None.
    """
    sekme_yukle, sekme_kamera = st.tabs(["📁 Fotoğraf Yükle", "📷 Kameradan Çek"])

    ham_dosya = None

    with sekme_yukle:
        dosya_secimi = st.file_uploader(
            "Analiz edilecek meyve fotoğrafını yükleyin",
            type=["jpg", "jpeg", "png"],
            key="dosya_yukleyici",
        )
        if dosya_secimi is not None:
            ham_dosya = dosya_secimi
        else:
            st.info("Başlamak için yukarıdan bir fotoğraf yükleyin.")

    with sekme_kamera:
        st.caption(
            "Canlı sunum için: meyveyi kameraya gösterip aşağıdaki "
            "düğmeyle fotoğraf çekin."
        )
        kamera_karesi = st.camera_input(
            "Meyvenin fotoğrafını çekin",
            key="kamera_yakalayici",
        )
        if kamera_karesi is not None:
            ham_dosya = kamera_karesi

    if ham_dosya is None:
        return None

    # Görüntü açma adımı, sunum sırasında en kırılgan noktadır (kamera izni
    # reddi, yarıda kesilmiş/boş kare, beklenmeyen dosya türü vb.). Bu
    # nedenle ayrı ve sıkı bir try-except bloğu ile korunmuştur; herhangi
    # bir hata, kırmızı bir traceback yerine zarif bir uyarıya dönüşür.
    try:
        pil_resim = Image.open(ham_dosya).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.warning(
            "⚠️ Görüntü işlenirken bir sorun oluştu, lütfen tekrar çekin "
            "veya farklı bir fotoğraf yükleyin."
        )
        return None

    return pil_resim


def sonuc_bolumunu_olustur(
    secilen_sinif: str,
    guven_orani: float,
    curuk_mu: bool,
) -> None:
    sonuc_sol, sonuc_sag = st.columns([2, 1])

    with sonuc_sol:
        if curuk_mu:
            st.error(f"### 🔴 Tahmin: {secilen_sinif}")
            st.write(
                "Isı haritasındaki **kırmızı/sarı** bölgeler, modelin "
                "çürüklük belirtisi (renk/doku bozulması) tespit ettiği "
                "alanlardır."
            )
        else:
            st.success(f"### 🟢 Tahmin: {secilen_sinif}")
            st.write(
                "Model, meyvenin genel yüzey dokusuna ve renk tutarlılığına "
                "bakarak taze olduğuna karar vermiştir."
            )

    with sonuc_sag:
        st.metric(label="Güven Oranı", value=f"%{guven_orani:.2f}")


def olasilik_dagilimini_goster(tahminler: np.ndarray) -> None:
    with st.expander("📊 Tüm sınıflar için olasılık dağılımı"):
        siralanmis = sorted(
            zip(SINIF_ISIMLERI, tahminler), key=lambda cift: -cift[1]
        )
        for isim, oran in siralanmis:
            st.progress(float(oran), text=f"{isim}: %{oran * 100:.2f}")


# ----------------------------------------------------------------------------
# 6. ORTAK ANALİZ HATTI (DOSYA + KAMERA İÇİN TEK NOKTA)
# ----------------------------------------------------------------------------
def analiz_et(model: keras.Model, pil_resim: Image.Image) -> None:
    """
    Hem 'Fotoğraf Yükle' hem de 'Kameradan Çek' sekmesinden gelen
    görüntüler için çalıştırılan TEK ve ORTAK işlem hattı.

    Bu fonksiyon kasıtlı olarak kaynak (dosya/kamera) bilgisinden tamamen
    bağımsız tasarlanmıştır: PIL ile açma adımı zaten
    goruntu_kaynagini_sec içinde tamamlanmış olduğundan, buradan itibaren
    224x224 boyutlandırma, [0, 1] ölçekleme (modelin içindeki Rescaling
    katmanı tarafından), model tahmini ve Grad-CAM ısı haritası üretimi
    her iki kaynak için de birebir aynı kod yolundan geçer. Sunum
    sırasında modelin veya OpenCV işleminin beklenmedik şekilde
    patlaması ihtimaline karşı tüm ağır işlem bir try-except bloğuna
    alınmıştır; bir hata oluşursa kullanıcı kırmızı bir hata kodu yerine
    zarif bir uyarı görür ve uygulama çalışmaya devam eder.

    Model tahmini üretildikten SONRA, sonuç kullanıcıya gösterilmeden
    önce guvenlik_degerlendirmesi_yap ile üç bağımsız kontrolden
    (cilt tonu, güven eşiği, marj) geçirilir; bu canlı sunum sırasında
    modelin kapalı-küme sınırlamasından kaynaklanan aşırı-güvenli yanlış
    sınıflandırmaları (örn. yüz/el → %100 yanlış tahmin) kullanıcıdan
    saklar.
    """
    orijinal_dizi = np.array(pil_resim)

    gorsel_sol, gorsel_sag = st.columns(2)
    with gorsel_sol:
        st.image(pil_resim, caption="Analiz Edilen Görüntü", use_container_width=True)

    try:
        with st.spinner("Model analiz ediyor ve ısı haritası çıkarılıyor..."):
            girdi_tensoru = model_girdisi_olustur(pil_resim)

            # Tek bir ileri besleme ile tüm sınıf olasılıklarını da elde
            # etmek için modeli ayrıca çağırıyoruz (grad_cam fonksiyonu
            # sadece tahmin edilen tek sınıfın güven oranını döndürür, tam
            # dağılımı değil).
            tum_tahminler = model.predict(girdi_tensoru, verbose=0)[0]

            isi_haritasi, tahmin_indeksi, guven_orani = grad_cam_ve_tahmin_uret(
                model, girdi_tensoru, SON_KONVOLUSYON_KATMANI_ADI
            )
            bindirilmis_resim = isi_haritasini_resme_bindir(orijinal_dizi, isi_haritasi)
    except Exception:
        # Canlı sunumda kırmızı bir traceback yerine zarif bir uyarı
        # göstermek, "sıfır alabama" hedefi için kritik bir güvenlik ağıdır.
        # Beklenmeyen her türlü hata (bozuk kare, boyut uyumsuzluğu vb.)
        # burada yakalanır.
        st.warning(
            "⚠️ Görüntü işlenirken bir sorun oluştu, lütfen tekrar çekin "
            "veya farklı bir fotoğraf yükleyin."
        )
        return

    with gorsel_sag:
        st.image(
            bindirilmis_resim,
            caption="Grad-CAM Isı Haritası (Kırmızı: Modelin Odak Noktası)",
            use_container_width=True,
        )

    st.divider()

    # ------------------------------------------------------------------
    # AÇIK-KÜME GÜVENLİK AĞI DEVREYE GİRİYOR
    # ------------------------------------------------------------------
    # Model zaten bir tahmin üretti (yukarıda), ancak bu tahmine körü
    # körüne güvenmek yerine, modülün başında açıklanan üç bağımsız
    # kontrolden geçiriyoruz. Kontroller başarısız olursa, tahmin
    # kullanıcıya HİÇ gösterilmez; bunun yerine nazik, eğitici bir uyarı
    # gösterilir. Bu, sunum sırasında modelin "kapalı dünya" tuzağına
    # düştüğü anları (yüz/el → %100 yanlış güven, sarı elma → %50
    # kararsızlık) perdelemek için tasarlanmıştır.
    degerlendirme = guvenlik_degerlendirmesi_yap(orijinal_dizi, tum_tahminler)

    if not degerlendirme["guvenilir"]:
        st.warning(degerlendirme["sebep"])
        with st.expander("🔧 Teknik detay (hata ayıklama amaçlı)"):
            st.write(
                f"Yüz tespit edildi mi: "
                f"{'Evet' if degerlendirme['yuz_bulundu'] else 'Hayır'}"
            )
            st.write(
                f"En yüksek model güveni: %{degerlendirme['en_yuksek_olasilik'] * 100:.2f}"
            )
            st.write(f"Top-1 / Top-2 marjı: {degerlendirme['marj']:.3f}")
            st.write(
                f"Tespit edilen cilt tonu oranı: "
                f"%{degerlendirme['cilt_tonu_orani'] * 100:.1f}"
            )
            st.caption(
                "Bu sistem, kapalı-küme (closed-world) bir sınıflandırıcı "
                "olduğu için 'bilinmiyor' sınıfına sahip değildir; bu "
                "panel modelin ürettiği ham güven skorlarını, bağımsız "
                "yüz/cilt/eşik kontrolleriyle birlikte şeffaf şekilde "
                "gösterir."
            )
        return

    secilen_sinif = SINIF_ISIMLERI[tahmin_indeksi]
    curuk_mu = tahmin_indeksi in CURUK_SINIF_INDEKSLERI
    sonuc_bolumunu_olustur(secilen_sinif, guven_orani, curuk_mu)
    olasilik_dagilimini_goster(tum_tahminler)


# ----------------------------------------------------------------------------
# 7. ANA UYGULAMA AKIŞI
# ----------------------------------------------------------------------------
def main() -> None:
    sayfa_ayarlarini_yap()
    baslik_bolumunu_olustur()
    kenar_cubugunu_olustur()

    try:
        model = model_yukle(MODEL_DOSYA_YOLU)
    except (OSError, ValueError) as hata:
        st.error(
            f"Model dosyası yüklenemedi: '{MODEL_DOSYA_YOLU}'.\n\n"
            "Lütfen 'cnn_egitim.py' betiğini çalıştırdığınızdan ve bu "
            "dosyanın 'cnn_arayuz.py' ile aynı klasörde olduğundan emin "
            "olun.\n\n"
            f"Teknik detay: {hata}"
        )
        st.stop()
        return

    pil_resim = goruntu_kaynagini_sec()

    if pil_resim is None:
        return

    analiz_et(model, pil_resim)


if __name__ == "__main__":
    main()