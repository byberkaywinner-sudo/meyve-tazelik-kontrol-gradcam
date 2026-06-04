import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# 1. Eğittiğimiz 1.5 saatlik emeği (Beyni) yüklüyoruz
@st.cache_resource
def model_yukle():
    return tf.keras.models.load_model('meyve_cnn_model.h5')

model = model_yukle()

# Web sitesi başlıkları
st.title("🍎 Isı Haritalı Akıllı Meyve Kontrol Sistemi")
st.write("Hocanın istediği o özel proje: Çürük meyveleri sadece tespit etmekle kalmaz, **neresinin çürük olduğunu ısı haritasıyla (kırmızı renk) gösterir!**")

# Keras 3'ün unutkanlık hatasını çözmek için röntgen cihazını (grafiği) koda baştan kurduruyoruz
def isisi_haritasi_cikar(img_array, model, son_katman_adi):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = inputs
    son_katman_ciktisi = None
    
    # Modelin içindeki katmanları tek tek birbirine manuel bağlıyoruz
    for layer in model.layers:
        x = layer(x)
        if layer.name == son_katman_adi:
            son_katman_ciktisi = x
            
    # Yeni ve sorunsuz Grad-CAM modelimiz
    grad_model = tf.keras.Model(inputs=inputs, outputs=[son_katman_ciktisi, x])
    
    with tf.GradientTape() as tape:
        son_katman_ciktisi_tape, tahminler = grad_model(img_array)
        hedef_sinif = tf.argmax(tahminler[0])
        sinif_kanali = tahminler[:, hedef_sinif]

    gradlar = tape.gradient(sinif_kanali, son_katman_ciktisi_tape)
    havuzlanmis_gradlar = tf.reduce_mean(gradlar, axis=(0, 1, 2))
    son_katman_ciktisi_tape = son_katman_ciktisi_tape[0]
    
    isi_haritasi = son_katman_ciktisi_tape @ havuzlanmis_gradlar[..., tf.newaxis]
    isi_haritasi = tf.squeeze(isi_haritasi)
    isi_haritasi = tf.maximum(isi_haritasi, 0) / tf.math.reduce_max(isi_haritasi)
    return isi_haritasi.numpy()

# Kullanıcıdan fotoğraf alıyoruz
yuklenen_resim = st.file_uploader("Test etmek için bir meyve fotoğrafı yükleyin", type=["jpg", "png", "jpeg"])

if yuklenen_resim is not None:
    resim = Image.open(yuklenen_resim).convert('RGB')
    st.image(resim, caption='Yüklediğiniz Meyve', use_container_width=True)
    
    st.info("Derin Öğrenme (CNN) modeli analiz ediyor ve röntgen çekiyor...")
    
    # Resmi modelin anlayacağı boyuta (224x224) getiriyoruz
    resim_boyutlu = resim.resize((224, 224))
    img_array = np.array(resim_boyutlu)
    img_array_genis = np.expand_dims(img_array, axis=0)
    
    # Modele tahmin yaptırıyoruz
    tahmin_sonucu = model.predict(img_array_genis)
    tahmin_index = np.argmax(tahmin_sonucu[0])
    guven_orani = tahmin_sonucu[0][tahmin_index] * 100
    
    # Sınıfları belirliyoruz (Klasörlerin alfabetik sırasına göre 6 sınıf)
    siniflar = [
        "Taze Elma 🍏", 
        "Taze Muz 🍌", 
        "Taze Portakal 🍊", 
        "Çürük Elma 🍎", 
        "Çürük Muz 🍌", 
        "Çürük Portakal 🍊"
    ]
    
    secilen_sinif = siniflar[tahmin_index]
    
    st.success(f"🎉 Sonuç: Bu meyve %{guven_orani:.2f} ihtimalle **{secilen_sinif}**!")
    
    # Şov Kısmı: Isı Haritasını Çizdirme
    st.write("### 🔍 Model Nereye Bakarak Karar Verdi? (Isı Haritası)")
    
    # Eğitimde adını 'son_conv_katmani' koyduğumuz yerden veriyi çekiyoruz
    isi_haritasi = isisi_haritasi_cikar(img_array_genis, model, 'son_conv_katmani')
    
    # Isı haritasını renklendirip orjinal resmin üstüne yapıştırıyoruz
    isi_haritasi_boyutlu = cv2.resize(isi_haritasi, (img_array.shape[1], img_array.shape[0]))
    isi_haritasi_boyutlu = np.uint8(255 * isi_haritasi_boyutlu)
    isi_haritasi_renkli = cv2.applyColorMap(isi_haritasi_boyutlu, cv2.COLORMAP_JET)
    isi_haritasi_renkli = cv2.cvtColor(isi_haritasi_renkli, cv2.COLOR_BGR2RGB) # Renkleri düzelt
    
    # Resimleri üst üste bindir (%60 orjinal, %40 renkler)
    birlestirilmis_resim = cv2.addWeighted(img_array, 0.6, isi_haritasi_renkli, 0.4, 0)
    
    st.image(birlestirilmis_resim, caption="Kırmızı alanlar modelin odaklandığı (çürük) yerleri gösterir.", use_container_width=True)