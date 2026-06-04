import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. Eğittiğimiz o "en iyi" beyni dosya yolundan çağırıyoruz. 
# Ekran görüntündeki best.pt dosyasının yolunu buraya yazdık.
model = YOLO('runs/classify/train/weights/best.pt')

# 2. Web sitesinin başlığı ve açıklaması
st.title("🍎 Akıllı Meyve Tazelik Kontrol Sistemi")
st.write("Bu sistem, yüklediğiniz meyvenin taze mi yoksa çürük mü olduğunu anlar.")

# 3. Kullanıcıdan bir fotoğraf yüklemesini istiyoruz
yuklenen_resim = st.file_uploader("Test etmek için bir meyve fotoğrafı yükleyin", type=["jpg", "png", "jpeg"])

if yuklenen_resim is not None:
    # Fotoğraf yüklendiyse bunu ekranda göster
    resim = Image.open(yuklenen_resim)
    st.image(resim, caption='Yüklediğiniz Meyve', use_column_width=True)
    
    st.info("Yapay zeka analiz ediyor, lütfen bekleyin...")
    
    # 4. Resmi bizim eğittiğimiz modele yolluyoruz (Hocanın "Çıkarım/Inference" dediği şart bu)
    sonuc = model(resim)
    
    # 5. Modelden gelen sonucu Türkçeye çevirip ekrana yazdırıyoruz
    tahmin_edilen_sinif = sonuc[0].names[sonuc[0].probs.top1]
    guven_orani = sonuc[0].probs.top1conf.item() * 100
    
    st.success(f"🎉 Sonuç: Bu meyve **{tahmin_edilen_sinif}**! (Eminlik Oranı: %{guven_orani:.2f})")