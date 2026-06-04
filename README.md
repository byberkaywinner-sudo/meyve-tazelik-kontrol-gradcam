====================================================================
YAPAY ZEKA DESTEKLİ AKILLI MEYVE TAZELİK KONTROL SİSTEMİ VE GRAD-CAM
====================================================================

Geliştirici: İbrahim Berkay Tunçdemir - Hüseyin Emre Şeker
Üniversite: Mersin Üniversitesi - Bilişim Sistemleri ve Teknolojileri

PROJE HAKKINDA
--------------
Bu proje, endüstriyel üretim bantlarından geçen meyvelerin (Elma, Muz, Portakal) taze mi yoksa çürük mü olduğunu insan müdahalesi olmadan saniyeler içinde tespit eden bir Derin Öğrenme (Deep Learning) sistemidir. 

Projenin en önemli özelliği, sadece sınıflandırma yapmakla kalmayıp Açıklanabilir Yapay Zeka (XAI) konsepti olan Grad-CAM algoritmasını kullanmasıdır. Sistem, çürük olarak tespit ettiği meyvenin tam olarak neresine bakarak bu kararı verdiğini orijinal fotoğraf üzerine çizdiği kırmızı/sarı bir Isı Haritası (Heatmap) ile kullanıcıya kanıtlar.

ÖZELLİKLER
----------
* 6 Farklı Sınıf Tespiti: Taze Elma, Taze Muz, Taze Portakal, Çürük Elma, Çürük Muz, Çürük Portakal.
* Özelleştirilmiş CNN Mimarisi: Özellik çıkarımı (feature extraction) için özel olarak eğitilmiş, ezberlemeyi önleyen Dropout katmanlarına sahip 20 Epoch'luk model.
* Grad-CAM Entegrasyonu: Modelin karar mekanizmasının "röntgenini" çeken ısı haritası sistemi.
* Web Arayüzü: Streamlit kullanılarak geliştirilmiş, herkesin kolayca fotoğraf yükleyip test edebileceği kullanıcı dostu dashboard.

KULLANILAN TEKNOLOJİLER
-----------------------
* Python 3.10
* TensorFlow & Keras 3 (Derin Öğrenme Modeli)
* OpenCV (Görüntü İşleme ve Isı Haritası Giydirme)
* Streamlit (Web Arayüzü)
* NumPy & Pillow (Matris ve Resim İşlemleri)

KURULUM VE ÇALIŞTIRMA
---------------------
1. Gerekli kütüphaneleri kurmak için terminale aşağıdaki komutu yazın:
   pip install tensorflow opencv-python matplotlib tf-keras streamlit pillow numpy

2. Proje dizininde temiz bir terminal açın ve arayüzü başlatmak için şu komutu çalıştırın:
   python -m streamlit run cnn_arayuz.py
   (Not: Python 3.10 kullandığınızdan emin olun)

3. Tarayıcınızda açılan (genellikle http://localhost:8501) ekrandan bir meyve fotoğrafı yükleyerek sistemi test edebilirsiniz.

EK BİLGİ
--------
Projenin veri seti (dataset) boyutu çok büyük olduğu için bu GitHub deposuna dahil edilmemiştir. Proje klasöründeki kodlar, önceden eğittiğimiz 'meyve_cnn_model.h5' ağırlık dosyası üzerinden doğrudan çıkarım (inference) yapacak şekilde yapılandırılmıştır.
