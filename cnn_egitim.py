import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Veri setini yüklüyoruz (Klasörlerden otomatik alıyor)
# Resimleri 224x224 boyutuna getiriyoruz ki model standart çalışsın
train_dataset = tf.keras.utils.image_dataset_from_directory(
    'dataset/dataset/train',
    image_size=(224, 224),
    batch_size=32
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    'dataset/dataset/val',
    image_size=(224, 224),
    batch_size=32
)

# Kaç farklı meyve sınıfımız olduğunu otomatik buluyoruz
sinif_isimleri = train_dataset.class_names
sinif_sayisi = len(sinif_isimleri)
print(f"Eğitilecek Sınıflar: {sinif_isimleri}")

# 2. CNN Beynini İnşa Ediyoruz (Hocanın istediği Derin Öğrenme modeli)
model = models.Sequential([
    # Gözlerimiz (Resmi algılayan kısımlar)
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255), # Renkleri 0-1 arasına sıkıştırıp işi kolaylaştırıyoruz
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Isı haritası (Grad-CAM) için bu katmana bir isim veriyoruz!
    # Hocanın çürük yeri kırmızı göstermesini bu katman sayesinde yapacağız.
    layers.Conv2D(128, (3, 3), activation='relu', name='son_conv_katmani'),
    layers.MaxPooling2D((2, 2)),
    
    # Karar verme aşaması (Beynin karar mekanizması)
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Ezberlemeyi önlemek için (Hocanın gözüne girer)
    layers.Dense(sinif_sayisi, activation='softmax')
])

# 3. Beyne nasıl öğreneceğini söylüyoruz
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Eğitimi Başlatıyoruz (Hocanın istediği gibi Epoch sayısını yüksek tutuyoruz)
print("--------------------------------------------------")
print("🚀 Eğitim başlıyor... Bu işlem bilgisayarın hızına göre biraz sürebilir.")
history = model.fit(train_dataset, validation_data=val_dataset, epochs=20)

# 5. Eğitilen beyni bilgisayara kaydediyoruz
model.save('meyve_cnn_model.h5')
print("--------------------------------------------------")
print("✅ Beyin başarıyla eğitildi ve 'meyve_cnn_model.h5' olarak kaydedildi!")