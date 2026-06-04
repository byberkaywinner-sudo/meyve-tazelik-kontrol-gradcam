from ultralytics import YOLO

# 1. Adım: YOLO'nun sınıflandırma için eğitilmiş en küçük ve hızlı modelini (beynini) çağırıyoruz.
model = YOLO('yolov8n-cls.pt')  

# 2. Adım: Eğitimi başlatıyoruz. 
# data='dataset' diyerek fotoğrafların nerede olduğunu gösteriyoruz.
# epochs=10 diyerek yapay zekanın bu fotoğraflara 10 tur boyunca baştan sona çalışmasını istiyoruz.
sonuclar = model.train(data='dataset/dataset', epochs=10, imgsz=224)