import os
import sys
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image

# Uygulamamızı başlatıyoruz
app = FastAPI(
    title="Fashion MNIST Yapay Zeka API",
    description="Bu API, eğitilmiş CNN modelini kullanarak kıyafet resimlerini sınıflandırır.",
    version="1.0.0"
)

# Modelin klasör yolu
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "training_data", "trained_model")

# Model değişkenimizi başta boş (None) bırakıyoruz, uygulama başlayınca yükleyeceğiz
model = None

# Sınıf isimleri (Fashion MNIST etiketleri)
CLASS_NAMES = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

@app.on_event("startup")
async def load_model():
    """
    API sunucusu (Render) ayağa kalktığında ilk olarak bu fonksiyon çalışır.
    Modeli diskten okur ve belleğe (RAM) yükler. Böylece her istekte tekrar tekrar
    model yüklenmez, cevaplar milisaniyeler içinde verilir.
    """
    global model
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Model yükleniyor: {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model başarıyla yüklendi!")
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {str(e)}")
    else:
        print("UYARI: Eğitilmiş model dosyası bulunamadı! Tahmin yapılamayacak.")


@app.get("/")
async def root():
    """
    API'nin ana sayfasına girildiğinde çalışan basit bir sağlık kontrolü (Health Check).
    """
    status = "Çalışıyor" if model is not None else "Model Bekleniyor"
    return {"mesaj": "Fashion MNIST Yapay Zeka Fabrikasına Hoş Geldiniz", "model_durumu": status}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Kullanıcının gönderdiği resmi alıp tahmin yapan ana Endpoint.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi veya bulunamadı.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir resim dosyası gönderin.")

    try:
        # 1. Kullanıcının gönderdiği dosyayı oku
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('L') # Resmi Gri Tonlamaya (Siyah Beyaz) çevir
        
        # 2. Resmi modelimizin beklediği boyuta (28x28) getir
        image = image.resize((28, 28))
        
        # 3. Resmi NumPy matrisine çevir ve pikselleri 0-1 arasına sıkıştır (Build adımındaki gibi)
        image_array = np.array(image)
        image_array = image_array.astype('float32') / 255.0
        
        # 4. Modelin beklediği (1, 28, 28, 1) boyutuna getir (Batch formatı)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = np.expand_dims(image_array, axis=-1)
        
        # 5. Tahmini yap!
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) # Tahmin kesinlik yüzdesi
        
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        
        # 6. Sonucu müşteriye JSON olarak döndür
        return JSONResponse(content={
            "tahmin": predicted_class_name,
            "sinif_kodu": int(predicted_class_index),
            "guven_skoru_yuzde": round(confidence * 100, 2)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resim işlenirken hata oluştu: {str(e)}")