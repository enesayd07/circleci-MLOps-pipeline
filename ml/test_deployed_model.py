import os
import sys
import time
import requests
import numpy as np
from PIL import Image
from io import BytesIO

class LiveAPITester:
    API_URL: str = os.getenv("API_URL", "https://circleci-mlops-pipeline.onrender.com")
    MAX_RETRIES: int = 20  
    WAIT_TIME: int = 15  
    def __init__(self) -> None:
        # flush=True sayesinde yazılar terminale anında düşer, donmuş gibi görünmez
        print(f"[TEST BAŞLIYOR] Hedef API: {self.API_URL}", flush=True)

    # API'nin gerçekten uyanık olup olmadığını kontrol eder
    def check_health(self) -> bool:
        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.get(f"{self.API_URL}/", timeout=15)
                if response.status_code == 200:
                    print(f"[{attempt+1}/{self.MAX_RETRIES}] Sağlık Kontrolü Başarılı! Sunucu Ayakta.", flush=True)
                    return True
            except requests.exceptions.RequestException as e:
                print(f"[{attempt+1}/{self.MAX_RETRIES}] Sunucu henüz uyanmadı. {self.WAIT_TIME} saniye bekleniyor... (Hata: {type(e).__name__})", flush=True)
            
            time.sleep(self.WAIT_TIME)
        
        raise Exception("KRİTİK HATA: Dağıtım yapıldı ancak API sunucusu (Render) yanıt vermiyor!")

    # API'ye sahte bir resim gönderip tahmin döndürüp döndürmediğini test eder
    def test_prediction_endpoint(self) -> None:
        print("\n[TEST] Tahmin Uç Noktası (/predict) test ediliyor...", flush=True)
        
        dummy_image_array = np.zeros((28, 28), dtype=np.uint8)
        image = Image.fromarray(dummy_image_array, mode='L')
        
        image_bytes = BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        
        files = {'file': ('test_image.png', image_bytes, 'image/png')}
        
        # API'ye gönder (POST isteği)
        try:
            response = requests.post(f"{self.API_URL}/predict", files=files, timeout=30)
        except Exception as e:
            raise Exception(f"Tahmin isteği gönderilirken ağ hatası oluştu: {str(e)}")

        # Gelen cevabı doğrula
        if response.status_code != 200:
            raise Exception(f"API Hata Döndürdü! Durum Kodu: {response.status_code}, Detay: {response.text}")
        
        result = response.json()
        
        # Beklenen JSON formatının dönüp dönmediğini kontrol et 
        if "tahmin" not in result or "guven_skoru_yuzde" not in result:
            raise Exception(f"API Beklenmeyen JSON Formatı Döndürdü! Cevap: {result}")
        
        print("MÜKEMMEL! Uçtan Uca (E2E) Dağıtım Testi başarıyla tamamlandı.", flush=True)
        print(f"API'den Gelen Örnek Yanıt: {result}", flush=True)

    # Testleri sırasıyla çalıştırır
    def run_tests(self) -> None:
        self.check_health()
        self.test_prediction_endpoint()

if __name__ == "__main__":
    tester = LiveAPITester()
    tester.run_tests()