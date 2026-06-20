import os
import sys
import time
import requests
import numpy as np
from PIL import Image
from io import BytesIO

class LiveAPITester:
    # CircleCI'dan çevresel değişken olarak Render URL'imizi alacağız, yoksa lokalde dener.
    API_URL: str = os.getenv("API_URL", "http://localhost:8000")
    MAX_RETRIES: int = 5
    WAIT_TIME: int = 5

    def __init__(self) -> None:
        print(f"[TEST BAŞLIYOR] Hedef API: {self.API_URL}")

    # API'nin gerçekten uyanık olup olmadığını kontrol eder
    def check_health(self) -> bool:
        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.get(f"{self.API_URL}/")
                if response.status_code == 200:
                    print(f"[{attempt+1}/{self.MAX_RETRIES}] Sağlık Kontrolü Başarılı! Sunucu Ayakta.")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            
            print(f"[{attempt+1}/{self.MAX_RETRIES}] Sunucuya ulaşılamadı. {self.WAIT_TIME} saniye bekleniyor...")
            time.sleep(self.WAIT_TIME)
        
        raise Exception("KRİTİK HATA: Dağıtım yapıldı ancak API sunucusu (Render) yanıt vermiyor!")

    # API'ye sahte bir resim gönderip tahmin döndürüp döndürmediğini test eder
    def test_prediction_endpoint(self) -> None:
        print("\n[TEST] Tahmin Uç Noktası (/predict) test ediliyor...")
        
        # 1. Havada sanal bir 28x28 siyah resim oluştur
        dummy_image_array = np.zeros((28, 28), dtype=np.uint8)
        image = Image.fromarray(dummy_image_array, mode='L')
        
        # 2. Resmi bellekte tut ve API'ye gönderilecek formata çevir
        image_bytes = BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        
        files = {'file': ('test_image.png', image_bytes, 'image/png')}
        
        # 3. API'ye gönder (POST isteği)
        try:
            response = requests.post(f"{self.API_URL}/predict", files=files)
        except Exception as e:
            raise Exception(f"Tahmin isteği gönderilirken ağ hatası oluştu: {str(e)}")

        # 4. Gelen cevabı doğrula
        if response.status_code != 200:
            raise Exception(f"API Hata Döndürdü! Durum Kodu: {response.status_code}, Detay: {response.text}")
        
        result = response.json()
        
        # 5. Beklenen JSON formatının dönüp dönmediğini kontrol et (main.py'de yazdığımız format)
        if "tahmin" not in result or "guven_skoru_yuzde" not in result:
            raise Exception(f"API Beklenmeyen JSON Formatı Döndürdü! Cevap: {result}")
        
        print("MÜKEMMEL! Uçtan Uca (E2E) Dağıtım Testi başarıyla tamamlandı.")
        print(f"API'den Gelen Örnek Yanıt: {result}")

    # Testleri sırasıyla çalıştırır
    def run_tests(self) -> None:
        self.check_health()
        self.test_prediction_endpoint()

if __name__ == "__main__":
    tester = LiveAPITester()
    tester.run_tests()