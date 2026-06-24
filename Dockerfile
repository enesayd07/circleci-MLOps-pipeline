# 1. Temel İşletim Sistemi ve Python Sürümü 
FROM python:3.11-slim

# 2. Çalışma Dizinini Ayarla 
WORKDIR /app

# 3. Sistem kütüphanelerini güncelle 
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Gereksinim Dosyasını Kopyala ve Kütüphaneleri Kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Tüm Proje Dosyalarını Kopyala
COPY . .

# Modeli baştan eğitip diske kaydettiriyoruz
RUN python ml/build.py && python ml/train.py

# 6. Render'ın uygulamayı dışarı açması için Port ayarı
EXPOSE 8000

# 7. Uygulamayı Başlatma Komutu 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
