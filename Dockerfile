# 1. Temel İşletim Sistemi ve Python Sürümü (Hafif bir sürüm seçiyoruz)
FROM python:3.11-slim

# 2. Çalışma Dizinini Ayarla (Konteynerin içindeki klasör)
WORKDIR /app

# 3. Sistem kütüphanelerini güncelle (Görüntü işleme kütüphaneleri için bazen gerekebilir)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Gereksinim Dosyasını Kopyala ve Kütüphaneleri Kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Tüm Proje Dosyalarını Kopyala
COPY . .

# -- YENİ EKLENEN KISIM --
# 5.5 Render inşası sırasında modeli baştan eğitip diskte oluşturmasını sağlıyoruz.
# Bu işlem Render'ın build süresini 1 dakika uzatır ama modeli sunucunun içine gömer.
RUN python ml/build.py && python ml/train.py

# 6. Render'ın uygulamayı dışarı açması için Port ayarı
EXPOSE 8000

# 7. Uygulamayı Başlatma Komutu (FastAPI'yi uvicorn ile başlatır)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]