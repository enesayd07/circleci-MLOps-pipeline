#!/bin/bash

# Geliştirme aşamasında Makine Öğrenimi adımlarını lokalde test eder.
# Herhangi bir hata olursa işlemi anında durdurur.
set -e

echo "1. Veri Hazırlama Başlıyor..."
python3 ./ml/build.py

echo "2. Model Eğitimi Başlıyor..."
python3 ./ml/train.py

echo "3. Kalite Kontrol (Test) Başlıyor..."
python3 ./ml/test.py

echo "Başarılı! Model lokal ortamda hatasız eğitildi ve kalite kontrolü geçti."
echo "Not: Dağıtım (Deploy) ve E2E test işlemleri CircleCI ve Render üzerinden otomatik yapılacaktır."