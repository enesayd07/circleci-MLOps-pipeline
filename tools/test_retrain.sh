#!/bin/bash

# Geliştirme aşamasında Yeniden Eğitim (Retrain) senaryosunu lokalde test eder.
set -e

echo "1. Veri Hazırlama Başlıyor..."
python3 ./ml/build.py 

echo "2. Yeniden Eğitim ve Kalite Kontrol Başlıyor..."
python3 ./ml/retrain.py

echo "Başarılı! Yeniden eğitim senaryosu lokal ortamda test edildi."