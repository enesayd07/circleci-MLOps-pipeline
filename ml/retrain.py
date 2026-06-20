import os
import sys
import numpy as np
import tensorflow as tf
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class Retrain:
    DATA_DIR: str = "training_data"
    EPOCHS: int = 5 # Simülasyon olduğu için kısa tutuyoruz
    BATCH_SIZE: int = 64
    MIN_REQUIRED_ACCURACY: float = 0.95 # Bilerek ÇOK YÜKSEK tutuyoruz ki CircleCI'da başarısız olsun!
    DRIFT_NOISE: float = 0.5 # Veriyi bozacak aşırı gürültü (Data Drift simülasyonu)

    def __init__(self, base_path: str) -> None:
        self.base_path: str = base_path
        self.data_path: str = os.path.join(self.base_path, self.DATA_DIR)

    # Diskteki verileri yükler
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_x: np.ndarray = np.load(os.path.join(self.data_path, "train_images.npy"))
        train_y: np.ndarray = np.load(os.path.join(self.data_path, "train_labels.npy"))
        test_x: np.ndarray = np.load(os.path.join(self.data_path, "test_images.npy"))
        test_y: np.ndarray = np.load(os.path.join(self.data_path, "test_labels.npy"))
        return train_x, train_y, test_x, test_y

    # Yeni gelen verinin bozuk/farklı olduğunu simüle eder (Kavram Kayması - Concept Drift)
    def simulate_data_drift(self, x: np.ndarray) -> np.ndarray:
        print("\n[UYARI] Veri kayması (Data Drift) simüle ediliyor! Piksellere aşırı gürültü ekleniyor...")
        noise: np.ndarray = self.DRIFT_NOISE * np.random.randn(*x.shape)
        x_corrupted: np.ndarray = x + noise
        return np.clip(x_corrupted, 0.0, 1.0)

    # Yeni (ve bozuk) veriyle modeli hızlıca baştan eğitir
    def retrain_model(self, train_x: np.ndarray, train_y: np.ndarray) -> Sequential:
        model = Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("\n[BİLGİ] Yeni verilerle Yeniden Eğitim (Retraining) başlatılıyor...")
        model.fit(train_x, train_y, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, verbose=2)
        return model

    # Yeniden eğitilen modelin canlıya çıkmaya hazır olup olmadığını çok katı bir barajla test eder
    def validate_new_model(self, model: Sequential, test_x: np.ndarray, test_y: np.ndarray) -> None:
        print("\n[KALİTE KONTROL] Yeniden eğitilen model test ediliyor...")
        loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
        print(f"Yeni Model Başarımı: {accuracy:.4f} (Hedef: {self.MIN_REQUIRED_ACCURACY})")

        if accuracy < self.MIN_REQUIRED_ACCURACY:
            # Bu hata CircleCI'ı durduracak ve Slack/Email alarmı verdirecek (when: on_fail)
            raise Exception(f"MLOps ALARMI: Yeni veriler modelin doğruluğunu düşürdü! "
                            f"Başarım {accuracy:.4f} seviyesinde kaldı. Modelin canlıya çıkması engellendi.")
        print("Model kalite kontrolü geçti. Canlıya alınabilir.")

    # Gece mesaisi (Cron Job) çalıştığında sırasıyla bu adımları uygular
    def execute_nightly_job(self) -> None:
        train_x, train_y, test_x, test_y = self.load_data()
        
        # Sadece eğitim verisini bozuyoruz ki model kötü öğrensin, ama test verisi gerçek kalsın
        corrupted_train_x = self.simulate_data_drift(train_x)
        
        new_model = self.retrain_model(corrupted_train_x, train_y)
        self.validate_new_model(new_model, test_x, test_y)

if __name__ == "__main__":
    current_dir: str = sys.path[0]
    retrainer = Retrain(base_path=current_dir)
    retrainer.execute_nightly_job()