import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from typing import Dict

class Test:
    DATA_DIR: str = "training_data"
    MIN_OVERALL_ACCURACY: float = 0.80
    MIN_MINORITY_ACCURACY: float = 0.50
    MINORITY_CLASS: int = 9

    def __init__(self, base_path: str) -> None:
        self.base_path: str = base_path
        self.data_path: str = os.path.join(self.base_path, self.DATA_DIR)
        self.model_path: str = os.path.join(self.base_path, "training_data/trained_model")

    # Diske kaydedilmiş olan test verilerini ve eğitilmiş modeli yükler
    def load_test_files(self) -> tuple:
        model = tf.keras.models.load_model(self.model_path)
        x_test = np.load(os.path.join(self.data_path, "test_images.npy"))
        y_test = np.load(os.path.join(self.data_path, "test_labels.npy"))
        return model, x_test, y_test

    # Modelin performansını hesaplar ve raporu konsola yazdırır
    def run_evaluation(self, model, x_test: np.ndarray, y_test: np.ndarray) -> Dict[int, float]:
        predictions = np.argmax(model.predict(x_test), axis=-1)
        report = classification_report(y_test, predictions, output_dict=True)
        
        overall_acc = report['accuracy']
        minority_acc = report[str(self.MINORITY_CLASS)]['recall']
        
        print(f"\nGenel Başarım (Accuracy): {overall_acc:.4f}")
        print(f"Azınlık Sınıf ({self.MINORITY_CLASS}) Başarımı: {minority_acc:.4f}")
        
        return {"overall": overall_acc, "minority": minority_acc}

    # Modelin başarısı belirlenen barajın altında kalırsa CI/CD hattını durdurur
    def check_quality(self, metrics: Dict[str, float]) -> None:
        if metrics["overall"] < self.MIN_OVERALL_ACCURACY:
            raise Exception(f"HATA: Genel başarı {self.MIN_OVERALL_ACCURACY} altında!")
        
        if metrics["minority"] < self.MIN_MINORITY_ACCURACY:
            raise Exception(f"HATA: Azınlık sınıf başarısı {self.MIN_MINORITY_ACCURACY} altında!")
        
        print("\nKalite kontrol başarıyla geçildi!")

    # Tüm test süreçlerini sırasıyla çalıştırır
    def run_all(self) -> None:
        model, x_test, y_test = self.load_test_files()
        metrics = self.run_evaluation(model, x_test, y_test)
        self.check_quality(metrics)

if __name__ == "__main__":
    current_dir: str = sys.path[0]
    tester = Test(base_path=current_dir)
    tester.run_all()
