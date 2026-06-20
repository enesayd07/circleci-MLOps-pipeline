import os
import sys
import numpy as np
from typing import Tuple, Dict
from tensorflow.keras.datasets import fashion_mnist

class Build:
    SAVE_DIR: str = "training_data"
    VAL_SPLIT: float = 0.15
    NOISE_FACTOR: float = 0.05
    IMBALANCE_CLASS: int = 9
    IMBALANCE_RATIO: float = 0.2
    SEED: int = 42

    def __init__(self, base_path: str) -> None:
        self.base_path: str = base_path
        self.save_path: str = os.path.join(self.base_path, self.SAVE_DIR)
        os.makedirs(self.save_path, exist_ok=True)

    # Veriyi internetten ham haliyle indirir
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
        return train_x, train_y, test_x, test_y

    # Belli bir kıyafet sınıfını bilerek azaltıp veriyi dengesizleştirir
    def reduce_data(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(self.SEED)
        mask: np.ndarray = y == self.IMBALANCE_CLASS
        imbalance_idx: np.ndarray = np.where(mask)[0]
        normal_idx: np.ndarray = np.where(~mask)[0]

        keep_count: int = int(len(imbalance_idx) * self.IMBALANCE_RATIO)
        keep_idx: np.ndarray = np.random.choice(imbalance_idx, keep_count, replace=False)

        final_idx: np.ndarray = np.concatenate([normal_idx, keep_idx])
        np.random.shuffle(final_idx)

        return x[final_idx], y[final_idx]

    # Pikselleri 0-1 arasına sıkıştırır ve matris boyutunu ayarlar
    def format_data(self, x: np.ndarray) -> np.ndarray:
        x_scaled: np.ndarray = x.astype(np.float32) / 255.0
        x_reshaped: np.ndarray = np.expand_dims(x_scaled, axis=-1)
        return x_reshaped

    # Modelin ezberlemesini önlemek için resimlere kirlilik (gürültü) ekler
    def add_noise(self, x: np.ndarray) -> np.ndarray:
        np.random.seed(self.SEED)
        noise: np.ndarray = self.NOISE_FACTOR * np.random.randn(*x.shape)
        x_noisy: np.ndarray = x + noise
        return np.clip(x_noisy, 0.0, 1.0)

    # Veriyi eğitim (train) ve doğrulama (validation) olarak ikiye böler
    def split_data(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(self.SEED)
        indices: np.ndarray = np.random.permutation(len(x))
        val_size: int = int(len(x) * self.VAL_SPLIT)

        val_idx: np.ndarray = indices[:val_size]
        train_idx: np.ndarray = indices[val_size:]

        return x[train_idx], y[train_idx], x[val_idx], y[val_idx]

    # Hazırlanan verilerde bir hata veya anormallik var mı diye kontrol eder
    def check_errors(self, x: np.ndarray) -> None:
        assert np.max(x) <= 1.0
        assert np.min(x) >= 0.0
        assert len(x.shape) == 4

    # Tüm işlemleri biten verileri bilgisayara (diske) kaydeder
    def save_files(self, data_dict: Dict[str, np.ndarray]) -> None:
        for name, data in data_dict.items():
            file_path: str = os.path.join(self.save_path, f"{name}.npy")
            np.save(file_path, data)

    # Fabrikadaki tüm bu işlemleri sırasıyla çalıştırır
    def run_all(self) -> None:
        raw_train_x, raw_train_y, raw_test_x, raw_test_y = self.get_data()

        train_x, train_y = self.reduce_data(raw_train_x, raw_train_y)

        train_x = self.format_data(train_x)
        test_x = self.format_data(raw_test_x)

        train_x = self.add_noise(train_x)
        test_x = self.add_noise(test_x)

        train_x, train_y, val_x, val_y = self.split_data(train_x, train_y)

        self.check_errors(train_x)
        self.check_errors(val_x)
        self.check_errors(test_x)

        processed_data: Dict[str, np.ndarray] = {
            "train_images": train_x,
            "train_labels": train_y,
            "val_images": val_x,
            "val_labels": val_y,
            "test_images": test_x,
            "test_labels": raw_test_y
        }

        self.save_files(processed_data)

if __name__ == "__main__":
    current_dir: str = sys.path[0]
    preparer = Build(base_path=current_dir)
    preparer.run_all()