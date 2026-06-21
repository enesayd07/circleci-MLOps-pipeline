import os
import sys
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, List, Any
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class Train:
    DATA_DIR: str = "training_data"
    MODEL_DIR: str = "training_data/trained_model"
    EPOCHS: int = 30
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.001
    NUM_CLASSES: int = 10
    PATIENCE_STOP: int = 5
    PATIENCE_LR: int = 3

    def __init__(self, base_path: str) -> None:
        self.base_path: str = base_path
        self.data_path: str = os.path.join(self.base_path, self.DATA_DIR)
        self.model_path: str = os.path.join(self.base_path, self.MODEL_DIR)

    def load_files(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_x: np.ndarray = np.load(os.path.join(self.data_path, "train_images.npy"))
        train_y: np.ndarray = np.load(os.path.join(self.data_path, "train_labels.npy"))
        val_x: np.ndarray = np.load(os.path.join(self.data_path, "val_images.npy"))
        val_y: np.ndarray = np.load(os.path.join(self.data_path, "val_labels.npy"))
        test_x: np.ndarray = np.load(os.path.join(self.data_path, "test_images.npy"))
        test_y: np.ndarray = np.load(os.path.join(self.data_path, "test_labels.npy"))
        return train_x, train_y, val_x, val_y, test_x, test_y

    def calculate_weights(self, y: np.ndarray) -> Dict[int, float]:
        class_weights: Dict[int, float] = {}
        total_samples: int = len(y)
        for i in range(self.NUM_CLASSES):
            class_count: int = int(np.sum(y == i))
            if class_count > 0:
                class_weights[i] = total_samples / (self.NUM_CLASSES * float(class_count))
            else:
                class_weights[i] = 1.0
        return class_weights

    def create_model(self) -> Sequential:
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3)) # Ezberlemeyi önler
        model.add(Dense(self.NUM_CLASSES, activation='softmax'))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def setup_callbacks(self) -> List[Any]:
        early_stop = EarlyStopping(monitor='val_loss', patience=self.PATIENCE_STOP, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=self.PATIENCE_LR, min_lr=1e-6)
        return [early_stop, reduce_lr]

    def test_model(self, model: Sequential, x_test: np.ndarray, y_test: np.ndarray) -> None:
        print("\n--- Kör Test (Test Seti) Değerlendirmesi ---")
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Kaybı (Loss): {loss:.4f}")
        print(f"Test Başarımı (Accuracy): {accuracy:.4f}")

    def run_all(self) -> None:
        train_x, train_y, val_x, val_y, test_x, test_y = self.load_files()
        
        weights: Dict[int, float] = self.calculate_weights(train_y)
        model: Sequential = self.create_model()
        callbacks: List[Any] = self.setup_callbacks()

        print("\nEğitim Başlıyor...")
        model.fit(
            train_x, train_y,
            validation_data=(val_x, val_y),
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            class_weight=weights,
            callbacks=callbacks,
            verbose=2
        )

        self.test_model(model, test_x, test_y)

        print(f"\nModel Kaydediliyor: {self.model_path}")
        model.save(self.model_path)

if __name__ == "__main__":
    current_dir: str = sys.path[0]
    trainer = Train(base_path=current_dir)
    trainer.run_all()