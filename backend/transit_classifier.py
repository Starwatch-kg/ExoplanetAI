import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class TransitClassifier:
    """
    CNN для классификации транзитов экзопланет
    """
    def __init__(self, input_shape=(100, 1)):
        self.model = self.build_model(input_shape)
        
    def build_model(self, input_shape):
        """Создает архитектуру CNN"""
        model = models.Sequential([
            layers.InputLayer(input_shape=input_shape),
            
            # Блок 1
            layers.Conv1D(32, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            
            # Блок 2
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            
            # Блок 3
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            
            # Классификатор
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(3, activation='softmax')  # 3 класса: планета, звезда, шум
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
        
    def train(self, X_train, y_train, epochs=10):
        """Обучение модели"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_split=0.2,
            batch_size=32
        )
        return history
        
    def predict(self, light_curve):
        """Предсказание класса для кривой блеска"""
        # Предобработка и нормализация
        processed = self.preprocess(light_curve)
        
        # Предсказание
        predictions = self.model.predict(np.expand_dims(processed, axis=0))
        return np.argmax(predictions[0]), predictions[0]
        
    def preprocess(self, light_curve):
        """Нормализация кривой блеска"""
        light_curve = np.array(light_curve)
        
        # Удаляем выбросы (3-sigma clipping)
        mean = np.mean(light_curve)
        std = np.std(light_curve)
        mask = np.abs(light_curve - mean) < 3 * std
        
        if np.sum(mask) > len(light_curve) * 0.5:  # Если осталось больше 50% точек
            clean_curve = light_curve[mask]
            mean = np.mean(clean_curve)
            std = np.std(clean_curve)
        
        # Нормализация
        normalized = (light_curve - mean) / (std + 1e-8)
        
        # Обрезаем экстремальные значения
        normalized = np.clip(normalized, -5, 5)
        
        return normalized.reshape(-1, 1)  # Добавляем размерность канала
    
    def save_model(self, filepath):
        """Сохранение модели"""
        self.model.save(filepath)
        
    def load_model(self, filepath):
        """Загрузка модели"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False
