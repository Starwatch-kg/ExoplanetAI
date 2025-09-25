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
        mean = np.mean(light_curve)
        std = np.std(light_curve)
        return (light_curve - mean) / std
