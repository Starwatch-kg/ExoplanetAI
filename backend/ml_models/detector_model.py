"""
Advanced CNN Detector Model for Exoplanet Transit Classification
"""
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class DetectorModel(tf.keras.Model):
    """Advanced CNN-based exoplanet transit detector with multiple classes"""

    def __init__(self, input_shape, num_classes: int = 3):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Build model architecture
        self.model = self._build_model()

    def _build_model(self):
        """Build advanced CNN architecture"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),

            # Batch normalization for input
            layers.BatchNormalization(),

            # First convolutional block
            layers.Conv1D(32, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),

            # Second convolutional block
            layers.Conv1D(64, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),

            # Third convolutional block
            layers.Conv1D(128, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            # Global pooling and dense layers
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Output layer with softmax for multi-class classification
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        return model

    def call(self, inputs):
        """Forward pass"""
        return self.model(inputs)

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        return tf.nn.softmax(self.predict(X)).numpy()

    def train_model(self, X_train, y_train, epochs=50, batch_size=32,
                   validation_split=0.2, callbacks=None):
        """Train the model"""
        if callbacks is None:
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        return history.history

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        results = self.model.evaluate(X_test, y_test, verbose=0)
        return {
            "loss": results[0],
            "accuracy": results[1],
            "precision": results[2],
            "recall": results[3]
        }

    def get_model_info(self):
        """Get model information"""
        return {
            "name": "Advanced CNN Detector",
            "type": "Convolutional Neural Network",
            "description": "Advanced CNN for multi-class exoplanet transit classification",
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "architecture": "CNN with batch norm, dropout, and regularization",
            "optimizer": "Adam (lr=1e-4)",
            "loss": "sparse_categorical_crossentropy"
        }

    def get_model_summary(self):
        """Get model architecture summary"""
        return str(self.model.summary())
