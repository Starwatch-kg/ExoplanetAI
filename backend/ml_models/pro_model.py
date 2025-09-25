"""
Advanced Pro Model with Autoencoder and Attention for Exoplanet Detection
"""
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ProModel(tf.keras.Model):
    """Advanced professional model with autoencoder and attention mechanism"""

    def __init__(self, input_shape, latent_dim: int = 32):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Build model architecture
        self.model = self._build_model()
        self.threshold = None

    def _build_model(self):
        """Build advanced autoencoder with attention"""

        # Encoder with attention
        encoder_input = layers.Input(shape=self.input_shape, name='encoder_input')

        # Convolutional feature extraction
        x = layers.Conv1D(64, 5, activation='relu', padding='same')(encoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)

        # Attention mechanism
        attention = layers.Attention()([x, x])

        # Combine features with attention
        x = layers.Add()([x, attention])
        x = layers.BatchNormalization()(x)

        # Latent representation
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)

        # Decoder
        x = layers.Dense(64, activation='relu')(latent)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Output reconstruction
        decoder_output = layers.Dense(self.input_shape[0], activation='linear', name='decoder_output')(x)

        # Full autoencoder model
        autoencoder = models.Model(encoder_input, decoder_output, name='autoencoder')

        # Compile with reconstruction loss
        autoencoder.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='mean_squared_error',
            metrics=['mae']
        )

        return autoencoder

    def call(self, inputs):
        """Forward pass"""
        return self.model(inputs)

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)

    def train_model(self, X_train, y_train=None, epochs=50, batch_size=32,
                   validation_split=0.2, callbacks=None):
        """Train autoencoder (unsupervised)"""
        if callbacks is None:
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]

        logger.info("Training Pro autoencoder model...")

        # Train autoencoder (unsupervised)
        history = self.model.fit(
            X_train, X_train,  # Reconstruction target is input itself
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        # Calculate reconstruction error threshold on training data
        reconstructions = self.model.predict(X_train, verbose=0)
        reconstruction_errors = np.mean(np.square(X_train - reconstructions), axis=1)
        self.threshold = np.percentile(reconstruction_errors, 95)  # 95th percentile

        logger.info(f"Reconstruction threshold set to: {self.threshold:.4f}")
        return history.history

    def predict_anomaly(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies based on reconstruction error"""
        reconstructions = self.model.predict(X, verbose=0)
        reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)

        # Binary classification based on threshold
        predictions = (reconstruction_errors > self.threshold).astype(int)

        return predictions, reconstruction_errors

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly probabilities"""
        predictions, errors = self.predict_anomaly(X)

        # Convert to probabilities (higher error = higher anomaly probability)
        probs = 1 / (1 + np.exp(-(errors - self.threshold)))  # Sigmoid
        return np.column_stack([1-probs, probs])  # [normal_prob, anomaly_prob]

    def evaluate_model(self, X_test, y_test=None):
        """Evaluate model performance"""
        reconstructions = self.model.predict(X_test, verbose=0)
        reconstruction_errors = np.mean(np.square(X_test - reconstructions), axis=1)

        # Calculate metrics if labels available
        if y_test is not None:
            # For anomaly detection, we consider anomalies as class 1
            predicted_anomalies = (reconstruction_errors > self.threshold).astype(int)

            accuracy = np.mean(predicted_anomalies == y_test)
            precision = np.sum((predicted_anomalies == 1) & (y_test == 1)) / max(np.sum(predicted_anomalies == 1), 1)
            recall = np.sum((predicted_anomalies == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1)

            return {
                "reconstruction_loss": np.mean(reconstruction_errors),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "threshold": self.threshold
            }
        else:
            return {
                "reconstruction_loss": np.mean(reconstruction_errors),
                "threshold": self.threshold
            }

    def get_model_info(self):
        """Get model information"""
        return {
            "name": "Advanced Pro Autoencoder",
            "type": "Autoencoder with Attention",
            "description": "Professional autoencoder model for anomaly detection",
            "input_shape": self.input_shape,
            "latent_dim": self.latent_dim,
            "architecture": "Conv1D + Attention + Autoencoder",
            "optimizer": "Adam (lr=1e-4)",
            "loss": "mean_squared_error",
            "threshold": self.threshold
        }

    def get_model_summary(self):
        """Get model architecture summary"""
        return str(self.model.summary())
