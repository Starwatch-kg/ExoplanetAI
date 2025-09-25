"""
Real ML Models for Exoplanet Detection
Advanced deep learning models using TensorFlow/Keras
"""
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import io
import base64

logger = logging.getLogger(__name__)

class ExoplanetDetectorBase:
    """Base class for exoplanet detection models"""

    def __init__(self, input_shape: Tuple[int, ...], model_name: str = "detector"):
        self.input_shape = input_shape
        self.model_name = model_name
        self.model = None
        self.history = None

    def build_model(self):
        """Build the model architecture - override in subclasses"""
        raise NotImplementedError

    def compile_model(self, learning_rate: float = 1e-4):
        """Compile the model"""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2,
              callbacks: List = None) -> Dict[str, Any]:
        """Train the model"""
        if callbacks is None:
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]

        logger.info(f"Training {self.model_name} model...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X, verbose=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        return tf.nn.softmax(self.predict(X)).numpy()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        results = self.model.evaluate(X_test, y_test, verbose=0)
        return {
            "loss": results[0],
            "accuracy": results[1],
            "precision": results[2],
            "recall": results[3]
        }

    def save_model(self, path: str):
        """Save model to file"""
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from file"""
        self.model = tf.keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")

    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model:
            return str(self.model.summary())
        return "Model not built yet"

class CNNTransitClassifier(ExoplanetDetectorBase):
    """
    CNN-based classifier for exoplanet transit detection
    Uses 1D convolutions to analyze lightcurve patterns
    """

    def __init__(self, input_shape: Tuple[int, ...], num_classes: int = 3):
        super().__init__(input_shape, "CNN_Classifier")
        self.num_classes = num_classes

    def build_model(self):
        """Build CNN architecture for transit classification"""
        model = models.Sequential([
            # Input layer with normalization
            layers.Input(shape=self.input_shape),
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

            # Fourth convolutional block
            layers.Conv1D(256, 5, activation='relu', padding='same',
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

            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model = model
        self.compile_model()

        return model

class LSTMTimeSeriesAnalyzer(ExoplanetDetectorBase):
    """
    LSTM-based analyzer for time series classification
    Good for capturing temporal dependencies in lightcurves
    """

    def __init__(self, input_shape: Tuple[int, ...], num_classes: int = 3):
        super().__init__(input_shape, "LSTM_Analyzer")
        self.num_classes = num_classes

    def build_model(self):
        """Build LSTM architecture"""
        model = models.Sequential([
            layers.Input(shape=self.input_shape),

            # LSTM layers
            layers.LSTM(64, return_sequences=True,
                       kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.LSTM(32, return_sequences=True,
                       kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.LSTM(16, kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            # Dense layers
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Output
            layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model = model
        self.compile_model()

        return model

class AutoencoderAnomalyDetector(ExoplanetDetectorBase):
    """
    Autoencoder for anomaly detection in lightcurves
    Reconstructs normal lightcurves, anomalies have high reconstruction error
    """

    def __init__(self, input_shape: Tuple[int, ...], latent_dim: int = 32):
        super().__init__(input_shape, "Autoencoder_Detector")
        self.latent_dim = latent_dim
        self.threshold = None

    def build_model(self):
        """Build autoencoder architecture"""
        # Encoder
        encoder_input = layers.Input(shape=self.input_shape, name='encoder_input')
        x = layers.Dense(128, activation='relu')(encoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)

        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)

        # Decoder
        x = layers.Dense(32, activation='relu')(latent)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)

        decoder_output = layers.Dense(self.input_shape[0], activation='linear', name='decoder_output')(x)

        # Full autoencoder
        autoencoder = models.Model(encoder_input, decoder_output, name='autoencoder')

        # Encoder model
        encoder = models.Model(encoder_input, latent, name='encoder')

        self.model = autoencoder
        self.encoder = encoder

        # Compile with reconstruction loss
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='mean_squared_error',
            metrics=['mae']
        )

        return autoencoder

    def train(self, X_train: np.ndarray, y_train: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2,
              callbacks: List = None) -> Dict[str, Any]:
        """Train autoencoder (unsupervised)"""
        if callbacks is None:
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]

        logger.info(f"Training {self.model_name} autoencoder...")
        self.history = self.model.fit(
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

        logger.info(f"Reconstruction threshold set to: {self.threshold".4f"}")
        return self.history.history

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

class ExoplanetEnsembleDetector:
    """
    Ensemble model combining CNN, LSTM, and Autoencoder
    Uses voting and weighted predictions for final classification
    """

    def __init__(self, input_shape: Tuple[int, ...], num_classes: int = 3):
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Initialize individual models
        self.cnn_classifier = CNNTransitClassifier(input_shape, num_classes)
        self.lstm_analyzer = LSTMTimeSeriesAnalyzer(input_shape, num_classes)
        self.autoencoder = AutoencoderAnomalyDetector(input_shape)

        # Model weights for ensemble (can be tuned)
        self.weights = {
            'cnn': 0.4,
            'lstm': 0.3,
            'autoencoder': 0.3
        }

        # Training status
        self.trained = False

    def build_models(self):
        """Build all models"""
        logger.info("Building ensemble models...")
        self.cnn_classifier.build_model()
        self.lstm_analyzer.build_model()
        self.autoencoder.build_model()

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    epochs: int = 50, batch_size: int = 32,
                    validation_split: float = 0.2) -> Dict[str, Any]:
        """Train all models"""
        logger.info("Training ensemble models...")

        # Train supervised models
        cnn_history = self.cnn_classifier.train(X_train, y_train, epochs, batch_size, validation_split)
        lstm_history = self.lstm_analyzer.train(X_train, y_train, epochs, batch_size, validation_split)

        # Train autoencoder (unsupervised)
        ae_history = self.autoencoder.train(X_train, None, epochs, batch_size, validation_split)

        self.trained = True

        return {
            'cnn': cnn_history,
            'lstm': lstm_history,
            'autoencoder': ae_history
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.trained:
            raise ValueError("Models must be trained before prediction")

        # Get predictions from each model
        cnn_pred = self.cnn_classifier.predict_proba(X)
        lstm_pred = self.lstm_analyzer.predict_proba(X)

        # Autoencoder gives anomaly scores
        ae_pred, ae_errors = self.autoencoder.predict_anomaly(X)
        ae_proba = self.autoencoder.predict_proba(X)

        # Convert autoencoder binary predictions to one-hot
        ae_onehot = np.zeros((len(ae_pred), 2))
        ae_onehot[np.arange(len(ae_pred)), ae_pred] = 1

        # Weighted ensemble voting
        ensemble_pred = (
            self.weights['cnn'] * cnn_pred +
            self.weights['lstm'] * lstm_pred +
            self.weights['autoencoder'] * ae_proba
        )

        return np.argmax(ensemble_pred, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble prediction probabilities"""
        if not self.trained:
            raise ValueError("Models must be trained before prediction")

        # Get predictions from each model
        cnn_pred = self.cnn_classifier.predict_proba(X)
        lstm_pred = self.lstm_analyzer.predict_proba(X)
        ae_proba = self.autoencoder.predict_proba(X)

        # Weighted ensemble
        ensemble_pred = (
            self.weights['cnn'] * cnn_pred +
            self.weights['lstm'] * lstm_pred +
            self.weights['autoencoder'] * ae_proba
        )

        return ensemble_pred

    def evaluate_ensemble(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate ensemble performance"""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)

        # Calculate metrics
        accuracy = np.mean(predictions == y_test)

        # Per-class performance
        unique_classes = np.unique(y_test)
        class_metrics = {}

        for cls in unique_classes:
            mask = y_test == cls
            class_acc = np.mean(predictions[mask] == y_test[mask])
            class_metrics[f"class_{cls}"] = {
                "accuracy": class_acc,
                "support": np.sum(mask)
            }

        return {
            "accuracy": accuracy,
            "class_metrics": class_metrics,
            "predictions": predictions,
            "probabilities": probabilities
        }

def create_synthetic_dataset(num_samples: int = 10000, sequence_length: int = 100,
                           noise_level: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic dataset for training ML models
    0 = no transit, 1 = transit, 2 = noise/anomaly
    """
    X = []
    y = []

    for i in range(num_samples):
        # Generate base lightcurve
        time = np.linspace(0, 1, sequence_length)
        flux = np.ones(sequence_length) + np.random.normal(0, noise_level, sequence_length)

        # Randomly choose class
        cls = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])

        if cls == 1:  # Transit
            # Add transit signal
            transit_center = np.random.randint(sequence_length//4, 3*sequence_length//4)
            transit_duration = np.random.randint(5, 15)
            transit_depth = np.random.uniform(0.005, 0.05)

            start_idx = max(0, transit_center - transit_duration//2)
            end_idx = min(sequence_length, transit_center + transit_duration//2)

            flux[start_idx:end_idx] *= (1 - transit_depth)

        elif cls == 2:  # Anomaly/Noise
            # Add various types of noise/anomalies
            anomaly_type = np.random.choice(['spike', 'trend', 'periodic_noise'])

            if anomaly_type == 'spike':
                spike_pos = np.random.randint(0, sequence_length)
                flux[spike_pos] += np.random.uniform(0.1, 0.3)
            elif anomaly_type == 'trend':
                trend = np.linspace(0, np.random.uniform(0.05, 0.15), sequence_length)
                flux += trend
            elif anomaly_type == 'periodic_noise':
                freq = np.random.uniform(10, 20)
                noise = 0.02 * np.sin(2 * np.pi * freq * time)
                flux += noise

        X.append(flux.reshape(1, -1))
        y.append(cls)

    return np.array(X), np.array(y)

def train_exoplanet_models():
    """Train all exoplanet detection models"""
    logger.info("Starting exoplanet ML model training...")

    # Create dataset
    X, y = create_synthetic_dataset(num_samples=50000, sequence_length=200)
    print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")

    # Reshape for models
    input_shape = (X.shape[1], X.shape[2])

    # Create and train ensemble
    ensemble = ExoplanetEnsembleDetector(input_shape, num_classes=3)
    ensemble.build_models()

    # Train models
    history = ensemble.train_models(X, y, epochs=30, batch_size=64)

    # Evaluate
    # Create test set
    X_test, y_test = create_synthetic_dataset(num_samples=10000, sequence_length=200)
    evaluation = ensemble.evaluate_ensemble(X_test, y_test)

    logger.info(f"Ensemble accuracy: {evaluation['accuracy']".4f"}")

    return ensemble, history, evaluation

# Global model instances
cnn_model = None
lstm_model = None
autoencoder_model = None
ensemble_model = None

def get_trained_models():
    """Get pre-trained models (train if not available)"""
    global cnn_model, lstm_model, autoencoder_model, ensemble_model

    if ensemble_model is None:
        logger.info("Training exoplanet detection models...")
        ensemble_model, _, _ = train_exoplanet_models()

        # Extract individual models for separate use
        cnn_model = ensemble_model.cnn_classifier
        lstm_model = ensemble_model.lstm_analyzer
        autoencoder_model = ensemble_model.autoencoder

    return {
        'cnn': cnn_model,
        'lstm': lstm_model,
        'autoencoder': autoencoder_model,
        'ensemble': ensemble_model
    }

if __name__ == "__main__":
    # Test the models
    logger.info("Testing exoplanet ML models...")

    models = get_trained_models()

    # Create test data
    X_test, y_test = create_synthetic_dataset(1000, 200)

    # Test predictions
    ensemble = models['ensemble']
    predictions = ensemble.predict(X_test[:10])
    probabilities = ensemble.predict_proba(X_test[:10])

    print(f"Test predictions: {predictions}")
    print(f"Test probabilities shape: {probabilities.shape}")
    print(f"Sample probabilities: {probabilities[0]}")

    logger.info("Exoplanet ML models ready!")
