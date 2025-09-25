"""
Individual ML Model interfaces for easy integration
"""
from exoplanet_detector import CNNTransitClassifier, LSTMTimeSeriesAnalyzer, AutoencoderAnomalyDetector, ExoplanetEnsembleDetector
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class DetectorModel:
    """CNN-based detector model"""

    def __init__(self, input_shape):
        self.model = CNNTransitClassifier(input_shape, num_classes=3)

    def build(self):
        return self.model.build_model()

    def predict(self, X):
        return self.model.predict_proba(X)

    def train(self, X, y):
        return self.model.train(X, y)

class VerifierModel:
    """LSTM-based verifier model"""

    def __init__(self, input_shape):
        self.model = LSTMTimeSeriesAnalyzer(input_shape, num_classes=3)

    def build(self):
        return self.model.build_model()

    def predict(self, X):
        return self.model.predict_proba(X)

    def train(self, X, y):
        return self.model.train(X, y)

class ProModel:
    """Autoencoder-based professional model"""

    def __init__(self, input_shape):
        self.model = AutoencoderAnomalyDetector(input_shape)

    def build(self):
        return self.model.build_model()

    def predict(self, X):
        predictions, errors = self.model.predict_anomaly(X)
        probabilities = self.model.predict_proba(X)
        return probabilities

    def train(self, X, y=None):
        return self.model.train(X, y)

# Pre-configured models for different use cases
def create_detector_model(input_shape):
    """Create CNN detector model"""
    return DetectorModel(input_shape)

def create_verifier_model(input_shape):
    """Create LSTM verifier model"""
    return VerifierModel(input_shape)

def create_pro_model(input_shape):
    """Create Autoencoder pro model"""
    return ProModel(input_shape)

def create_ensemble_model(input_shape):
    """Create ensemble model"""
    return ExoplanetEnsembleDetector(input_shape, num_classes=3)
