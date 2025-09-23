"""
AI Module for Exoplanet Transit Detection

Модуль искусственного интеллекта для анализа кривых блеска звезд 
и классификации транзитов экзопланет.

Включает:
- CNN для выделения сигналов от шума
- RNN/LSTM для анализа временных рядов
- Transformers для повышения точности
- Transfer Learning и Active Learning
- Embeddings и кэширование результатов
"""

from .models import CNNClassifier, LSTMClassifier, TransformerClassifier
from .ensemble import EnsembleClassifier
from .trainer import ModelTrainer
from .predictor import TransitPredictor
from .embeddings import EmbeddingManager
from .database import DatabaseManager

__version__ = "1.0.0"
__author__ = "Exoplanet AI Team"

__all__ = [
    "CNNClassifier",
    "LSTMClassifier", 
    "TransformerClassifier",
    "EnsembleClassifier",
    "ModelTrainer",
    "TransitPredictor",
    "EmbeddingManager",
    "DatabaseManager"
]
