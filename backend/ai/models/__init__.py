"""
Neural Network Models for Transit Detection

Коллекция моделей машинного обучения для обнаружения транзитов экзопланет:
- CNN: Сверточные нейронные сети для выделения паттернов
- LSTM: Рекуррентные сети для временных рядов  
- Transformers: Современная архитектура для последовательностей
"""

from .cnn_classifier import CNNClassifier
from .lstm_classifier import LSTMClassifier
from .transformer_classifier import TransformerClassifier
from .base_model import BaseTransitModel

__all__ = [
    "CNNClassifier",
    "LSTMClassifier", 
    "TransformerClassifier",
    "BaseTransitModel"
]
