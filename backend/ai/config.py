"""
Configuration for AI Module

Конфигурация для AI модуля анализа кривых блеска.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import torch

class AIConfig:
    """Конфигурация AI модуля"""
    
    # Устройство вычислений
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Пути к моделям
    MODEL_DIR = Path("models")
    CHECKPOINT_DIR = Path("checkpoints")
    CACHE_DIR = Path("data/cache")
    EMBEDDINGS_DIR = Path("data/embeddings")
    
    # Параметры моделей
    CNN_CONFIG = {
        'input_size': 1024,
        'num_classes': 2,
        'num_filters': (32, 64, 128, 256),
        'kernel_sizes': (7, 5, 3, 3),
        'dropout': 0.1,
        'use_attention': True,
        'use_residual': True
    }
    
    LSTM_CONFIG = {
        'input_size': 1024,
        'num_classes': 2,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.1,
        'bidirectional': True,
        'use_attention': True,
        'use_layer_norm': True
    }
    
    TRANSFORMER_CONFIG = {
        'input_size': 1024,
        'num_classes': 2,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'max_seq_length': 2048,
        'use_positional_encoding': True,
        'pooling_strategy': 'attention'
    }
    
    # Параметры ансамбля
    ENSEMBLE_CONFIG = {
        'combination_strategy': 'weighted',
        'weights': {'cnn': 0.3, 'lstm': 0.3, 'transformer': 0.4},
        'use_meta_learner': True,
        'temperature': 1.0
    }
    
    # Параметры обучения
    TRAINING_CONFIG = {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 100,
        'early_stopping_patience': 10,
        'gradient_clip_norm': 1.0,
        'optimizer_type': 'adamw',
        'scheduler_type': 'cosine'
    }
    
    # Параметры embeddings
    EMBEDDING_CONFIG = {
        'embedding_dim': 256,
        'similarity_threshold': 0.95,
        'max_cache_size': 10000,
        'use_faiss': True
    }
    
    # База данных
    DATABASE_CONFIG = {
        'url': os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/exoplanet_ai'),
        'max_connections': 10,
        'enable_database': True
    }
    
    # Логирование
    LOGGING_CONFIG = {
        'use_wandb': False,
        'use_mlflow': False,
        'log_level': 'INFO'
    }
    
    # Предобработка данных
    PREPROCESSING_CONFIG = {
        'target_length': 1024,
        'normalization_method': 'median',
        'outlier_removal': True,
        'outlier_threshold': 1.5  # IQR multiplier
    }
    
    # Валидация
    VALIDATION_CONFIG = {
        'confidence_threshold': 0.7,
        'snr_threshold': 7.0,
        'use_ai_validation': True
    }
    
    @classmethod
    def get_model_config(cls, model_type: str) -> Dict[str, Any]:
        """Получение конфигурации модели по типу"""
        configs = {
            'cnn': cls.CNN_CONFIG,
            'lstm': cls.LSTM_CONFIG,
            'transformer': cls.TRANSFORMER_CONFIG,
            'ensemble': cls.ENSEMBLE_CONFIG
        }
        return configs.get(model_type.lower(), {})
    
    @classmethod
    def update_config(cls, config_dict: Dict[str, Any]):
        """Обновление конфигурации из словаря"""
        for key, value in config_dict.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)
    
    @classmethod
    def create_directories(cls):
        """Создание необходимых директорий"""
        directories = [
            cls.MODEL_DIR,
            cls.CHECKPOINT_DIR,
            cls.CACHE_DIR,
            cls.EMBEDDINGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls) -> bool:
        """Валидация конфигурации"""
        try:
            # Проверяем доступность устройства
            if cls.DEVICE == 'cuda' and not torch.cuda.is_available():
                cls.DEVICE = 'cpu'
            
            # Создаем директории
            cls.create_directories()
            
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

# Глобальная конфигурация
config = AIConfig()

# Валидация при импорте
if not config.validate_config():
    print("Warning: AI configuration validation failed, using defaults")
