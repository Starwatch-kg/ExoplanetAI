"""
Base Model Class for Transit Detection

Базовый класс для всех моделей обнаружения транзитов.
Определяет общий интерфейс и функциональность.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class BaseTransitModel(nn.Module, ABC):
    """
    Базовый класс для всех моделей обнаружения транзитов
    """
    
    def __init__(self, 
                 input_size: int = 1024,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.model_type = self.__class__.__name__
        
        # Метаданные модели
        self.metadata = {
            'model_type': self.model_type,
            'input_size': input_size,
            'num_classes': num_classes,
            'dropout': dropout,
            'version': '1.0.0',
            'created_at': None,
            'trained_on': [],
            'performance_metrics': {}
        }
        
        # Инициализация архитектуры
        self._build_model()
        
    @abstractmethod
    def _build_model(self):
        """Построение архитектуры модели"""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход модели"""
        pass
    
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        Предсказание вероятностей классов
        
        Args:
            x: Входные данные [batch_size, sequence_length] или [batch_size, channels, length]
            
        Returns:
            Вероятности классов [batch_size, num_classes]
        """
        self.eval()
        with torch.no_grad():
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # Добавляем канал если нужно
            
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=-1)
            
        return probabilities.cpu().numpy()
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """
        Предсказание классов
        
        Args:
            x: Входные данные
            threshold: Порог для бинарной классификации
            
        Returns:
            Предсказанные классы [batch_size]
        """
        probabilities = self.predict_proba(x)
        
        if self.num_classes == 2:
            # Бинарная классификация
            return (probabilities[:, 1] > threshold).astype(int)
        else:
            # Многоклассовая классификация
            return np.argmax(probabilities, axis=1)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Извлечение признаков из промежуточных слоев
        
        Args:
            x: Входные данные
            
        Returns:
            Вектор признаков
        """
        # Базовая реализация - возвращает выход предпоследнего слоя
        self.eval()
        with torch.no_grad():
            features = self._extract_features_impl(x)
        return features
    
    @abstractmethod
    def _extract_features_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Реализация извлечения признаков для конкретной модели"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            **self.metadata,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2),  # Приблизительно
        }
        
        return info
    
    def save_model(self, path: str, save_metadata: bool = True):
        """
        Сохранение модели
        
        Args:
            path: Путь для сохранения
            save_metadata: Сохранять ли метаданные
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем веса модели
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_size': self.input_size,
                'num_classes': self.num_classes,
                'dropout': self.dropout
            },
            'metadata': self.metadata
        }, path)
        
        # Сохраняем метаданные отдельно
        if save_metadata:
            metadata_path = path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.get_model_info(), f, indent=2)
        
        logger.info(f"Модель сохранена: {path}")
    
    def load_model(self, path: str):
        """
        Загрузка модели
        
        Args:
            path: Путь к сохраненной модели
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # Загружаем веса
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Загружаем метаданные
        if 'metadata' in checkpoint:
            self.metadata.update(checkpoint['metadata'])
        
        logger.info(f"Модель загружена: {path}")
    
    def update_metadata(self, **kwargs):
        """Обновление метаданных модели"""
        self.metadata.update(kwargs)
    
    def add_training_history(self, dataset_name: str, metrics: Dict[str, float]):
        """
        Добавление информации об обучении
        
        Args:
            dataset_name: Название датасета
            metrics: Метрики производительности
        """
        if dataset_name not in self.metadata['trained_on']:
            self.metadata['trained_on'].append(dataset_name)
        
        self.metadata['performance_metrics'][dataset_name] = metrics
    
    def freeze_layers(self, layer_names: Optional[List[str]] = None):
        """
        Заморозка слоев для transfer learning
        
        Args:
            layer_names: Список имен слоев для заморозки. Если None, замораживаются все слои
        """
        if layer_names is None:
            # Замораживаем все слои
            for param in self.parameters():
                param.requires_grad = False
        else:
            # Замораживаем указанные слои
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
        
        logger.info(f"Заморожены слои: {layer_names if layer_names else 'все'}")
    
    def unfreeze_layers(self, layer_names: Optional[List[str]] = None):
        """
        Разморозка слоев
        
        Args:
            layer_names: Список имен слоев для разморозки. Если None, размораживаются все слои
        """
        if layer_names is None:
            # Размораживаем все слои
            for param in self.parameters():
                param.requires_grad = True
        else:
            # Размораживаем указанные слои
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
        
        logger.info(f"Разморожены слои: {layer_names if layer_names else 'все'}")
    
    def get_layer_names(self) -> List[str]:
        """Получение списка имен слоев"""
        return [name for name, _ in self.named_modules() if name]
    
    def count_parameters(self) -> Dict[str, int]:
        """Подсчет параметров модели"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }
