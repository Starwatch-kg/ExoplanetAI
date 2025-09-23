"""
Ensemble Classifier for Transit Detection

Ансамбль моделей для повышения точности классификации транзитов.
Комбинирует CNN, LSTM и Transformer для получения лучших результатов.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json

from .models import CNNClassifier, LSTMClassifier, TransformerClassifier, BaseTransitModel

logger = logging.getLogger(__name__)

class EnsembleClassifier(nn.Module):
    """
    Ансамбль классификаторов для обнаружения транзитов
    
    Поддерживает различные стратегии комбинирования:
    - Voting: простое голосование
    - Weighted: взвешенное голосование
    - Stacking: мета-модель для комбинирования
    - Dynamic: адаптивное взвешивание на основе уверенности
    """
    
    def __init__(self,
                 models: Dict[str, BaseTransitModel],
                 combination_strategy: str = 'weighted',
                 weights: Optional[Dict[str, float]] = None,
                 use_meta_learner: bool = False,
                 meta_learner_hidden_size: int = 128,
                 temperature: float = 1.0):
        
        super().__init__()
        
        self.models = nn.ModuleDict(models)
        self.combination_strategy = combination_strategy
        self.temperature = temperature
        self.use_meta_learner = use_meta_learner
        
        # Веса для взвешенного голосования
        if weights is None:
            weights = {name: 1.0 / len(models) for name in models.keys()}
        self.weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(weight))
            for name, weight in weights.items()
        })
        
        # Мета-обучающаяся модель для stacking
        if use_meta_learner or combination_strategy == 'stacking':
            feature_size = sum(model.num_classes for model in models.values())
            self.meta_learner = nn.Sequential(
                nn.Linear(feature_size, meta_learner_hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(meta_learner_hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, list(models.values())[0].num_classes)
            )
        
        # Для динамического взвешивания
        if combination_strategy == 'dynamic':
            self.confidence_network = nn.Sequential(
                nn.Linear(len(models), 64),
                nn.ReLU(),
                nn.Linear(64, len(models)),
                nn.Softmax(dim=-1)
            )
        
        self.model_names = list(models.keys())
        self.num_classes = list(models.values())[0].num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход ансамбля
        
        Args:
            x: Входные данные
            
        Returns:
            Логиты классов [batch_size, num_classes]
        """
        # Получаем предсказания от всех моделей
        model_outputs = {}
        model_probs = {}
        
        for name, model in self.models.items():
            with torch.no_grad() if not self.training else torch.enable_grad():
                logits = model(x)
                probs = F.softmax(logits / self.temperature, dim=-1)
                
                model_outputs[name] = logits
                model_probs[name] = probs
        
        # Комбинируем предсказания
        if self.combination_strategy == 'voting':
            # Простое голосование (среднее арифметическое)
            ensemble_logits = torch.stack(list(model_outputs.values())).mean(dim=0)
            
        elif self.combination_strategy == 'weighted':
            # Взвешенное голосование
            weighted_logits = []
            total_weight = sum(self.weights.values())
            
            for name, logits in model_outputs.items():
                weight = self.weights[name] / total_weight
                weighted_logits.append(weight * logits)
            
            ensemble_logits = torch.stack(weighted_logits).sum(dim=0)
            
        elif self.combination_strategy == 'stacking':
            # Мета-обучение
            stacked_probs = torch.cat(list(model_probs.values()), dim=-1)
            ensemble_logits = self.meta_learner(stacked_probs)
            
        elif self.combination_strategy == 'dynamic':
            # Динамическое взвешивание на основе уверенности
            confidences = []
            for probs in model_probs.values():
                # Уверенность как максимальная вероятность
                confidence = torch.max(probs, dim=-1)[0]
                confidences.append(confidence)
            
            confidence_tensor = torch.stack(confidences, dim=-1)  # [batch_size, num_models]
            dynamic_weights = self.confidence_network(confidence_tensor)  # [batch_size, num_models]
            
            # Взвешенное комбинирование
            weighted_logits = []
            for i, (name, logits) in enumerate(model_outputs.items()):
                weight = dynamic_weights[:, i:i+1]  # [batch_size, 1]
                weighted_logits.append(weight * logits)
            
            ensemble_logits = torch.stack(weighted_logits, dim=0).sum(dim=0)
            
        else:
            raise ValueError(f"Unknown combination strategy: {self.combination_strategy}")
        
        return ensemble_logits
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Предсказание с оценкой неопределенности
        
        Args:
            x: Входные данные
            
        Returns:
            Tuple из (predictions, uncertainties, individual_predictions)
        """
        self.eval()
        
        with torch.no_grad():
            # Получаем предсказания от всех моделей
            individual_probs = {}
            for name, model in self.models.items():
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                individual_probs[name] = probs.cpu().numpy()
            
            # Ансамбль предсказание
            ensemble_logits = self.forward(x)
            ensemble_probs = F.softmax(ensemble_logits, dim=-1).cpu().numpy()
            
            # Вычисляем неопределенность как дисперсию между моделями
            all_probs = np.stack(list(individual_probs.values()), axis=0)  # [num_models, batch_size, num_classes]
            uncertainty = np.var(all_probs, axis=0)  # [batch_size, num_classes]
            
            # Общая неопределенность как сумма дисперсий по классам
            total_uncertainty = np.sum(uncertainty, axis=-1)  # [batch_size]
            
            predictions = np.argmax(ensemble_probs, axis=-1)
        
        return predictions, total_uncertainty, individual_probs
    
    def get_model_contributions(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Получение вклада каждой модели в финальное предсказание
        
        Args:
            x: Входные данные
            
        Returns:
            Словарь с вкладами моделей
        """
        self.eval()
        
        with torch.no_grad():
            if self.combination_strategy == 'weighted':
                total_weight = sum(self.weights.values())
                contributions = {
                    name: (weight / total_weight).item()
                    for name, weight in self.weights.items()
                }
            
            elif self.combination_strategy == 'dynamic':
                # Получаем динамические веса
                model_probs = {}
                for name, model in self.models.items():
                    logits = model(x)
                    probs = F.softmax(logits, dim=-1)
                    model_probs[name] = probs
                
                confidences = []
                for probs in model_probs.values():
                    confidence = torch.max(probs, dim=-1)[0]
                    confidences.append(confidence)
                
                confidence_tensor = torch.stack(confidences, dim=-1)
                dynamic_weights = self.confidence_network(confidence_tensor)
                
                # Усредняем по батчу
                avg_weights = dynamic_weights.mean(dim=0)
                contributions = {
                    name: avg_weights[i].item()
                    for i, name in enumerate(self.model_names)
                }
            
            else:
                # Равные вклады для других стратегий
                contributions = {name: 1.0 / len(self.models) for name in self.model_names}
        
        return contributions
    
    def update_weights(self, performance_metrics: Dict[str, float]):
        """
        Обновление весов моделей на основе метрик производительности
        
        Args:
            performance_metrics: Словарь с метриками для каждой модели
        """
        if self.combination_strategy != 'weighted':
            logger.warning("Weight update is only supported for 'weighted' strategy")
            return
        
        # Нормализуем метрики (предполагаем, что больше = лучше)
        total_performance = sum(performance_metrics.values())
        
        for name in self.model_names:
            if name in performance_metrics:
                new_weight = performance_metrics[name] / total_performance
                self.weights[name].data = torch.tensor(new_weight)
        
        logger.info(f"Updated ensemble weights: {dict(self.weights)}")
    
    def add_model(self, name: str, model: BaseTransitModel, weight: float = 1.0):
        """
        Добавление новой модели в ансамбль
        
        Args:
            name: Имя модели
            model: Модель для добавления
            weight: Вес модели (для weighted strategy)
        """
        self.models[name] = model
        self.weights[name] = nn.Parameter(torch.tensor(weight))
        self.model_names.append(name)
        
        logger.info(f"Added model '{name}' to ensemble")
    
    def remove_model(self, name: str):
        """
        Удаление модели из ансамбля
        
        Args:
            name: Имя модели для удаления
        """
        if name in self.models:
            del self.models[name]
            del self.weights[name]
            self.model_names.remove(name)
            
            logger.info(f"Removed model '{name}' from ensemble")
        else:
            logger.warning(f"Model '{name}' not found in ensemble")
    
    def save_ensemble(self, path: str):
        """
        Сохранение ансамбля
        
        Args:
            path: Путь для сохранения
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем конфигурацию ансамбля
        config = {
            'combination_strategy': self.combination_strategy,
            'model_names': self.model_names,
            'weights': {name: weight.item() for name, weight in self.weights.items()},
            'temperature': self.temperature,
            'use_meta_learner': self.use_meta_learner,
            'num_classes': self.num_classes
        }
        
        with open(path / 'ensemble_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Сохраняем каждую модель отдельно
        for name, model in self.models.items():
            model.save_model(path / f'{name}_model.pth')
        
        # Сохраняем состояние ансамбля (веса, мета-модель и т.д.)
        ensemble_state = {
            'weights': self.weights.state_dict(),
        }
        
        if hasattr(self, 'meta_learner'):
            ensemble_state['meta_learner'] = self.meta_learner.state_dict()
        
        if hasattr(self, 'confidence_network'):
            ensemble_state['confidence_network'] = self.confidence_network.state_dict()
        
        torch.save(ensemble_state, path / 'ensemble_state.pth')
        
        logger.info(f"Ensemble saved to {path}")
    
    def load_ensemble(self, path: str, model_classes: Dict[str, type]):
        """
        Загрузка ансамбля
        
        Args:
            path: Путь к сохраненному ансамблю
            model_classes: Словарь с классами моделей для каждого имени
        """
        path = Path(path)
        
        # Загружаем конфигурацию
        with open(path / 'ensemble_config.json', 'r') as f:
            config = json.load(f)
        
        # Загружаем модели
        models = {}
        for name in config['model_names']:
            if name in model_classes:
                model = model_classes[name]()
                model.load_model(path / f'{name}_model.pth')
                models[name] = model
        
        # Пересоздаем ансамбль
        self.__init__(
            models=models,
            combination_strategy=config['combination_strategy'],
            weights=config['weights'],
            use_meta_learner=config['use_meta_learner'],
            temperature=config['temperature']
        )
        
        # Загружаем состояние ансамбля
        ensemble_state = torch.load(path / 'ensemble_state.pth')
        
        if 'weights' in ensemble_state:
            self.weights.load_state_dict(ensemble_state['weights'])
        
        if 'meta_learner' in ensemble_state and hasattr(self, 'meta_learner'):
            self.meta_learner.load_state_dict(ensemble_state['meta_learner'])
        
        if 'confidence_network' in ensemble_state and hasattr(self, 'confidence_network'):
            self.confidence_network.load_state_dict(ensemble_state['confidence_network'])
        
        logger.info(f"Ensemble loaded from {path}")
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Получение информации об ансамбле"""
        info = {
            'combination_strategy': self.combination_strategy,
            'num_models': len(self.models),
            'model_names': self.model_names,
            'weights': {name: weight.item() for name, weight in self.weights.items()},
            'temperature': self.temperature,
            'use_meta_learner': self.use_meta_learner,
            'total_parameters': sum(
                sum(p.numel() for p in model.parameters())
                for model in self.models.values()
            )
        }
        
        # Добавляем информацию о каждой модели
        info['models_info'] = {}
        for name, model in self.models.items():
            info['models_info'][name] = model.get_model_info()
        
        return info


def create_default_ensemble(input_size: int = 1024, 
                          num_classes: int = 2,
                          device: str = 'cpu') -> EnsembleClassifier:
    """
    Создание ансамбля по умолчанию с CNN, LSTM и Transformer
    
    Args:
        input_size: Размер входной последовательности
        num_classes: Количество классов
        device: Устройство для вычислений
        
    Returns:
        Настроенный ансамбль моделей
    """
    # Создаем модели с оптимизированными параметрами
    models = {
        'cnn': CNNClassifier(
            input_size=input_size,
            num_classes=num_classes,
            num_filters=(32, 64, 128, 256),
            dropout=0.1,
            use_attention=True
        ),
        'lstm': LSTMClassifier(
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            use_attention=True
        ),
        'transformer': TransformerClassifier(
            input_size=input_size,
            num_classes=num_classes,
            d_model=256,
            nhead=8,
            num_layers=4,
            dropout=0.1,
            pooling_strategy='attention'
        )
    }
    
    # Перемещаем модели на устройство
    for model in models.values():
        model.to(device)
    
    # Создаем ансамбль с взвешенным голосованием
    ensemble = EnsembleClassifier(
        models=models,
        combination_strategy='weighted',
        weights={'cnn': 0.3, 'lstm': 0.3, 'transformer': 0.4},  # Transformer получает больший вес
        use_meta_learner=True
    )
    
    ensemble.to(device)
    
    logger.info(f"Created default ensemble with {len(models)} models")
    return ensemble
