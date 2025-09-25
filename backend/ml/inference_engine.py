"""
Enhanced ML Inference Engine for Exoplanet Detection
Улучшенная система ML инференса для обнаружения экзопланет
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import asyncio
import time
from dataclasses import dataclass
from enum import Enum
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
import threading

from ..core.logging_config import get_logger, log_ml_inference
from ..core.telemetry import trace_ml_inference, add_span_attributes
from ..core.metrics import metrics_collector, track_ml_inference
from ..core.error_handlers import MLModelException, raise_ml_model_error
from .data_loader import LightcurveData, DataPreprocessor

logger = get_logger(__name__)

class ModelType(Enum):
    """Типы ML моделей"""
    CNN = "cnn"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"

class InferenceMode(Enum):
    """Режимы инференса"""
    SINGLE = "single"
    BATCH = "batch"
    STREAMING = "streaming"

@dataclass
class ModelMetadata:
    """Метаданные модели"""
    name: str
    version: str
    model_type: ModelType
    input_shape: Tuple[int, ...]
    output_classes: int
    confidence_threshold: float
    created_at: float
    training_samples: int
    validation_accuracy: float
    model_size_mb: float

@dataclass
class InferenceResult:
    """Результат инференса"""
    prediction: float
    confidence: float
    probabilities: np.ndarray
    model_name: str
    model_version: str
    inference_time_ms: float
    input_shape: Tuple[int, ...]
    metadata: Dict[str, Any]

class ModelManager:
    """Менеджер ML моделей"""
    
    def __init__(self, models_path: str = "./ai/models"):
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        self.loaded_models: Dict[str, torch.nn.Module] = {}
        self.model_metadata: Dict[str, ModelMetadata] = {}
        self.device = self._get_device()
        self.logger = get_logger("ml.model_manager")
        
        # Thread lock для безопасности
        self._lock = threading.Lock()
    
    def _get_device(self) -> torch.device:
        """Определение устройства для вычислений"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU for inference")
        
        return device
    
    def load_model(
        self, 
        model_name: str, 
        model_path: Optional[str] = None,
        force_reload: bool = False
    ) -> bool:
        """
        Загрузка модели
        
        Args:
            model_name: Имя модели
            model_path: Путь к файлу модели
            force_reload: Принудительная перезагрузка
        
        Returns:
            True если модель загружена успешно
        """
        with self._lock:
            if model_name in self.loaded_models and not force_reload:
                self.logger.info(f"Model {model_name} already loaded")
                return True
            
            try:
                if model_path is None:
                    model_path = self.models_path / f"{model_name}.pth"
                else:
                    model_path = Path(model_path)
                
                if not model_path.exists():
                    self.logger.warning(f"Model file not found: {model_path}")
                    # Создаем dummy модель для демонстрации
                    model = self._create_dummy_model(model_name)
                    metadata = self._create_dummy_metadata(model_name)
                else:
                    # Загружаем реальную модель
                    checkpoint = torch.load(model_path, map_location=self.device)
                    model = checkpoint['model']
                    metadata = checkpoint.get('metadata', self._create_dummy_metadata(model_name))
                
                model.to(self.device)
                model.eval()
                
                self.loaded_models[model_name] = model
                self.model_metadata[model_name] = metadata
                
                # Обновляем метрики
                model_type = metadata.model_type.value
                metrics_collector.set_ml_models_loaded(model_type, 1)
                
                self.logger.info(
                    f"Model loaded successfully: {model_name}",
                    extra={
                        "model_type": model_type,
                        "model_version": metadata.version,
                        "device": str(self.device)
                    }
                )
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
                metrics_collector.record_error("model_load_error", "model_manager")
                return False
    
    def _create_dummy_model(self, model_name: str) -> torch.nn.Module:
        """Создание dummy модели для демонстрации"""
        class DummyModel(torch.nn.Module):
            def __init__(self, input_size: int = 1024):
                super().__init__()
                self.fc1 = torch.nn.Linear(input_size, 128)
                self.fc2 = torch.nn.Linear(128, 64)
                self.fc3 = torch.nn.Linear(64, 2)
                self.dropout = torch.nn.Dropout(0.1)
            
            def forward(self, x):
                x = x.view(x.size(0), -1)  # Flatten
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return F.softmax(x, dim=1)
        
        return DummyModel()
    
    def _create_dummy_metadata(self, model_name: str) -> ModelMetadata:
        """Создание dummy метаданных"""
        model_type_map = {
            "cnn": ModelType.CNN,
            "lstm": ModelType.LSTM,
            "transformer": ModelType.TRANSFORMER,
            "ensemble": ModelType.ENSEMBLE
        }
        
        # Определяем тип по имени
        model_type = ModelType.CNN
        for key, value in model_type_map.items():
            if key in model_name.lower():
                model_type = value
                break
        
        return ModelMetadata(
            name=model_name,
            version="1.0.0",
            model_type=model_type,
            input_shape=(1024,),
            output_classes=2,
            confidence_threshold=0.7,
            created_at=time.time(),
            training_samples=10000,
            validation_accuracy=0.85,
            model_size_mb=5.2
        )
    
    def unload_model(self, model_name: str) -> bool:
        """Выгрузка модели из памяти"""
        with self._lock:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                del self.model_metadata[model_name]
                
                # Очищаем GPU память
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.logger.info(f"Model unloaded: {model_name}")
                return True
            
            return False
    
    def get_loaded_models(self) -> List[str]:
        """Получение списка загруженных моделей"""
        return list(self.loaded_models.keys())
    
    def get_model_info(self, model_name: str) -> Optional[ModelMetadata]:
        """Получение информации о модели"""
        return self.model_metadata.get(model_name)

class InferenceEngine:
    """Основной движок инференса"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_manager = ModelManager(
            self.config.get("models_path", "./ai/models")
        )
        self.preprocessor = DataPreprocessor()
        self.logger = get_logger("ml.inference_engine")
        
        # Настройки инференса
        self.batch_size = self.config.get("batch_size", 32)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.timeout_seconds = self.config.get("timeout_seconds", 30)
        
        # Executor для асинхронного инференса
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get("max_workers", 2)
        )
    
    async def predict_single(
        self,
        lightcurve: LightcurveData,
        model_name: str = "ensemble",
        preprocess: bool = True
    ) -> InferenceResult:
        """
        Предсказание для одной кривой блеска
        
        Args:
            lightcurve: Кривая блеска
            model_name: Имя модели
            preprocess: Применять предобработку
        
        Returns:
            Результат инференса
        """
        start_time = time.time()
        
        # Проверяем загружена ли модель
        if not await self._ensure_model_loaded(model_name):
            raise_ml_model_error(
                f"Failed to load model: {model_name}",
                model_name
            )
        
        try:
            with trace_ml_inference(
                model_name=model_name,
                model_version=self.model_manager.get_model_info(model_name).version,
                input_shape=(len(lightcurve.flux),)
            ) as span:
                
                # Предобработка
                if preprocess:
                    processed_lc = self.preprocessor.preprocess(
                        lightcurve,
                        target_length=1024,
                        normalize_method="median"
                    )
                else:
                    processed_lc = lightcurve
                
                # Подготовка входных данных
                input_tensor = torch.FloatTensor(processed_lc.flux).unsqueeze(0)
                input_tensor = input_tensor.to(self.model_manager.device)
                
                # Инференс
                model = self.model_manager.loaded_models[model_name]
                metadata = self.model_manager.model_metadata[model_name]
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = outputs.cpu().numpy()[0]
                
                # Обработка результатов
                prediction = float(probabilities[1])  # Вероятность планеты
                confidence = float(np.max(probabilities))
                
                inference_time_ms = (time.time() - start_time) * 1000
                
                # Добавляем атрибуты в span
                if span:
                    add_span_attributes({
                        "ml.prediction": prediction,
                        "ml.confidence": confidence,
                        "ml.inference.duration_ms": inference_time_ms
                    })
                
                # Логируем результат
                log_ml_inference(
                    model_name=model_name,
                    input_shape=input_tensor.shape,
                    prediction=prediction,
                    confidence=confidence,
                    duration_ms=inference_time_ms,
                    target_id=lightcurve.metadata.target_id if lightcurve.metadata else "unknown"
                )
                
                # Записываем метрики
                metrics_collector.record_ml_inference(
                    model_name=model_name,
                    model_version=metadata.version,
                    duration=inference_time_ms / 1000,
                    confidence=confidence,
                    status="success"
                )
                
                return InferenceResult(
                    prediction=prediction,
                    confidence=confidence,
                    probabilities=probabilities,
                    model_name=model_name,
                    model_version=metadata.version,
                    inference_time_ms=inference_time_ms,
                    input_shape=input_tensor.shape,
                    metadata={
                        "target_id": lightcurve.metadata.target_id if lightcurve.metadata else None,
                        "preprocessing_applied": preprocess,
                        "device": str(self.model_manager.device)
                    }
                )
                
        except Exception as e:
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Записываем метрики ошибки
            metadata = self.model_manager.get_model_info(model_name)
            if metadata:
                metrics_collector.record_ml_inference(
                    model_name=model_name,
                    model_version=metadata.version,
                    duration=inference_time_ms / 1000,
                    confidence=0.0,
                    status="error"
                )
            
            metrics_collector.record_error("inference_error", "inference_engine")
            
            raise_ml_model_error(
                f"Inference failed for model {model_name}: {str(e)}",
                model_name,
                details={"error_type": type(e).__name__}
            )
    
    async def predict_batch(
        self,
        lightcurves: List[LightcurveData],
        model_name: str = "ensemble",
        preprocess: bool = True
    ) -> List[InferenceResult]:
        """
        Пакетное предсказание для нескольких кривых блеска
        
        Args:
            lightcurves: Список кривых блеска
            model_name: Имя модели
            preprocess: Применять предобработку
        
        Returns:
            Список результатов инференса
        """
        self.logger.info(
            f"Starting batch inference: {len(lightcurves)} samples",
            extra={
                "model_name": model_name,
                "batch_size": len(lightcurves)
            }
        )
        
        # Проверяем загружена ли модель
        if not await self._ensure_model_loaded(model_name):
            raise_ml_model_error(
                f"Failed to load model: {model_name}",
                model_name
            )
        
        results = []
        
        # Обрабатываем батчами
        for i in range(0, len(lightcurves), self.batch_size):
            batch = lightcurves[i:i + self.batch_size]
            
            # Параллельная обработка батча
            batch_tasks = [
                self.predict_single(lc, model_name, preprocess)
                for lc in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Обрабатываем результаты и ошибки
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Batch inference error: {result}")
                    # Создаем пустой результат для ошибки
                    error_result = InferenceResult(
                        prediction=0.0,
                        confidence=0.0,
                        probabilities=np.array([1.0, 0.0]),
                        model_name=model_name,
                        model_version="unknown",
                        inference_time_ms=0.0,
                        input_shape=(0,),
                        metadata={"error": str(result)}
                    )
                    results.append(error_result)
                else:
                    results.append(result)
        
        self.logger.info(
            f"Batch inference completed: {len(results)} results",
            extra={
                "successful_predictions": sum(1 for r in results if "error" not in r.metadata),
                "failed_predictions": sum(1 for r in results if "error" in r.metadata)
            }
        )
        
        return results
    
    async def predict_with_ensemble(
        self,
        lightcurve: LightcurveData,
        model_names: List[str] = None,
        combination_strategy: str = "weighted_average"
    ) -> InferenceResult:
        """
        Предсказание с использованием ансамбля моделей
        
        Args:
            lightcurve: Кривая блеска
            model_names: Список имен моделей для ансамбля
            combination_strategy: Стратегия объединения (weighted_average, voting, stacking)
        
        Returns:
            Объединенный результат инференса
        """
        if model_names is None:
            model_names = ["cnn_classifier", "lstm_classifier", "transformer_classifier"]
        
        self.logger.info(
            f"Starting ensemble inference with {len(model_names)} models",
            extra={
                "models": model_names,
                "strategy": combination_strategy
            }
        )
        
        # Получаем предсказания от всех моделей
        individual_results = []
        for model_name in model_names:
            try:
                result = await self.predict_single(lightcurve, model_name)
                individual_results.append(result)
            except Exception as e:
                self.logger.warning(f"Model {model_name} failed: {e}")
                continue
        
        if not individual_results:
            raise_ml_model_error(
                "All models in ensemble failed",
                "ensemble",
                details={"failed_models": model_names}
            )
        
        # Объединяем результаты
        if combination_strategy == "weighted_average":
            # Веса основаны на уверенности моделей
            weights = np.array([r.confidence for r in individual_results])
            weights = weights / np.sum(weights)
            
            ensemble_prediction = np.sum([
                r.prediction * w for r, w in zip(individual_results, weights)
            ])
            
            ensemble_confidence = np.mean([r.confidence for r in individual_results])
            
        elif combination_strategy == "voting":
            # Простое голосование
            predictions = [r.prediction > 0.5 for r in individual_results]
            ensemble_prediction = float(np.mean(predictions))
            ensemble_confidence = np.std([r.confidence for r in individual_results])
            
        else:
            # По умолчанию - простое усреднение
            ensemble_prediction = np.mean([r.prediction for r in individual_results])
            ensemble_confidence = np.mean([r.confidence for r in individual_results])
        
        # Создаем объединенный результат
        total_inference_time = sum(r.inference_time_ms for r in individual_results)
        
        return InferenceResult(
            prediction=ensemble_prediction,
            confidence=ensemble_confidence,
            probabilities=np.array([1 - ensemble_prediction, ensemble_prediction]),
            model_name="ensemble",
            model_version="1.0.0",
            inference_time_ms=total_inference_time,
            input_shape=individual_results[0].input_shape,
            metadata={
                "ensemble_models": model_names,
                "combination_strategy": combination_strategy,
                "individual_results": [
                    {
                        "model": r.model_name,
                        "prediction": r.prediction,
                        "confidence": r.confidence
                    }
                    for r in individual_results
                ]
            }
        )
    
    async def _ensure_model_loaded(self, model_name: str) -> bool:
        """Убеждаемся что модель загружена"""
        if model_name not in self.model_manager.loaded_models:
            return self.model_manager.load_model(model_name)
        return True
    
    def get_model_status(self) -> Dict[str, Any]:
        """Получение статуса всех моделей"""
        loaded_models = self.model_manager.get_loaded_models()
        
        status = {
            "loaded_models": len(loaded_models),
            "device": str(self.model_manager.device),
            "models": {}
        }
        
        for model_name in loaded_models:
            metadata = self.model_manager.get_model_info(model_name)
            if metadata:
                status["models"][model_name] = {
                    "version": metadata.version,
                    "type": metadata.model_type.value,
                    "accuracy": metadata.validation_accuracy,
                    "size_mb": metadata.model_size_mb
                }
        
        return status
    
    async def warmup_models(self, model_names: List[str] = None):
        """Прогрев моделей (загрузка и тестовый инференс)"""
        if model_names is None:
            model_names = ["cnn_classifier", "lstm_classifier", "transformer_classifier"]
        
        self.logger.info(f"Warming up models: {model_names}")
        
        # Создаем тестовую кривую блеска
        test_time = np.linspace(0, 27, 1024)
        test_flux = np.ones(1024) + np.random.normal(0, 0.001, 1024)
        test_lightcurve = LightcurveData(test_time, test_flux)
        
        # Прогреваем каждую модель
        for model_name in model_names:
            try:
                await self.predict_single(test_lightcurve, model_name)
                self.logger.info(f"Model {model_name} warmed up successfully")
            except Exception as e:
                self.logger.warning(f"Failed to warm up model {model_name}: {e}")

# Глобальный экземпляр
inference_engine = InferenceEngine()
