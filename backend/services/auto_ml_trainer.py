"""
Автоматический тренер ML модели
Фоновая служба для автоматического обновления ML модели при поступлении новых данных
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path
import json

from core.logging import get_logger
from ml.lightcurve_preprocessor import LightCurvePreprocessor
from ml.feature_extractor import ExoplanetFeatureExtractor
from ml.exoplanet_classifier import ExoplanetEnsembleClassifier, create_synthetic_training_data
from services.nasa import NASADataService
from core.cache import get_cache_manager

logger = get_logger(__name__)

class AutoMLTrainer:
    """
    Автоматический тренер ML модели
    
    Функции:
    - Мониторинг новых данных из NASA/MAST
    - Автоматическое переобучение модели
    - Валидация качества модели
    - Откат к предыдущей версии при деградации
    """
    
    def __init__(self):
        self.preprocessor = LightCurvePreprocessor()
        self.feature_extractor = ExoplanetFeatureExtractor()
        self.classifier = ExoplanetEnsembleClassifier()
        self.nasa_service = NASADataService()
        self.cache_manager = get_cache_manager()
        
        # Настройки автообучения
        self.training_interval_hours = 24  # Переобучение каждые 24 часа
        self.min_new_samples = 10  # Минимум новых образцов для переобучения
        self.quality_threshold = 0.85  # Минимальная точность модели
        
        # Состояние
        self.last_training_time = None
        self.model_version = 1
        self.training_history = []
        self.is_training = False
        
        # Пути для сохранения
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    async def start_auto_training_loop(self):
        """Запуск основного цикла автообучения"""
        logger.info("Starting auto ML training loop")
        
        while True:
            try:
                await self.check_and_retrain()
                # Ждем до следующей проверки
                await asyncio.sleep(self.training_interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Error in auto training loop: {e}")
                # При ошибке ждем меньше времени
                await asyncio.sleep(3600)  # 1 час
    
    async def check_and_retrain(self):
        """Проверка необходимости переобучения и запуск при необходимости"""
        if self.is_training:
            logger.info("Training already in progress, skipping")
            return
            
        try:
            # Проверяем, нужно ли переобучение
            should_retrain = await self.should_retrain()
            
            if should_retrain:
                logger.info("Starting automatic model retraining")
                await self.retrain_model()
            else:
                logger.info("No retraining needed")
                
        except Exception as e:
            logger.error(f"Error in check_and_retrain: {e}")
    
    async def should_retrain(self) -> bool:
        """Определяет, нужно ли переобучение модели"""
        
        # Проверка времени последнего обучения
        if self.last_training_time is None:
            return True
            
        time_since_training = datetime.now() - self.last_training_time
        if time_since_training > timedelta(hours=self.training_interval_hours):
            logger.info(f"Time for scheduled retraining: {time_since_training}")
            return True
        
        # Проверка новых данных
        new_samples_count = await self.count_new_samples()
        if new_samples_count >= self.min_new_samples:
            logger.info(f"Found {new_samples_count} new samples, triggering retraining")
            return True
            
        # Проверка качества текущей модели
        current_quality = await self.evaluate_current_model()
        if current_quality < self.quality_threshold:
            logger.info(f"Model quality degraded to {current_quality:.3f}, retraining needed")
            return True
            
        return False
    
    async def count_new_samples(self) -> int:
        """Подсчитывает количество новых образцов для обучения"""
        try:
            # Получаем список известных объектов
            known_targets = await self.get_known_targets()
            
            # Проверяем новые TOI/TIC объекты
            new_targets = await self.nasa_service.get_recent_discoveries(
                days_back=7  # Последние 7 дней
            )
            
            # Фильтруем уже известные
            truly_new = [t for t in new_targets if t not in known_targets]
            
            return len(truly_new)
            
        except Exception as e:
            logger.error(f"Error counting new samples: {e}")
            return 0
    
    async def get_known_targets(self) -> List[str]:
        """Получает список уже известных целей"""
        try:
            cache_key = "ml_training_known_targets"
            cached = await self.cache_manager.get(cache_key)
            
            if cached:
                return json.loads(cached)
            else:
                # Возвращаем базовый список известных объектов
                return [
                    "TOI-715", "TIC-441420236", "Kepler-452b", "KOI-123",
                    "TOI-849", "TOI-1338", "TOI-2109", "TOI-1899"
                ]
                
        except Exception as e:
            logger.error(f"Error getting known targets: {e}")
            return []
    
    async def evaluate_current_model(self) -> float:
        """Оценивает качество текущей модели"""
        try:
            # Генерируем тестовые данные
            test_data = await self.generate_test_dataset(n_samples=100)
            
            if not test_data:
                return 1.0  # Если нет тестовых данных, считаем модель хорошей
            
            # Тестируем модель
            correct_predictions = 0
            total_predictions = len(test_data)
            
            for sample in test_data:
                try:
                    features = sample['features']
                    true_label = sample['label']
                    
                    prediction = self.classifier.predict_single(features)
                    predicted_label = prediction['predicted_class']
                    
                    if predicted_label == true_label:
                        correct_predictions += 1
                        
                except Exception as e:
                    logger.warning(f"Error evaluating sample: {e}")
                    continue
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            logger.info(f"Current model accuracy: {accuracy:.3f}")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error evaluating current model: {e}")
            return 1.0  # При ошибке считаем модель хорошей
    
    async def generate_test_dataset(self, n_samples: int = 100) -> List[Dict]:
        """Генерирует тестовый набор данных"""
        try:
            test_data = []
            
            # Генерируем синтетические данные для тестирования
            synthetic_data = create_synthetic_training_data(n_samples)
            
            for i, (time, flux, flux_err, label) in enumerate(synthetic_data):
                try:
                    # Предобработка
                    processed = self.preprocessor.preprocess_lightcurve(time, flux, flux_err)
                    
                    # Извлечение признаков
                    features = self.feature_extractor.extract_features(
                        processed['time'], processed['flux'], processed['flux_err']
                    )
                    
                    test_data.append({
                        'features': features,
                        'label': label,
                        'sample_id': f"test_{i}"
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing test sample {i}: {e}")
                    continue
            
            return test_data
            
        except Exception as e:
            logger.error(f"Error generating test dataset: {e}")
            return []
    
    async def retrain_model(self):
        """Выполняет переобучение модели"""
        self.is_training = True
        training_start = datetime.now()
        
        try:
            logger.info("Starting model retraining")
            
            # 1. Сохраняем текущую модель как бэкап
            await self.backup_current_model()
            
            # 2. Собираем новые данные
            training_data = await self.collect_training_data()
            
            if len(training_data) < 50:  # Минимум данных для обучения
                logger.warning(f"Insufficient training data: {len(training_data)} samples")
                return
            
            # 3. Обучаем новую модель
            new_classifier = ExoplanetEnsembleClassifier()
            await self.train_classifier(new_classifier, training_data)
            
            # 4. Валидируем новую модель
            new_model_quality = await self.validate_model(new_classifier, training_data)
            
            if new_model_quality >= self.quality_threshold:
                # 5. Заменяем текущую модель
                self.classifier = new_classifier
                self.model_version += 1
                self.last_training_time = training_start
                
                # 6. Сохраняем новую модель
                await self.save_model()
                
                logger.info(f"Model retrained successfully. Version: {self.model_version}, Quality: {new_model_quality:.3f}")
                
                # 7. Записываем в историю
                self.training_history.append({
                    'timestamp': training_start.isoformat(),
                    'version': self.model_version,
                    'quality': new_model_quality,
                    'training_samples': len(training_data),
                    'status': 'success'
                })
                
            else:
                logger.warning(f"New model quality too low: {new_model_quality:.3f}, keeping old model")
                
                # Записываем неудачную попытку
                self.training_history.append({
                    'timestamp': training_start.isoformat(),
                    'version': self.model_version,
                    'quality': new_model_quality,
                    'training_samples': len(training_data),
                    'status': 'failed_quality_check'
                })
                
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            
            # Записываем ошибку
            self.training_history.append({
                'timestamp': training_start.isoformat(),
                'version': self.model_version,
                'error': str(e),
                'status': 'error'
            })
            
        finally:
            self.is_training = False
            training_duration = (datetime.now() - training_start).total_seconds()
            logger.info(f"Training completed in {training_duration:.1f} seconds")
    
    async def collect_training_data(self) -> List[Dict]:
        """Собирает данные для обучения"""
        try:
            training_data = []
            
            # 1. Получаем новые реальные данные
            real_data = await self.collect_real_data()
            training_data.extend(real_data)
            
            # 2. Дополняем синтетическими данными
            synthetic_data = await self.collect_synthetic_data(target_count=500)
            training_data.extend(synthetic_data)
            
            logger.info(f"Collected {len(training_data)} training samples")
            return training_data
            
        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            return []
    
    async def collect_real_data(self) -> List[Dict]:
        """Собирает реальные данные для обучения"""
        real_data = []
        
        try:
            # Список известных объектов с метками
            known_objects = [
                ("TOI-715", "Confirmed"),
                ("TIC-441420236", "Confirmed"),
                ("Kepler-452b", "Confirmed"),
                ("TOI-849", "Candidate"),
                ("TOI-1338", "Confirmed"),
                ("False-Target-1", "False Positive"),
                ("Noise-Target-1", "False Positive")
            ]
            
            for target_name, label in known_objects:
                try:
                    # Загружаем данные кривой блеска
                    lightcurve_data = await self.nasa_service.get_lightcurve(target_name)
                    
                    if lightcurve_data and len(lightcurve_data.get('time', [])) > 100:
                        # Обрабатываем данные
                        time = np.array(lightcurve_data['time'])
                        flux = np.array(lightcurve_data['flux'])
                        flux_err = np.array(lightcurve_data.get('flux_err', np.ones_like(flux) * 0.001))
                        
                        # Предобработка
                        processed = self.preprocessor.preprocess_lightcurve(time, flux, flux_err)
                        
                        # Извлечение признаков
                        features = self.feature_extractor.extract_features(
                            processed['time'], processed['flux'], processed['flux_err']
                        )
                        
                        real_data.append({
                            'features': features,
                            'label': label,
                            'target_name': target_name,
                            'data_source': 'real'
                        })
                        
                except Exception as e:
                    logger.warning(f"Error processing real data for {target_name}: {e}")
                    continue
            
            logger.info(f"Collected {len(real_data)} real training samples")
            return real_data
            
        except Exception as e:
            logger.error(f"Error collecting real data: {e}")
            return []
    
    async def collect_synthetic_data(self, target_count: int = 500) -> List[Dict]:
        """Собирает синтетические данные для обучения"""
        synthetic_data = []
        
        try:
            # Генерируем синтетические данные
            raw_synthetic = create_synthetic_training_data(target_count)
            
            for i, (time, flux, flux_err, label) in enumerate(raw_synthetic):
                try:
                    # Предобработка
                    processed = self.preprocessor.preprocess_lightcurve(time, flux, flux_err)
                    
                    # Извлечение признаков
                    features = self.feature_extractor.extract_features(
                        processed['time'], processed['flux'], processed['flux_err']
                    )
                    
                    synthetic_data.append({
                        'features': features,
                        'label': label,
                        'target_name': f"synthetic_{i}",
                        'data_source': 'synthetic'
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing synthetic sample {i}: {e}")
                    continue
            
            logger.info(f"Collected {len(synthetic_data)} synthetic training samples")
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error collecting synthetic data: {e}")
            return []
    
    async def train_classifier(self, classifier: ExoplanetEnsembleClassifier, training_data: List[Dict]):
        """Обучает классификатор на данных"""
        try:
            # Подготавливаем данные для обучения
            X = []
            y = []
            
            for sample in training_data:
                X.append(sample['features'])
                y.append(sample['label'])
            
            X = np.array(X)
            y = np.array(y)
            
            # Обучаем модель
            classifier.train(X, y)
            
            logger.info(f"Classifier trained on {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training classifier: {e}")
            raise
    
    async def validate_model(self, classifier: ExoplanetEnsembleClassifier, training_data: List[Dict]) -> float:
        """Валидирует качество модели"""
        try:
            # Используем часть данных для валидации
            validation_size = min(100, len(training_data) // 5)
            validation_data = training_data[-validation_size:]
            
            correct = 0
            total = len(validation_data)
            
            for sample in validation_data:
                prediction = classifier.predict_single(sample['features'])
                if prediction['predicted_class'] == sample['label']:
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0.0
            logger.info(f"Model validation accuracy: {accuracy:.3f}")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return 0.0
    
    async def backup_current_model(self):
        """Создает бэкап текущей модели"""
        try:
            backup_path = self.models_dir / f"model_backup_v{self.model_version}.pkl"
            self.classifier.save_model(str(backup_path))
            logger.info(f"Model backed up to {backup_path}")
            
        except Exception as e:
            logger.error(f"Error backing up model: {e}")
    
    async def save_model(self):
        """Сохраняет текущую модель"""
        try:
            model_path = self.models_dir / f"model_v{self.model_version}.pkl"
            self.classifier.save_model(str(model_path))
            
            # Сохраняем метаданные
            metadata = {
                'version': self.model_version,
                'training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'training_history': self.training_history[-10:]  # Последние 10 записей
            }
            
            metadata_path = self.models_dir / f"model_v{self.model_version}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved: {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def get_training_status(self) -> Dict:
        """Возвращает статус обучения"""
        return {
            'is_training': self.is_training,
            'model_version': self.model_version,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_history': self.training_history[-5:],  # Последние 5 записей
            'next_training_in_hours': self.training_interval_hours if self.last_training_time else 0
        }

# Глобальный экземпляр
_auto_trainer = None

def get_auto_trainer() -> AutoMLTrainer:
    """Получает глобальный экземпляр автотренера"""
    global _auto_trainer
    if _auto_trainer is None:
        _auto_trainer = AutoMLTrainer()
    return _auto_trainer
