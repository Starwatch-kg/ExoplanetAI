"""
Enhanced Auto ML Trainer для ExoplanetAI
Улучшенная система автоматического обучения с реальными данными NASA + fallback на синтетику
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json
import lightkurve as lk
from astroquery.mast import Catalogs

from core.logging import get_logger
from ml.lightcurve_preprocessor import LightCurvePreprocessor
from ml.feature_extractor import ExoplanetFeatureExtractor
from ml.exoplanet_classifier import ExoplanetEnsembleClassifier, create_synthetic_training_data
from services.nasa import NASADataService
from core.cache import get_cache_manager

logger = get_logger(__name__)

class EnhancedAutoMLTrainer:
    """
    Улучшенная система автоматического обучения ИИ
    
    Особенности:
    - Приоритет реальным данным NASA/TESS/Kepler
    - Fallback на синтетические данные при недостатке реальных
    - Адаптивное обучение на основе качества данных
    - Мониторинг производительности модели
    """
    
    def __init__(self):
        self.preprocessor = LightCurvePreprocessor()
        self.feature_extractor = ExoplanetFeatureExtractor()
        self.classifier = ExoplanetEnsembleClassifier()
        self.nasa_service = NASADataService()
        self.cache_manager = get_cache_manager()
        
        # Настройки
        self.training_interval_hours = 12  # Чаще проверяем
        self.min_real_samples = 20  # Минимум реальных данных
        self.synthetic_ratio = 0.7  # 70% синтетики, 30% реальных данных
        self.quality_threshold = 0.80
        
        # Состояние
        self.last_training_time = None
        self.model_version = 1
        self.training_history = []
        self.is_training = False
        self.real_data_cache = {}
        
        # Пути
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    async def start_enhanced_training_loop(self):
        """Запуск улучшенного цикла автообучения"""
        logger.info("Starting enhanced auto ML training loop")
        
        while True:
            try:
                await self.intelligent_training_check()
                await asyncio.sleep(self.training_interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Error in enhanced training loop: {e}")
                await asyncio.sleep(1800)  # 30 минут при ошибке
    
    async def intelligent_training_check(self):
        """Интеллектуальная проверка необходимости обучения"""
        if self.is_training:
            return
            
        try:
            # Собираем метрики для принятия решения
            metrics = await self.collect_training_metrics()
            
            should_train = self.decide_training_necessity(metrics)
            
            if should_train:
                logger.info(f"Training triggered by metrics: {metrics}")
                await self.enhanced_model_training()
            else:
                logger.info("No training needed based on current metrics")
                
        except Exception as e:
            logger.error(f"Error in intelligent training check: {e}")
    
    async def collect_training_metrics(self) -> Dict:
        """Собирает метрики для принятия решения об обучении"""
        metrics = {
            'time_since_last_training': 0,
            'new_real_data_count': 0,
            'model_performance_score': 1.0,
            'data_quality_score': 0.8,
            'cache_hit_rate': 0.9
        }
        
        try:
            # Время с последнего обучения
            if self.last_training_time:
                time_diff = datetime.now() - self.last_training_time
                metrics['time_since_last_training'] = time_diff.total_seconds() / 3600
            
            # Новые реальные данные
            metrics['new_real_data_count'] = await self.count_new_real_data()
            
            # Производительность модели
            metrics['model_performance_score'] = await self.evaluate_model_performance()
            
            # Качество данных в кэше
            metrics['data_quality_score'] = await self.assess_data_quality()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting training metrics: {e}")
            return metrics
    
    def decide_training_necessity(self, metrics: Dict) -> bool:
        """Принимает решение о необходимости обучения на основе метрик"""
        
        # Обязательное обучение по времени
        if metrics['time_since_last_training'] > self.training_interval_hours:
            return True
        
        # Обучение при наличии новых данных
        if metrics['new_real_data_count'] >= self.min_real_samples:
            return True
        
        # Обучение при деградации модели
        if metrics['model_performance_score'] < self.quality_threshold:
            return True
        
        # Комбинированная оценка
        combined_score = (
            metrics['new_real_data_count'] * 0.4 +
            (1 - metrics['model_performance_score']) * 0.3 +
            metrics['time_since_last_training'] / 24 * 0.3
        )
        
        return combined_score > 0.5
    
    async def enhanced_model_training(self):
        """Улучшенное обучение модели с реальными данными"""
        self.is_training = True
        training_start = datetime.now()
        
        try:
            logger.info("Starting enhanced model training")
            
            # 1. Сбор реальных данных NASA
            real_data = await self.collect_nasa_real_data()
            logger.info(f"Collected {len(real_data)} real samples")
            
            # 2. Генерация синтетических данных для дополнения
            synthetic_needed = max(100, len(real_data) * 3)  # 3:1 соотношение
            synthetic_data = await self.generate_adaptive_synthetic_data(
                real_data, synthetic_needed
            )
            logger.info(f"Generated {len(synthetic_data)} synthetic samples")
            
            # 3. Объединение и балансировка данных
            training_data = await self.balance_training_data(real_data, synthetic_data)
            
            if len(training_data) < 50:
                logger.warning(f"Insufficient training data: {len(training_data)}")
                return
            
            # 4. Обучение с адаптивными параметрами
            new_classifier = ExoplanetEnsembleClassifier()
            training_metrics = await self.adaptive_model_training(
                new_classifier, training_data
            )
            
            # 5. Валидация и сравнение с текущей моделью
            validation_score = await self.comprehensive_model_validation(
                new_classifier, training_data
            )
            
            if validation_score >= self.quality_threshold:
                # Обновляем модель
                await self.update_model(new_classifier, training_metrics, validation_score)
                logger.info(f"Model updated successfully. Score: {validation_score:.3f}")
            else:
                logger.warning(f"New model quality insufficient: {validation_score:.3f}")
                
        except Exception as e:
            logger.error(f"Error in enhanced training: {e}")
            self.training_history.append({
                'timestamp': training_start.isoformat(),
                'status': 'error',
                'error': str(e)
            })
            
        finally:
            self.is_training = False
            duration = (datetime.now() - training_start).total_seconds()
            logger.info(f"Enhanced training completed in {duration:.1f}s")
    
    async def collect_nasa_real_data(self) -> List[Dict]:
        """Собирает реальные данные из NASA архивов"""
        real_data = []
        
        # Список подтвержденных экзопланет для обучения
        confirmed_targets = [
            ("TOI-715", "Confirmed"),
            ("TIC-441420236", "Confirmed"), 
            ("Kepler-452b", "Confirmed"),
            ("TOI-849", "Confirmed"),
            ("TOI-1338", "Confirmed"),
            ("K2-18b", "Confirmed"),
            ("WASP-121b", "Confirmed"),
            ("HD 209458b", "Confirmed")
        ]
        
        for target_name, label in confirmed_targets:
            try:
                # Попытка загрузки через lightkurve
                lightcurve_data = await self.fetch_lightkurve_data(target_name)
                
                if lightcurve_data:
                    processed_sample = await self.process_real_sample(
                        target_name, lightcurve_data, label
                    )
                    if processed_sample:
                        real_data.append(processed_sample)
                        
            except Exception as e:
                logger.warning(f"Failed to collect data for {target_name}: {e}")
                continue
        
        # Добавляем известные False Positive примеры
        false_positives = await self.generate_false_positive_samples()
        real_data.extend(false_positives)
        
        return real_data
    
    async def fetch_lightkurve_data(self, target_name: str) -> Optional[Dict]:
        """Загружает данные через lightkurve"""
        try:
            # Поиск в TESS
            search_result = lk.search_lightcurve(target_name, mission='TESS')
            
            if len(search_result) > 0:
                lc = search_result[0].download()
                
                # Очистка и нормализация
                lc = lc.remove_nans().remove_outliers()
                
                return {
                    'time': lc.time.value,
                    'flux': lc.flux.value,
                    'flux_err': lc.flux_err.value if lc.flux_err is not None else None,
                    'mission': 'TESS',
                    'quality': lc.quality.value if hasattr(lc, 'quality') else None
                }
            
            # Fallback к Kepler
            search_result = lk.search_lightcurve(target_name, mission='Kepler')
            if len(search_result) > 0:
                lc = search_result[0].download()
                lc = lc.remove_nans().remove_outliers()
                
                return {
                    'time': lc.time.value,
                    'flux': lc.flux.value, 
                    'flux_err': lc.flux_err.value if lc.flux_err is not None else None,
                    'mission': 'Kepler',
                    'quality': lc.quality.value if hasattr(lc, 'quality') else None
                }
                
        except Exception as e:
            logger.warning(f"Lightkurve fetch failed for {target_name}: {e}")
            
        return None
    
    async def process_real_sample(self, target_name: str, lightcurve_data: Dict, label: str) -> Optional[Dict]:
        """Обрабатывает реальный образец данных"""
        try:
            time = np.array(lightcurve_data['time'])
            flux = np.array(lightcurve_data['flux'])
            flux_err = np.array(lightcurve_data.get('flux_err', np.ones_like(flux) * 0.001))
            
            # Предобработка
            processed = self.preprocessor.preprocess_lightcurve(time, flux, flux_err)
            
            # Извлечение признаков
            features = self.feature_extractor.extract_features(
                processed['time'], processed['flux'], processed['flux_err']
            )
            
            return {
                'features': features,
                'label': label,
                'target_name': target_name,
                'data_source': 'real_nasa',
                'mission': lightcurve_data.get('mission', 'unknown'),
                'data_quality': processed.get('data_quality', 0.8)
            }
            
        except Exception as e:
            logger.error(f"Error processing real sample {target_name}: {e}")
            return None
    
    async def generate_adaptive_synthetic_data(self, real_data: List[Dict], count: int) -> List[Dict]:
        """Генерирует синтетические данные, адаптированные к реальным"""
        synthetic_data = []
        
        try:
            # Анализируем характеристики реальных данных
            real_characteristics = self.analyze_real_data_characteristics(real_data)
            
            # Генерируем синтетику с похожими характеристиками
            raw_synthetic = create_synthetic_training_data(count)
            
            for i, (time, flux, flux_err, label) in enumerate(raw_synthetic):
                try:
                    # Адаптируем под реальные характеристики
                    adapted_flux = self.adapt_synthetic_to_real(
                        flux, real_characteristics
                    )
                    
                    # Обработка
                    processed = self.preprocessor.preprocess_lightcurve(
                        time, adapted_flux, flux_err
                    )
                    
                    features = self.feature_extractor.extract_features(
                        processed['time'], processed['flux'], processed['flux_err']
                    )
                    
                    synthetic_data.append({
                        'features': features,
                        'label': label,
                        'target_name': f"synthetic_adapted_{i}",
                        'data_source': 'synthetic_adapted',
                        'data_quality': processed.get('data_quality', 0.7)
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing synthetic sample {i}: {e}")
                    continue
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error generating adaptive synthetic data: {e}")
            return []
    
    def analyze_real_data_characteristics(self, real_data: List[Dict]) -> Dict:
        """Анализирует характеристики реальных данных"""
        if not real_data:
            return {'noise_level': 0.001, 'systematic_trends': 0.1}
        
        # Извлекаем статистики из реальных данных
        noise_levels = []
        trend_levels = []
        
        for sample in real_data:
            if 'data_quality' in sample:
                noise_levels.append(1.0 - sample['data_quality'])
            
        return {
            'noise_level': np.mean(noise_levels) if noise_levels else 0.001,
            'systematic_trends': np.std(noise_levels) if noise_levels else 0.1
        }
    
    def adapt_synthetic_to_real(self, synthetic_flux: np.ndarray, characteristics: Dict) -> np.ndarray:
        """Адаптирует синтетические данные под характеристики реальных"""
        adapted_flux = synthetic_flux.copy()
        
        # Добавляем реалистичный шум
        noise_level = characteristics.get('noise_level', 0.001)
        noise = np.random.normal(0, noise_level, len(adapted_flux))
        adapted_flux += noise
        
        # Добавляем систематические тренды
        trend_level = characteristics.get('systematic_trends', 0.1)
        trend = np.linspace(0, trend_level, len(adapted_flux))
        trend += np.random.normal(0, trend_level * 0.1, len(adapted_flux))
        adapted_flux += trend
        
        return adapted_flux
    
    async def get_training_status(self) -> Dict:
        """Возвращает расширенный статус обучения"""
        return {
            'is_training': self.is_training,
            'model_version': self.model_version,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_history': self.training_history[-5:],
            'real_data_cache_size': len(self.real_data_cache),
            'next_check_in_hours': self.training_interval_hours,
            'quality_threshold': self.quality_threshold,
            'min_real_samples': self.min_real_samples
        }

# Глобальный экземпляр
_enhanced_trainer = None

def get_enhanced_trainer() -> EnhancedAutoMLTrainer:
    """Получает глобальный экземпляр улучшенного тренера"""
    global _enhanced_trainer
    if _enhanced_trainer is None:
        _enhanced_trainer = EnhancedAutoMLTrainer()
    return _enhanced_trainer
