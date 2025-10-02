"""
AI Service Integration

Интеграция AI модуля с основным API для анализа кривых блеска.
Объединяет BLS алгоритм с машинным обучением для повышения точности.
"""

import asyncio
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

from ..ai.models import CNNClassifier, LSTMClassifier, TransformerClassifier
from ..ai.ensemble import EnsembleClassifier, create_default_ensemble
from ..ai.trainer import ModelTrainer, TransitDataset
from ..ai.predictor import TransitPredictor, AIAssistant
from ..ai.embeddings import EmbeddingManager
from ..ai.database import DatabaseManager, AnalysisResult, UserFeedback
from ..services.bls_service import BLSService
from ..models.search_models import LightCurveData, TransitCandidate, SearchResult

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Уровни уверенности AI"""
    VERY_LOW = "very_low"
    LOW = "low" 
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class AIAnalysisResult:
    """Результат AI анализа"""
    target_name: str
    confidence: float
    prediction: str
    explanation: str
    uncertainty: float
    processing_time: float
    model_version: str
    features_importance: Dict[str, float]
    recommendations: List[str]

@dataclass
class RealTimeMetrics:
    """Метрики AI в реальном времени"""
    predictions_count: int = 0
    avg_confidence: float = 0.0
    avg_processing_time: float = 0.0
    success_rate: float = 0.0
    model_accuracy: float = 0.0
    last_updated: datetime = None

class AdvancedAIService:
    """
    Продвинутый AI сервис с множественными моделями и интеллектуальным анализом
    """
    
    def __init__(self):
        self.models = {}
        self.ensemble_weights = {}
        self.performance_tracker = {}
        self.real_time_metrics = RealTimeMetrics()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.cache = {}
        self.adaptive_thresholds = {
            'confidence_threshold': 0.7,
            'uncertainty_threshold': 0.3,
            'ensemble_agreement': 0.8
        }
        
    async def initialize_advanced_models(self):
        """Инициализация продвинутых AI моделей"""
        try:
            # 1. Transformer для последовательностей
            self.models['transformer'] = await self._load_transformer_model()
            
            # 2. CNN для паттернов
            self.models['cnn'] = await self._load_cnn_model()
            
            # 3. LSTM для временных рядов  
            self.models['lstm'] = await self._load_lstm_model()
            
            # 4. Gradient Boosting для табличных данных
            self.models['gbm'] = await self._load_gbm_model()
            
            # 5. Автоэнкодер для аномалий
            self.models['autoencoder'] = await self._load_autoencoder_model()
            
            logger.info(f"Initialized {len(self.models)} AI models")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
    
    async def intelligent_analysis(self, lightcurve_data: Dict) -> AIAnalysisResult:
        """Интеллектуальный анализ с множественными моделями"""
        start_time = time.time()
        
        # Извлекаем признаки
        features = await self._extract_advanced_features(lightcurve_data)
        
        # Получаем предсказания от всех моделей
        predictions = {}
        confidences = {}
        
        for model_name, model in self.models.items():
            try:
                pred_result = await self._predict_with_model(model, features, model_name)
                predictions[model_name] = pred_result['prediction']
                confidences[model_name] = pred_result['confidence']
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                continue
        
        # Интеллектуальное объединение предсказаний
        ensemble_result = await self._intelligent_ensemble(predictions, confidences)
        
        # Анализ неопределенности
        uncertainty = self._calculate_prediction_uncertainty(confidences)
        
        # Генерация объяснений
        explanation = await self._generate_advanced_explanation(
            ensemble_result, features, predictions
        )
        
        # Рекомендации
        recommendations = await self._generate_smart_recommendations(
            ensemble_result, uncertainty, features
        )
        
        processing_time = time.time() - start_time
        
        # Обновляем метрики
        await self._update_real_time_metrics(ensemble_result, processing_time)
        
        return AIAnalysisResult(
            target_name=lightcurve_data.get('target_name', 'Unknown'),
            confidence=ensemble_result['confidence'],
            prediction=ensemble_result['prediction'],
            explanation=explanation,
            uncertainty=uncertainty,
            processing_time=processing_time,
            model_version="v2.0-advanced",
            features_importance=features.get('importance', {}),
            recommendations=recommendations
        )
    
    async def adaptive_learning(self, feedback_data: List[Dict]) -> Dict:
        """Адаптивное обучение на основе обратной связи"""
        try:
            # Анализ паттернов в обратной связи
            patterns = await self._analyze_feedback_patterns(feedback_data)
            
            # Адаптация порогов
            await self._adapt_thresholds(patterns)
            
            # Обновление весов ансамбля
            await self._update_ensemble_weights(patterns)
            
            # Дообучение моделей
            retrain_results = {}
            for model_name in self.models.keys():
                if patterns.get(f'{model_name}_needs_update', False):
                    result = await self._retrain_model(model_name, feedback_data)
                    retrain_results[model_name] = result
            
            return {
                'patterns_found': len(patterns),
                'thresholds_updated': True,
                'models_retrained': list(retrain_results.keys()),
                'performance_improvement': await self._calculate_improvement()
            }
            
        except Exception as e:
            logger.error(f"Adaptive learning failed: {e}")
            return {'error': str(e)}
    
    async def real_time_monitoring(self) -> Dict:
        """Мониторинг AI в реальном времени"""
        return {
            'metrics': {
                'predictions_count': self.real_time_metrics.predictions_count,
                'avg_confidence': self.real_time_metrics.avg_confidence,
                'avg_processing_time': self.real_time_metrics.avg_processing_time,
                'success_rate': self.real_time_metrics.success_rate,
                'model_accuracy': self.real_time_metrics.model_accuracy
            },
            'model_status': {
                name: await self._check_model_health(model) 
                for name, model in self.models.items()
            },
            'adaptive_thresholds': self.adaptive_thresholds,
            'cache_stats': {
                'size': len(self.cache),
                'hit_rate': await self._calculate_cache_hit_rate()
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def explain_prediction_detailed(self, target_name: str) -> Dict:
        """Детальное объяснение предсказания"""
        if target_name not in self.cache:
            return {'error': 'Prediction not found in cache'}
        
        cached_result = self.cache[target_name]
        
        # Анализ важности признаков
        feature_analysis = await self._analyze_feature_importance(cached_result)
        
        # Сравнение с похожими объектами
        similar_objects = await self._find_similar_predictions(cached_result)
        
        # Визуализация данных для объяснения
        visualization_data = await self._prepare_visualization_data(cached_result)
        
        return {
            'basic_explanation': cached_result.explanation,
            'feature_analysis': feature_analysis,
            'similar_objects': similar_objects,
            'confidence_breakdown': await self._break_down_confidence(cached_result),
            'uncertainty_sources': await self._identify_uncertainty_sources(cached_result),
            'visualization_data': visualization_data,
            'recommendations': cached_result.recommendations
        }

class AIEnhancedBLSService:
    """
    Улучшенный BLS сервис с интеграцией ИИ
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 use_ensemble: bool = True,
                 enable_database: bool = True):
        
        self.device = device
        self.use_ensemble = use_ensemble
        self.enable_database = enable_database
        
        # Инициализация компонентов
        self.bls_service = BLSService()
        self.embedding_manager = EmbeddingManager()
        self.ai_assistant = AIAssistant()
        
        # База данных
        if enable_database:
            self.db_manager = DatabaseManager()
        else:
            self.db_manager = None
        
        # Загрузка или создание модели
        if model_path and Path(model_path).exists():
            self.model = self._load_model(model_path)
        else:
            self.model = self._create_default_model()
        
        # Предиктор
        self.predictor = TransitPredictor(
            model=self.model,
            embedding_manager=self.embedding_manager,
            device=device
        )
        
        # Тренер для онлайн обучения
        self.trainer = ModelTrainer(
            model=self.model,
            device=device,
            experiment_name='ai_enhanced_bls'
        )
        
        logger.info(f"AI Enhanced BLS Service initialized on {device}")
    
    async def initialize(self):
        """Асинхронная инициализация"""
        if self.db_manager:
            await self.db_manager.initialize()
        logger.info("AI Enhanced BLS Service ready")
    
    async def enhanced_search(self,
                            lightcurve_data: LightCurveData,
                            period_min: float = 0.5,
                            period_max: float = 50.0,
                            duration_min: float = 0.01,
                            duration_max: float = 0.5,
                            snr_threshold: float = 7.0,
                            use_ai_validation: bool = True) -> SearchResult:
        """
        Улучшенный поиск с ИИ валидацией
        
        Args:
            lightcurve_data: Данные кривой блеска
            period_min: Минимальный период
            period_max: Максимальный период
            duration_min: Минимальная длительность
            duration_max: Максимальная длительность
            snr_threshold: Пороговое значение SNR
            use_ai_validation: Использовать ли ИИ для валидации
            
        Returns:
            Расширенный результат поиска с ИИ анализом
        """
        start_time = datetime.now()
        
        # 1. Выполняем стандартный BLS анализ
        logger.info(f"Starting enhanced search for {lightcurve_data.target_name}")
        
        bls_result = await self.bls_service.run_bls_analysis(
            lightcurve_data=lightcurve_data,
            period_min=period_min,
            period_max=period_max,
            duration_min=duration_min,
            duration_max=duration_max,
            snr_threshold=snr_threshold
        )
        
        # 2. ИИ анализ кривой блеска
        ai_predictions = []
        if use_ai_validation and len(lightcurve_data.flux) > 100:
            
            # Подготавливаем данные для ИИ
            flux_array = np.array(lightcurve_data.flux)
            
            # Нормализация и предобработка
            flux_normalized = self._preprocess_lightcurve(flux_array)
            
            # Получаем предсказание ИИ
            ai_prediction = self.predictor.predict(
                lightcurve=flux_normalized,
                target_name=lightcurve_data.target_name,
                stellar_params=None  # Можно добавить параметры звезды если есть
            )
            
            ai_predictions.append(ai_prediction)
            
            # Валидация BLS кандидатов с помощью ИИ
            validated_candidates = await self._validate_candidates_with_ai(
                bls_result.candidates, flux_normalized, ai_prediction
            )
            
            # Обновляем список кандидатов
            bls_result.candidates = validated_candidates
        
        # 3. Сохраняем результат в базу данных
        if self.db_manager:
            await self._save_analysis_to_db(
                lightcurve_data, bls_result, ai_predictions, start_time
            )
        
        # 4. Добавляем ИИ метаданные к результату
        enhanced_result = self._enhance_search_result(bls_result, ai_predictions)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        enhanced_result.processing_time = processing_time
        
        logger.info(f"Enhanced search completed in {processing_time:.2f}s")
        return enhanced_result
    
    async def get_ai_explanation(self, 
                               target_name: str,
                               mode: str = 'detailed') -> Dict[str, str]:
        """
        Получение ИИ объяснения результатов
        
        Args:
            target_name: Имя цели
            mode: Режим объяснения ('detailed', 'beginner', 'scientific')
            
        Returns:
            Словарь с объяснениями
        """
        # Ищем последний анализ в кэше
        cached_prediction = self.embedding_manager.get_cached_prediction(target_name)
        
        if not cached_prediction:
            return {'error': f'No analysis found for {target_name}'}
        
        explanations = {}
        
        if mode == 'beginner':
            explanations['explanation'] = self.ai_assistant.explain_for_beginners(
                cached_prediction, target_name
            )
            explanations['comparison'] = self.ai_assistant.compare_with_known_planets(
                cached_prediction
            )
            explanations['habitability'] = self.ai_assistant.explain_habitability(
                cached_prediction
            )
        
        elif mode == 'detailed':
            explanations['explanation'] = cached_prediction.explanation
            explanations['recommendations'] = cached_prediction.recommendations
            explanations['uncertainty_sources'] = cached_prediction.uncertainty_sources
            explanations['confidence_level'] = cached_prediction.confidence_level.value
        
        elif mode == 'scientific':
            explanations['technical_analysis'] = self._generate_technical_analysis(
                cached_prediction
            )
            explanations['statistical_significance'] = self._assess_statistical_significance(
                cached_prediction
            )
        
        return explanations
    
    async def submit_user_feedback(self,
                                 target_name: str,
                                 user_id: str,
                                 feedback_type: str,
                                 confidence_rating: int,
                                 comments: Optional[str] = None) -> bool:
        """
        Отправка пользовательской обратной связи для активного обучения
        
        Args:
            target_name: Имя цели
            user_id: ID пользователя
            feedback_type: Тип обратной связи ('correct', 'incorrect', 'uncertain')
            confidence_rating: Оценка уверенности (1-5)
            comments: Комментарии
            
        Returns:
            Успешность сохранения
        """
        if not self.db_manager:
            logger.warning("Database not enabled, cannot save feedback")
            return False
        
        feedback = UserFeedback(
            target_name=target_name,
            user_id=user_id,
            feedback_type=feedback_type,
            confidence_rating=confidence_rating,
            comments=comments,
            timestamp=datetime.now()
        )
        
        try:
            await self.db_manager.save_user_feedback(feedback)
            
            # Запускаем онлайн обучение если накопилось достаточно обратной связи
            await self._check_and_trigger_online_learning()
            
            logger.info(f"User feedback saved for {target_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save user feedback: {e}")
            return False
    
    async def retrain_model(self,
                          dataset_name: str = 'user_feedback',
                          epochs: int = 10) -> Dict[str, Any]:
        """
        Переобучение модели на новых данных
        
        Args:
            dataset_name: Название датасета
            epochs: Количество эпох
            
        Returns:
            Результаты обучения
        """
        if not self.db_manager:
            raise ValueError("Database required for retraining")
        
        # Получаем данные обратной связи
        feedback_data = await self.db_manager.get_feedback_for_active_learning(limit=1000)
        
        if len(feedback_data) < 10:
            return {'error': 'Insufficient feedback data for retraining'}
        
        # Подготавливаем данные для ИИ из обратной связи
        training_data = await self._prepare_training_data_from_feedback(feedback_data)
        
        if not training_data:
            return {'error': 'Failed to prepare training data from feedback'}
        
        # Создаем датасет для обучения
        dataset = TransitDataset(
            lightcurves=training_data['lightcurves'],
            labels=training_data['labels'],
            augment=True
        )
        
        # Выполняем переобучение модели
        training_result = await self.trainer.train_with_feedback(
            dataset=dataset,
            epochs=epochs,
            learning_rate=1e-4,
            batch_size=32
        )
        
        # Сохраняем информацию об обучении
        await self.db_manager.save_training_session(
            model_name=self.model.__class__.__name__,
            model_version='1.1.0',
            dataset_name=dataset_name,
            training_params={'epochs': epochs},
            metrics={'accuracy': training_result['final_accuracy']},
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        logger.info(f"Model retrained on {len(feedback_data)} feedback samples")
        return training_result
    
    async def get_similar_targets(self, 
                                target_name: str,
                                top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Поиск похожих целей на основе embeddings
        
        Args:
            target_name: Имя цели для поиска похожих
            top_k: Количество похожих целей
            
        Returns:
            Список похожих целей с метаданными
        """
        # Получаем embedding для цели
        cached_prediction = self.embedding_manager.get_cached_prediction(target_name)
        
        if not cached_prediction:
            return []
        
        # Поиск похожих целей на основе embeddings
        similar_targets = []
        
        # Получаем embedding для текущей цели
        target_embedding = cached_prediction.embedding
        
        if self.db_manager and target_embedding is not None:
            # Получаем все анализы с embeddings
            all_analyses = await self.db_manager.get_analysis_history_with_embeddings(limit=1000)
            
            # Вычисляем косинусное сходство
            similarities = []
            for analysis in all_analyses:
                if analysis['target_name'] != target_name and analysis.get('embedding') is not None:
                    similarity = self._compute_cosine_similarity(
                        target_embedding, analysis['embedding']
                    )
                    similarities.append((analysis, similarity))
            
            # Сортируем по сходству и берем топ-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            for analysis, similarity in similarities[:top_k]:
                similar_targets.append({
                    'target_name': analysis['target_name'],
                    'similarity': float(similarity),
                    'is_transit': analysis['is_transit'],
                    'confidence': analysis['confidence'],
                    'analysis_date': analysis['analysis_timestamp'].isoformat()
                })
        
        return similar_targets
    
    def _load_model(self, model_path: str):
        """Загрузка сохраненной модели"""
        try:
            if self.use_ensemble:
                # Загрузка ансамбля
                model_classes = {
                    'cnn': CNNClassifier,
                    'lstm': LSTMClassifier,
                    'transformer': TransformerClassifier
                }
                ensemble = EnsembleClassifier({}, 'weighted')
                ensemble.load_ensemble(model_path, model_classes)
                return ensemble
            else:
                # Загрузка одиночной модели
                model = CNNClassifier()
                model.load_model(model_path)
                return model
                
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")
            return self._create_default_model()
    
    def _create_default_model(self):
        """Создание модели по умолчанию"""
        if self.use_ensemble:
            return create_default_ensemble(device=self.device)
        else:
            return CNNClassifier().to(self.device)
    
    def _preprocess_lightcurve(self, flux: np.ndarray) -> np.ndarray:
        """Предобработка кривой блеска для ИИ"""
        # Нормализация
        flux_normalized = flux / np.median(flux)
        
        # Удаление выбросов
        q1, q3 = np.percentile(flux_normalized, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        mask = (flux_normalized >= lower_bound) & (flux_normalized <= upper_bound)
        flux_clean = flux_normalized.copy()
        flux_clean[~mask] = np.median(flux_normalized[mask])
        
        # Приведение к стандартному размеру
        target_length = 1024
        if len(flux_clean) > target_length:
            # Даунсэмплинг
            indices = np.linspace(0, len(flux_clean) - 1, target_length).astype(int)
            flux_clean = flux_clean[indices]
        elif len(flux_clean) < target_length:
            # Интерполяция
            try:
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, len(flux_clean))
                x_new = np.linspace(0, 1, target_length)
                f = interp1d(x_old, flux_clean, kind='linear')
                flux_clean = f(x_new)
            except ImportError:
                # Fallback: простое повторение значений
                flux_clean = np.resize(flux_clean, target_length)
        
        return flux_clean
    
    async def _validate_candidates_with_ai(self,
                                         candidates: List[TransitCandidate],
                                         flux: np.ndarray,
                                         ai_prediction) -> List[TransitCandidate]:
        """Валидация BLS кандидатов с помощью ИИ"""
        validated_candidates = []
        
        for candidate in candidates:
            # Простая валидация на основе ИИ предсказания
            if ai_prediction.is_transit and ai_prediction.confidence > 0.6:
                # ИИ подтверждает наличие транзита
                candidate.planet_probability = min(
                    candidate.planet_probability * 1.2,  # Увеличиваем вероятность
                    0.99
                )
                validated_candidates.append(candidate)
            elif ai_prediction.confidence < 0.3:
                # ИИ сомневается - снижаем вероятность
                candidate.planet_probability *= 0.8
                if candidate.planet_probability > 0.1:  # Оставляем только если вероятность не слишком мала
                    validated_candidates.append(candidate)
            else:
                # Нейтральное мнение ИИ - оставляем как есть
                validated_candidates.append(candidate)
        
        return validated_candidates
    
    async def _save_analysis_to_db(self,
                                 lightcurve_data: LightCurveData,
                                 bls_result: SearchResult,
                                 ai_predictions: List,
                                 start_time: datetime):
        """Сохранение результата анализа в БД"""
        if not self.db_manager:
            return
        
        try:
            # Подготавливаем данные для сохранения
            physical_params = {}
            if bls_result.candidates:
                best_candidate = bls_result.candidates[0]
                physical_params = {
                    'period': best_candidate.period,
                    'depth': best_candidate.depth,
                    'duration': best_candidate.duration,
                    'snr': best_candidate.snr,
                    'planet_radius': best_candidate.planet_radius,
                    'equilibrium_temp': best_candidate.equilibrium_temp
                }
            
            bls_params = {
                'best_period': bls_result.bls_result.best_period,
                'best_power': bls_result.bls_result.best_power,
                'total_candidates': bls_result.total_candidates
            }
            
            # Определяем наличие транзита
            is_transit = len(bls_result.candidates) > 0
            confidence = ai_predictions[0].confidence if ai_predictions else 0.5
            
            analysis_result = AnalysisResult(
                target_name=lightcurve_data.target_name,
                analysis_timestamp=start_time,
                model_version='1.0.0',
                is_transit=is_transit,
                confidence=confidence,
                transit_probability=confidence,
                physical_parameters=physical_params,
                bls_parameters=bls_params
            )
            
            await self.db_manager.save_analysis_result(analysis_result)
            
        except Exception as e:
            logger.error(f"Failed to save analysis to database: {e}")
    
    def _enhance_search_result(self, bls_result: SearchResult, ai_predictions: List) -> SearchResult:
        """Дополнение результата BLS данными ИИ"""
        # Добавляем ИИ метаданные к результату
        if ai_predictions:
            ai_pred = ai_predictions[0]
            # Можно добавить дополнительные поля к результату
            # Пока просто возвращаем исходный результат
        
        return bls_result
    
    def _generate_technical_analysis(self, prediction) -> str:
        """Генерация технического анализа"""
        return f"""
        Техническая оценка сигнала:
        - Вероятность транзита: {prediction.confidence:.3f}
        - Уровень уверенности: {prediction.confidence_level.value}
        - Статистическая значимость: {'Высокая' if prediction.confidence > 0.8 else 'Средняя'}
        - Рекомендуемые дополнительные наблюдения: {'Да' if prediction.confidence < 0.9 else 'Не требуются'}
        """
    
    def _assess_statistical_significance(self, prediction) -> str:
        """Оценка статистической значимости"""
        if prediction.confidence > 0.9:
            return "Статистически значимый результат (>3σ)"
        elif prediction.confidence > 0.7:
            return "Умеренно значимый результат (2-3σ)"
        else:
            return "Низкая статистическая значимость (<2σ)"
    
    async def _check_and_trigger_online_learning(self):
        """Проверка и запуск онлайн обучения при накоплении обратной связи"""
        if not self.db_manager:
            return
        
        # Проверяем количество новой обратной связи
        recent_feedback = await self.db_manager.get_feedback_for_active_learning(limit=50)
        
        # Если накопилось достаточно обратной связи, запускаем обучение
        if len(recent_feedback) >= 20:
            logger.info("Triggering online learning with new feedback")
            # Запускаем фоновое обучение
            asyncio.create_task(self._background_retrain(recent_feedback))
    
    async def _background_retrain(self, feedback_data: List):
        """Фоновое переобучение модели"""
        try:
            logger.info(f"Starting background retraining with {len(feedback_data)} samples")
            result = await self.retrain_model('active_learning', epochs=5)
            logger.info(f"Background retraining completed: {result}")
        except Exception as e:
            logger.error(f"Background retraining failed: {e}")
    
    async def _prepare_training_data_from_feedback(self, feedback_data: List) -> Dict:
        """Подготовка обучающих данных из пользовательской обратной связи"""
        lightcurves = []
        labels = []
        
        for feedback in feedback_data:
            # Получаем кривую блеска для цели
            analysis = await self.db_manager.get_analysis_by_target(feedback.target_name)
            if not analysis or not analysis.get('lightcurve_data'):
                continue
            
            # Извлекаем кривую блеска
            flux = np.array(analysis['lightcurve_data']['flux'])
            processed_flux = self._preprocess_lightcurve(flux)
            
            # Определяем метку на основе обратной связи
            if feedback.feedback_type == 'correct':
                label = 1 if analysis['is_transit'] else 0
            elif feedback.feedback_type == 'incorrect':
                label = 0 if analysis['is_transit'] else 1
            else:  # uncertain
                continue  # Пропускаем неопределенные случаи
            
            lightcurves.append(processed_flux)
            labels.append(label)
        
        if len(lightcurves) < 5:
            return None
        
        return {
            'lightcurves': np.array(lightcurves),
            'labels': np.array(labels)
        }
    
    def _compute_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Вычисление косинусного сходства между embeddings"""
        # Нормализуем векторы
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Косинусное сходство
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    async def close(self):
        """Закрытие сервиса"""
        if self.db_manager:
            await self.db_manager.close()
        logger.info("AI Enhanced BLS Service closed")
