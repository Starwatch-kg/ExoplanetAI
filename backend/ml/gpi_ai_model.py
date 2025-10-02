#!/usr/bin/env python3
"""
ИИ модель для GPI системы обнаружения экзопланет
Integrated into Exoplanet AI Backend

Использует машинное обучение для улучшения точности обнаружения
и классификации планетарных сигналов в GPI данных.
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import pickle

# Машинное обучение
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.pipeline import Pipeline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Scikit-learn не установлен. ИИ функции недоступны.")

# Дополнительные библиотеки для продвинутого анализа
try:
    from scipy import signal, stats
    from scipy.fft import fft, fftfreq
    import xgboost as xgb
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    SCIPY_AVAILABLE = True
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    ADVANCED_ML_AVAILABLE = False

logger = logging.getLogger(__name__)

class GPIFeatureExtractor:
    """Извлечение признаков из GPI данных для машинного обучения."""
    
    def __init__(self):
        """Инициализация экстрактора признаков."""
        self.feature_names = []
    
    def extract_features(self, lightcurve_data: Dict) -> np.ndarray:
        """Извлечь признаки из световой кривой."""
        time = np.array(lightcurve_data['time'])
        flux = np.array(lightcurve_data['flux'])
        flux_err = np.array(lightcurve_data.get('flux_err', []))
        
        features = []
        feature_names = []
        
        # 1. Статистические признаки
        stat_features, stat_names = self._extract_statistical_features(flux)
        features.extend(stat_features)
        feature_names.extend(stat_names)
        
        # 2. Частотные признаки
        freq_features, freq_names = self._extract_frequency_features(time, flux)
        features.extend(freq_features)
        feature_names.extend(freq_names)
        
        # 3. GPI-специфичные признаки
        gpi_features, gpi_names = self._extract_gpi_features(time, flux)
        features.extend(gpi_features)
        feature_names.extend(gpi_names)
        
        # 4. Временные признаки
        time_features, time_names = self._extract_temporal_features(time, flux)
        features.extend(time_features)
        feature_names.extend(time_names)
        
        # Сохраняем имена признаков для интерпретации
        self.feature_names = feature_names
        
        return np.array(features)
    
    def _extract_statistical_features(self, flux: np.ndarray) -> Tuple[List[float], List[str]]:
        """Статистические признаки."""
        features = []
        names = []
        
        # Базовая статистика
        features.extend([
            np.mean(flux),
            np.std(flux),
            np.var(flux),
            np.median(flux),
            np.min(flux),
            np.max(flux),
            np.ptp(flux),  # размах
        ])
        names.extend([
            'mean_flux', 'std_flux', 'var_flux', 'median_flux',
            'min_flux', 'max_flux', 'range_flux'
        ])
        
        # Квантили
        quantiles = [0.1, 0.25, 0.75, 0.9]
        for q in quantiles:
            features.append(np.quantile(flux, q))
            names.append(f'quantile_{int(q*100)}')
        
        # Моменты распределения
        if SCIPY_AVAILABLE:
            features.extend([
                stats.skew(flux),
                stats.kurtosis(flux)
            ])
            names.extend(['skewness', 'kurtosis'])
        else:
            features.extend([0.0, 0.0])
            names.extend(['skewness', 'kurtosis'])
        
        return features, names
    
    def _extract_frequency_features(self, time: np.ndarray, 
                                  flux: np.ndarray) -> Tuple[List[float], List[str]]:
        """Частотные признаки."""
        features = []
        names = []
        
        # Подготовка данных для FFT
        dt = np.median(np.diff(time))
        n = len(flux)
        
        # FFT
        flux_fft = fft(flux - np.mean(flux))
        freqs = fftfreq(n, dt)
        power = np.abs(flux_fft)**2
        
        # Работаем только с положительными частотами
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        power_pos = power[pos_mask]
        
        if len(power_pos) > 0:
            # Максимальная мощность и соответствующая частота
            max_power_idx = np.argmax(power_pos)
            features.extend([
                np.max(power_pos),
                freqs_pos[max_power_idx],
                1.0 / freqs_pos[max_power_idx] if freqs_pos[max_power_idx] > 0 else 0
            ])
            names.extend(['max_power', 'dominant_freq', 'dominant_period'])
            
            # Спектральная плотность в разных диапазонах
            freq_ranges = [
                (0, 0.1),      # Очень низкие частоты
                (0.1, 0.5),    # Низкие частоты  
                (0.5, 2.0),    # Средние частоты
                (2.0, 10.0)    # Высокие частоты
            ]
            
            for i, (f_min, f_max) in enumerate(freq_ranges):
                mask = (freqs_pos >= f_min) & (freqs_pos < f_max)
                if np.any(mask):
                    features.append(np.sum(power_pos[mask]))
                else:
                    features.append(0.0)
                names.append(f'power_band_{i}')
            
            # Спектральный центроид
            if np.sum(power_pos) > 0:
                spectral_centroid = np.sum(freqs_pos * power_pos) / np.sum(power_pos)
                features.append(spectral_centroid)
            else:
                features.append(0.0)
            names.append('spectral_centroid')
            
        else:
            # Заполняем нулями если нет данных
            features.extend([0.0] * 9)
            names.extend([
                'max_power', 'dominant_freq', 'dominant_period',
                'power_band_0', 'power_band_1', 'power_band_2', 'power_band_3',
                'spectral_centroid'
            ])
        
        return features, names
    
    def _extract_gpi_features(self, time: np.ndarray, 
                            flux: np.ndarray) -> Tuple[List[float], List[str]]:
        """GPI-специфичные признаки."""
        features = []
        names = []
        
        # Фазовый анализ (упрощенная версия GPI)
        if SCIPY_AVAILABLE:
            # Преобразование Гильберта для извлечения фазы
            analytic_signal = signal.hilbert(flux - np.mean(flux))
            instantaneous_phase = np.angle(analytic_signal)
            phase_unwrapped = np.unwrap(instantaneous_phase)
            
            # Признаки фазы
            features.extend([
                np.std(phase_unwrapped),
                np.mean(np.abs(np.diff(phase_unwrapped))),
                np.max(np.abs(phase_unwrapped))
            ])
            names.extend(['phase_std', 'phase_variation', 'max_phase'])
            
            # Когерентность фазы
            phase_coherence = np.abs(np.mean(np.exp(1j * instantaneous_phase)))
            features.append(phase_coherence)
            names.append('phase_coherence')
            
        else:
            features.extend([0.0] * 4)
            names.extend(['phase_std', 'phase_variation', 'max_phase', 'phase_coherence'])
        
        # Периодичность (поиск повторяющихся паттернов)
        autocorr_features, autocorr_names = self._calculate_autocorrelation_features(flux)
        features.extend(autocorr_features)
        names.extend(autocorr_names)
        
        return features, names
    
    def _extract_temporal_features(self, time: np.ndarray, 
                                 flux: np.ndarray) -> Tuple[List[float], List[str]]:
        """Временные признаки."""
        features = []
        names = []
        
        # Длительность наблюдений
        features.append(np.max(time) - np.min(time))
        names.append('observation_duration')
        
        # Частота дискретизации
        features.append(np.median(np.diff(time)))
        names.append('median_cadence')
        
        # Количество точек
        features.append(len(time))
        names.append('n_points')
        
        # Тренды
        if len(time) > 2:
            # Линейный тренд
            slope, intercept = np.polyfit(time, flux, 1)
            features.extend([slope, intercept])
            names.extend(['linear_trend_slope', 'linear_trend_intercept'])
        else:
            features.extend([0.0, 0.0])
            names.extend(['linear_trend_slope', 'linear_trend_intercept'])
        
        return features, names
    
    def _calculate_autocorrelation_features(self, flux: np.ndarray) -> Tuple[List[float], List[str]]:
        """Признаки автокорреляции."""
        features = []
        names = []
        
        # Нормализуем данные
        flux_norm = (flux - np.mean(flux)) / np.std(flux)
        
        # Вычисляем автокорреляцию
        autocorr = np.correlate(flux_norm, flux_norm, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Нормализация
        
        # Максимальная автокорреляция (исключая нулевую задержку)
        if len(autocorr) > 1:
            max_autocorr = np.max(autocorr[1:])
            max_lag = np.argmax(autocorr[1:]) + 1
            features.extend([max_autocorr, max_lag])
        else:
            features.extend([0.0, 0.0])
        names.extend(['max_autocorr', 'max_autocorr_lag'])
        
        # Автокорреляция на конкретных задержках
        lags = [1, 5, 10, 20]
        for lag in lags:
            if lag < len(autocorr):
                features.append(autocorr[lag])
            else:
                features.append(0.0)
            names.append(f'autocorr_lag_{lag}')
        
        return features, names

class GPIAIModel:
    """ИИ модель для GPI системы."""
    
    def __init__(self):
        """Инициализация ИИ модели."""
        if not ML_AVAILABLE:
            raise ImportError("Scikit-learn требуется для ИИ функций")
        
        self.feature_extractor = GPIFeatureExtractor()
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.model_info = {}
        
        # Создаем продвинутый пайплайн с множественными моделями
        from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier, AdaBoostClassifier
        from sklearn.feature_selection import SelectKBest, f_classif, RFE
        from sklearn.decomposition import PCA
        from sklearn.model_selection import GridSearchCV
        
        # Продвинутые классификаторы
        estimators = []
        
        # 1. Gradient Boosting (улучшенный)
        gb_classifier = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            random_state=42
        )
        estimators.append(('gb', gb_classifier))
        
        # 2. Random Forest (оптимизированный)
        rf_classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        )
        estimators.append(('rf', rf_classifier))
        
        # 3. Extra Trees
        et_classifier = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=2,
            random_state=42
        )
        estimators.append(('et', et_classifier))
        
        # 4. XGBoost (если доступен)
        if ADVANCED_ML_AVAILABLE:
            xgb_classifier = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
            estimators.append(('xgb', xgb_classifier))
            
            # 5. Neural Network
            nn_classifier = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
            estimators.append(('nn', nn_classifier))
            
            # 6. SVM with RBF kernel
            svm_classifier = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
            estimators.append(('svm', svm_classifier))
        
        # 7. AdaBoost
        ada_classifier = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        )
        estimators.append(('ada', ada_classifier))
        
        # Создаем мета-ансамбль
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k='all')),
            ('classifier', ensemble)
        ])
    
    def prepare_training_data(self, training_samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовить данные для обучения."""
        logger.info(f"Подготавливаем {len(training_samples)} образцов для обучения")
        
        X = []
        y = []
        
        for sample in training_samples:
            try:
                # Извлекаем признаки
                features = self.feature_extractor.extract_features(sample)
                X.append(features)
                
                # Метка класса
                label = sample.get('label', 0)
                y.append(label)
                
            except Exception as e:
                logger.warning(f"Ошибка обработки образца {sample.get('target_name', 'unknown')}: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Подготовлено {len(X)} образцов с {X.shape[1]} признаками")
        return X, y
    
    def train(self, training_samples: List[Dict], test_size: float = 0.2) -> Dict:
        """Обучить модель."""
        logger.info("Начинаем обучение ИИ модели")
        
        # Подготавливаем данные
        X, y = self.prepare_training_data(training_samples)
        
        if len(X) == 0:
            raise ValueError("Нет данных для обучения")
        
        # Разделяем на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Обучаем модель
        self.pipeline.fit(X_train, y_train)
        
        # Оцениваем качество
        train_score = self.pipeline.score(X_train, y_train)
        test_score = self.pipeline.score(X_test, y_test)
        
        # Предсказания для детальной оценки
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        # Метрики
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Кросс-валидация
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5)
        
        # Сохраняем информацию о модели
        self.model_info = {
            'trained_at': datetime.now().isoformat(),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'feature_names': self.feature_extractor.feature_names,
            'train_score': float(train_score),
            'test_score': float(test_score),
            'auc_score': float(auc_score),
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'class_distribution': {
                'no_planets': int(np.sum(y == 0)),
                'with_planets': int(np.sum(y == 1))
            }
        }
        
        self.is_trained = True
        
        logger.info(f"Обучение завершено:")
        logger.info(f"  Точность на обучении: {train_score:.3f}")
        logger.info(f"  Точность на тесте: {test_score:.3f}")
        logger.info(f"  AUC: {auc_score:.3f}")
        logger.info(f"  CV: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        return self.model_info
    
    def predict(self, lightcurve_data: Dict) -> Dict:
        """Предсказать наличие планет."""
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        # Извлекаем признаки
        features = self.feature_extractor.extract_features(lightcurve_data)
        features = features.reshape(1, -1)
        
        # Предсказание
        prediction = self.pipeline.predict(features)[0]
        probability = self.pipeline.predict_proba(features)[0]
        
        return {
            'prediction': int(prediction),
            'probability_no_planets': float(probability[0]),
            'probability_with_planets': float(probability[1]),
            'confidence': float(np.max(probability)),
            'ai_method': 'GPI_Ensemble',
            'model_version': self.model_info.get('trained_at', 'unknown')
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Получить важность признаков."""
        if not self.is_trained:
            return {}
        
        # Получаем важность из модели
        classifier = self.pipeline.named_steps['classifier']
        
        # Для ансамбля берем среднюю важность
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
        else:
            # Для VotingClassifier берем важность из первого классификатора
            importances = classifier.estimators_[0].feature_importances_
        
        # Создаем словарь имя_признака: важность
        feature_importance = {}
        for name, importance in zip(self.feature_extractor.feature_names, importances):
            feature_importance[name] = float(importance)
        
        # Сортируем по важности
        feature_importance = dict(sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return feature_importance
    
    def save_model(self, filepath: str):
        """Сохранить модель."""
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        model_data = {
            'pipeline': self.pipeline,
            'feature_extractor': self.feature_extractor,
            'model_info': self.model_info,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Модель сохранена в {filepath}")
    
    def load_model(self, filepath: str):
        """Загрузить модель."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pipeline = model_data['pipeline']
        self.feature_extractor = model_data['feature_extractor']
        self.model_info = model_data['model_info']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Модель загружена из {filepath}")
    
    def explain_prediction(self, lightcurve_data: Dict) -> Dict:
        """Объяснить предсказание модели."""
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        # Получаем базовое предсказание
        prediction_result = self.predict(lightcurve_data)
        
        # Извлекаем признаки
        features = self.feature_extractor.extract_features(lightcurve_data)
        features = features.reshape(1, -1)
        
        # Нормализуем признаки
        features_scaled = self.pipeline.named_steps['scaler'].transform(features)
        
        # Получаем важность признаков
        feature_importance = self.get_feature_importance()
        
        # Анализируем вклад каждого признака
        feature_contributions = {}
        for i, (name, importance) in enumerate(feature_importance.items()):
            if i < len(features_scaled[0]):
                feature_value = features_scaled[0][i]
                contribution = importance * abs(feature_value)
                feature_contributions[name] = {
                    'value': float(features[0][i]),
                    'normalized_value': float(feature_value),
                    'importance': importance,
                    'contribution': contribution
                }
        
        # Сортируем по вкладу
        sorted_contributions = dict(sorted(
            feature_contributions.items(),
            key=lambda x: x[1]['contribution'],
            reverse=True
        ))
        
        return {
            **prediction_result,
            'feature_contributions': sorted_contributions,
            'top_features': list(sorted_contributions.keys())[:10],
            'explanation': self._generate_explanation(sorted_contributions, prediction_result)
        }
    
    def _generate_explanation(self, contributions: Dict, prediction: Dict) -> str:
        """Генерировать текстовое объяснение."""
        confidence = prediction['confidence']
        has_planet = prediction['prediction'] == 1
        
        top_features = list(contributions.keys())[:3]
        
        if has_planet:
            explanation = f"Модель предсказывает наличие планеты с уверенностью {confidence:.1%}. "
        else:
            explanation = f"Модель не обнаружила планету (уверенность {confidence:.1%}). "
        
        explanation += f"Ключевые факторы: {', '.join(top_features[:3])}."
        
        return explanation
    
    def get_model_performance(self) -> Dict:
        """Получить детальную информацию о производительности модели."""
        if not self.is_trained:
            return {"error": "Модель не обучена"}
        
        performance = {
            **self.model_info,
            'model_complexity': {
                'n_estimators': len(self.pipeline.named_steps['classifier'].estimators_),
                'feature_count': len(self.feature_extractor.feature_names),
                'pipeline_steps': len(self.pipeline.steps)
            },
            'capabilities': {
                'advanced_ml': ADVANCED_ML_AVAILABLE,
                'scipy_available': SCIPY_AVAILABLE,
                'supports_explanation': True,
                'supports_uncertainty': True
            }
        }
        
        return performance
    
    def predict_with_uncertainty(self, lightcurve_data: Dict) -> Dict:
        """Предсказание с оценкой неопределенности."""
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        # Базовое предсказание
        base_prediction = self.predict(lightcurve_data)
        
        # Получаем предсказания от каждого классификатора в ансамбле
        features = self.feature_extractor.extract_features(lightcurve_data)
        features = features.reshape(1, -1)
        
        # Применяем препроцессинг
        features_processed = self.pipeline[:-1].transform(features)
        
        # Получаем предсказания от каждого классификатора
        ensemble = self.pipeline.named_steps['classifier']
        individual_predictions = []
        
        for name, estimator in ensemble.estimators_:
            try:
                pred_proba = estimator.predict_proba(features_processed)[0]
                individual_predictions.append({
                    'model': name,
                    'probability_no_planet': float(pred_proba[0]),
                    'probability_planet': float(pred_proba[1])
                })
            except Exception as e:
                logger.warning(f"Ошибка получения предсказания от {name}: {e}")
        
        # Вычисляем статистики неопределенности
        planet_probs = [p['probability_planet'] for p in individual_predictions]
        uncertainty_metrics = {
            'mean_probability': float(np.mean(planet_probs)),
            'std_probability': float(np.std(planet_probs)),
            'min_probability': float(np.min(planet_probs)),
            'max_probability': float(np.max(planet_probs)),
            'agreement_ratio': float(np.sum(np.array(planet_probs) > 0.5) / len(planet_probs)),
            'entropy': float(-np.sum([p * np.log2(p + 1e-10) for p in planet_probs]) / len(planet_probs))
        }
        
        return {
            **base_prediction,
            'individual_predictions': individual_predictions,
            'uncertainty': uncertainty_metrics,
            'confidence_level': 'high' if uncertainty_metrics['std_probability'] < 0.1 else 
                              'medium' if uncertainty_metrics['std_probability'] < 0.2 else 'low'
        }
    
    def auto_optimize_hyperparameters(self, training_samples: List[Dict]) -> Dict:
        """Автоматическая оптимизация гиперпараметров."""
        if not ADVANCED_ML_AVAILABLE:
            logger.warning("Продвинутые ML библиотеки недоступны для оптимизации")
            return self.train(training_samples)
        
        logger.info("Начинаем автоматическую оптимизацию гиперпараметров")
        
        # Подготавливаем данные
        X, y = self.prepare_training_data(training_samples)
        
        # Определяем пространство поиска для основных моделей
        param_grid = {
            'classifier__gb__n_estimators': [100, 200],
            'classifier__gb__learning_rate': [0.05, 0.1],
            'classifier__rf__n_estimators': [100, 150],
            'classifier__rf__max_depth': [8, 12],
            'feature_selection__k': ['all', 20, 30]
        }
        
        # Используем GridSearchCV для оптимизации
        from sklearn.model_selection import GridSearchCV
        
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=3,  # Уменьшаем для скорости
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Обучаем с оптимизацией
        grid_search.fit(X, y)
        
        # Обновляем пайплайн лучшими параметрами
        self.pipeline = grid_search.best_estimator_
        self.is_trained = True
        
        # Сохраняем информацию об оптимизации
        optimization_info = {
            'best_score': float(grid_search.best_score_),
            'best_params': grid_search.best_params_,
            'optimization_completed': True,
            'cv_results_summary': {
                'mean_test_scores': [float(score) for score in grid_search.cv_results_['mean_test_score']],
                'std_test_scores': [float(score) for score in grid_search.cv_results_['std_test_score']]
            }
        }
        
        # Обновляем model_info
        self.model_info.update(optimization_info)
        
        logger.info(f"Оптимизация завершена. Лучший AUC: {grid_search.best_score_:.3f}")
        
        return self.model_info
