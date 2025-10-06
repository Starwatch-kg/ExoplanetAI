"""
Enhanced Exoplanet Classifier with Improved Architecture
Улучшенный классификатор экзопланет с продвинутой архитектурой
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
import shap

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, 
        Dropout, BatchNormalization, Input, LSTM, Attention,
        MultiHeadAttention, LayerNormalization, Add
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedFeatureExtractor:
    """Расширенный экстрактор признаков с астрофизическими параметрами"""
    
    def __init__(self):
        self.base_features = [
            'flux_mean', 'flux_std', 'flux_skewness', 'flux_kurtosis',
            'transit_depth', 'transit_snr', 'transit_duration',
            'spectral_centroid', 'zero_crossing_rate'
        ]
        
        # Новые астрофизические признаки
        self.astrophysical_features = [
            'stellar_contamination_ratio',
            'limb_darkening_coefficient', 
            'eccentricity_proxy',
            'ttv_amplitude',
            'photometric_precision',
            'systematic_noise_level',
            'data_completeness',
            'instrumental_effects_score'
        ]
    
    def extract_astrophysical_features(self, 
                                     time: np.ndarray,
                                     flux: np.ndarray,
                                     transit_params: Dict) -> Dict[str, float]:
        """Извлечение астрофизических признаков"""
        features = {}
        
        try:
            # Stellar contamination ratio
            primary_depth = transit_params.get('primary_depth', 0.01)
            secondary_depth = transit_params.get('secondary_depth', 0.0001)
            features['stellar_contamination_ratio'] = secondary_depth / primary_depth if primary_depth > 0 else 0
            
            # Limb darkening coefficient
            ingress_duration = transit_params.get('ingress_duration', 0.1)
            egress_duration = transit_params.get('egress_duration', 0.1)
            transit_duration = transit_params.get('transit_duration', 1.0)
            features['limb_darkening_coefficient'] = (ingress_duration + egress_duration) / transit_duration
            
            # Orbital eccentricity proxy
            secondary_phase = transit_params.get('secondary_eclipse_phase', 0.5)
            features['eccentricity_proxy'] = abs(secondary_phase - 0.5)
            
            # TTV amplitude (simplified)
            if len(time) > 100:
                # Разбиваем на сегменты и ищем вариации периода
                n_segments = min(10, len(time) // 50)
                segment_size = len(time) // n_segments
                period_variations = []
                
                for i in range(n_segments - 1):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size
                    segment_flux = flux[start_idx:end_idx]
                    
                    # Простая оценка периода через автокорреляцию
                    autocorr = np.correlate(segment_flux, segment_flux, mode='full')
                    period_estimate = np.argmax(autocorr[len(autocorr)//2:])
                    period_variations.append(period_estimate)
                
                features['ttv_amplitude'] = np.std(period_variations) / np.mean(period_variations) if len(period_variations) > 1 else 0
            else:
                features['ttv_amplitude'] = 0
            
        except Exception as e:
            logger.warning(f"Astrophysical feature extraction failed: {e}")
            for feature in self.astrophysical_features:
                features[feature] = 0.0
        
        return features
    
    def extract_quality_features(self,
                               time: np.ndarray,
                               flux: np.ndarray,
                               flux_err: np.ndarray) -> Dict[str, float]:
        """Извлечение признаков качества данных"""
        features = {}
        
        try:
            # Photometric precision
            features['photometric_precision'] = np.median(flux_err) / np.median(flux)
            
            # Systematic noise level (корреляция с временными трендами)
            time_normalized = (time - np.min(time)) / (np.max(time) - np.min(time))
            correlation = np.corrcoef(flux, time_normalized)[0, 1]
            features['systematic_noise_level'] = abs(correlation)
            
            # Data completeness
            expected_points = (np.max(time) - np.min(time)) / np.median(np.diff(time))
            features['data_completeness'] = len(time) / expected_points
            
            # Instrumental effects score (периодические вариации)
            fft_flux = np.fft.fft(flux - np.mean(flux))
            power_spectrum = np.abs(fft_flux)**2
            # Ищем пики на характерных частотах инструментальных эффектов
            instrumental_frequencies = [1.0, 2.0, 0.5]  # cycles per day
            dt = np.median(np.diff(time))
            freqs = np.fft.fftfreq(len(flux), dt)
            
            instrumental_power = 0
            for freq in instrumental_frequencies:
                idx = np.argmin(np.abs(freqs - freq))
                if idx < len(power_spectrum):
                    instrumental_power += power_spectrum[idx]
            
            features['instrumental_effects_score'] = instrumental_power / np.sum(power_spectrum)
            
        except Exception as e:
            logger.warning(f"Quality feature extraction failed: {e}")
            features.update({
                'photometric_precision': 0.001,
                'systematic_noise_level': 0.0,
                'data_completeness': 1.0,
                'instrumental_effects_score': 0.0
            })
        
        return features


class AdvancedCNN:
    """Продвинутая CNN архитектура с attention и LSTM"""
    
    def __init__(self, sequence_length: int = 128, n_classes: int = 3):
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.model = None
        
    def build_model(self, hyperparams: Dict = None):
        """Построение продвинутой CNN архитектуры"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for advanced CNN")
        
        # Гиперпараметры по умолчанию
        if hyperparams is None:
            hyperparams = {
                'filters_1': 64,
                'filters_2': 128,
                'kernel_size': 5,
                'lstm_units': 64,
                'dropout_rate': 0.3,
                'learning_rate': 0.001
            }
        
        # Input layer
        inputs = Input(shape=(self.sequence_length, 1))
        
        # First convolutional block with attention
        x = Conv1D(hyperparams['filters_1'], hyperparams['kernel_size'], activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(hyperparams['dropout_rate'])(x)
        
        # Self-attention mechanism
        attention_output = MultiHeadAttention(
            num_heads=4, 
            key_dim=hyperparams['filters_1']//4
        )(x, x)
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        
        # Second convolutional block
        x = Conv1D(hyperparams['filters_2'], 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(hyperparams['dropout_rate'])(x)
        
        # LSTM for temporal dependencies
        x = LSTM(hyperparams['lstm_units'], return_sequences=True)(x)
        x = LSTM(hyperparams['lstm_units']//2)(x)
        x = Dropout(hyperparams['dropout_rate'])(x)
        
        # Dense layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(hyperparams['dropout_rate'])(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with advanced optimizer
        optimizer = Adam(
            learning_rate=hyperparams['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


class OptimizedEnsemble:
    """Оптимизированный ансамбль с Optuna и SHAP"""
    
    def __init__(self):
        self.xgb_model = None
        self.cnn_model = None
        self.scaler = StandardScaler()
        self.best_params = None
        self.explainer = None
        
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50):
        """Оптимизация гиперпараметров с Optuna"""
        
        def objective(trial):
            # XGBoost параметры
            xgb_params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0)
            }
            
            # Создаем и обучаем модель
            model = xgb.XGBClassifier(**xgb_params, random_state=42)
            
            # Кросс-валидация
            cv_scores = cross_val_score(
                model, X, y, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='f1_macro'
            )
            
            return np.mean(cv_scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        logger.info(f"Best hyperparameters: {self.best_params}")
        
        return study.best_value
    
    def train_optimized_model(self, X: np.ndarray, y: np.ndarray):
        """Обучение оптимизированной модели"""
        
        # Нормализация данных
        X_scaled = self.scaler.fit_transform(X)
        
        # Обучение XGBoost с лучшими параметрами
        if self.best_params is None:
            logger.warning("Hyperparameters not optimized, using defaults")
            self.best_params = {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'scale_pos_weight': 3.0
            }
        
        self.xgb_model = xgb.XGBClassifier(**self.best_params, random_state=42)
        self.xgb_model.fit(X_scaled, y)
        
        # Инициализация SHAP explainer
        self.explainer = shap.TreeExplainer(self.xgb_model)
        
        logger.info("Optimized ensemble model trained successfully")
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Dict[str, Any]:
        """Предсказание с оценкой неопределенности"""
        if self.xgb_model is None:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X.reshape(1, -1))
        
        # Получаем вероятности
        probabilities = self.xgb_model.predict_proba(X_scaled)[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        # Оценка неопределенности через энтропию
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
        uncertainty = entropy / np.log(len(probabilities))  # Нормализованная энтропия
        
        # SHAP объяснения
        shap_values = self.explainer.shap_values(X_scaled)
        feature_importance = np.abs(shap_values[predicted_class][0])
        
        class_names = ['CANDIDATE', 'PC', 'FP']
        
        return {
            'class_name': class_names[predicted_class],
            'confidence': float(confidence),
            'uncertainty': float(uncertainty),
            'probabilities': {class_names[i]: float(prob) for i, prob in enumerate(probabilities)},
            'feature_importance': feature_importance.tolist(),
            'uncertainty_bounds': [
                float(confidence - uncertainty/2),
                float(confidence + uncertainty/2)
            ]
        }
    
    def explain_prediction(self, X: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """Объяснение предсказания через SHAP"""
        if self.explainer is None:
            raise ValueError("Model not trained or SHAP explainer not initialized")
        
        X_scaled = self.scaler.transform(X.reshape(1, -1))
        shap_values = self.explainer.shap_values(X_scaled)
        
        # Получаем важность признаков для каждого класса
        class_explanations = {}
        class_names = ['CANDIDATE', 'PC', 'FP']
        
        for i, class_name in enumerate(class_names):
            if feature_names:
                feature_importance = dict(zip(feature_names, shap_values[i][0]))
            else:
                feature_importance = {f'feature_{j}': val for j, val in enumerate(shap_values[i][0])}
            
            # Сортируем по важности
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            class_explanations[class_name] = {
                'feature_importance': dict(sorted_features[:10]),  # Топ-10 признаков
                'total_contribution': float(np.sum(np.abs(shap_values[i][0])))
            }
        
        return {
            'class_explanations': class_explanations,
            'reasoning': self._generate_reasoning(class_explanations)
        }
    
    def _generate_reasoning(self, explanations: Dict) -> List[str]:
        """Генерация текстового объяснения решения"""
        reasoning = []
        
        for class_name, data in explanations.items():
            top_features = list(data['feature_importance'].items())[:3]
            
            if top_features:
                feature_text = ", ".join([f"{feat}" for feat, _ in top_features])
                reasoning.append(
                    f"Для класса {class_name}: ключевые признаки - {feature_text}"
                )
        
        return reasoning


def generate_recommendations(prediction: Dict[str, Any]) -> List[str]:
    """Генерация рекомендаций на основе предсказания"""
    recommendations = []
    
    confidence = prediction['confidence']
    uncertainty = prediction['uncertainty']
    class_name = prediction['class_name']
    
    if confidence > 0.9:
        recommendations.append("Высокая уверенность в классификации")
    elif confidence > 0.7:
        recommendations.append("Умеренная уверенность, рекомендуется дополнительная проверка")
    else:
        recommendations.append("Низкая уверенность, требуется детальный анализ")
    
    if uncertainty > 0.3:
        recommendations.append("Высокая неопределенность модели, рекомендуется сбор дополнительных данных")
    
    if class_name == 'CANDIDATE':
        recommendations.append("Рекомендуется follow-up наблюдения для подтверждения")
    elif class_name == 'FP':
        recommendations.append("Проверьте на систематические ошибки и инструментальные эффекты")
    
    return recommendations
