"""
Real Data ML Classifier - Optimized for NASA/MAST real data
Классификатор ML оптимизированный для реальных данных NASA
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import joblib
import time

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    import xgboost as xgb
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from core.logging import get_logger

logger = get_logger(__name__)


class RealDataClassifier:
    """
    ML классификатор оптимизированный для реальных данных NASA
    """
    
    def __init__(self, model_dir: str = "models/real_data"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Модели
        self.rf_model = None
        self.xgb_model = None
        self.nn_model = None
        
        # Препроцессоры
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Веса для ансамбля
        self.ensemble_weights = {
            'random_forest': 0.3,
            'xgboost': 0.4,
            'neural_network': 0.3
        }
        
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Install: pip install scikit-learn xgboost")
        
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Install: pip install tensorflow")
    
    def extract_real_data_features(self, 
                                 time: np.ndarray, 
                                 flux: np.ndarray, 
                                 flux_err: Optional[np.ndarray] = None,
                                 bls_result: Optional[Dict] = None) -> Dict[str, float]:
        """
        Извлечение признаков оптимизированных для реальных данных NASA
        
        Args:
            time: Временной ряд
            flux: Поток
            flux_err: Ошибки потока
            bls_result: Результаты BLS анализа
            
        Returns:
            Словарь с признаками
        """
        features = {}
        
        try:
            # Базовые статистики
            features['mean_flux'] = float(np.mean(flux))
            features['std_flux'] = float(np.std(flux))
            features['median_flux'] = float(np.median(flux))
            features['mad_flux'] = float(np.median(np.abs(flux - np.median(flux))))
            
            # Статистики времени
            features['time_span'] = float(np.max(time) - np.min(time))
            features['cadence_median'] = float(np.median(np.diff(time)))
            features['cadence_std'] = float(np.std(np.diff(time)))
            features['data_points'] = len(time)
            
            # Качество данных
            if flux_err is not None:
                features['mean_error'] = float(np.mean(flux_err))
                features['snr_photometric'] = float(np.median(flux / flux_err))
                features['error_scatter'] = float(np.std(flux_err))
            else:
                features['mean_error'] = 0.0
                features['snr_photometric'] = 0.0
                features['error_scatter'] = 0.0
            
            # Вариабельность
            features['rms'] = float(np.sqrt(np.mean((flux - np.mean(flux))**2)))
            features['range_flux'] = float(np.max(flux) - np.min(flux))
            features['iqr_flux'] = float(np.percentile(flux, 75) - np.percentile(flux, 25))
            
            # Асимметрия и эксцесс
            from scipy import stats
            features['skewness'] = float(stats.skew(flux))
            features['kurtosis'] = float(stats.kurtosis(flux))
            
            # Периодические признаки
            try:
                from scipy.signal import periodogram
                freqs, power = periodogram(flux, fs=1/np.median(np.diff(time)))
                features['peak_power'] = float(np.max(power))
                features['power_ratio'] = float(np.max(power) / np.mean(power))
                
                # Находим доминантную частоту
                peak_idx = np.argmax(power[1:]) + 1  # Исключаем DC компоненту
                if peak_idx < len(freqs):
                    features['dominant_period'] = float(1.0 / freqs[peak_idx])
                else:
                    features['dominant_period'] = 0.0
            except:
                features['peak_power'] = 0.0
                features['power_ratio'] = 0.0
                features['dominant_period'] = 0.0
            
            # Автокорреляционные признаки
            try:
                autocorr = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0]  # Нормализация
                
                # Первый минимум автокорреляции
                if len(autocorr) > 10:
                    features['autocorr_min'] = float(np.min(autocorr[1:min(100, len(autocorr))]))
                    features['autocorr_lag_min'] = float(np.argmin(autocorr[1:min(100, len(autocorr))]) + 1)
                else:
                    features['autocorr_min'] = 0.0
                    features['autocorr_lag_min'] = 0.0
            except:
                features['autocorr_min'] = 0.0
                features['autocorr_lag_min'] = 0.0
            
            # BLS признаки (если доступны)
            if bls_result:
                features['bls_period'] = float(bls_result.get('best_period', 0))
                features['bls_power'] = float(bls_result.get('best_power', 0))
                features['bls_snr'] = float(bls_result.get('snr', 0))
                features['bls_depth'] = float(bls_result.get('transit_depth', 0))
                features['bls_duration'] = float(bls_result.get('transit_duration_hours', 0))
                features['bls_significance'] = float(bls_result.get('significance', 0))
            else:
                features['bls_period'] = 0.0
                features['bls_power'] = 0.0
                features['bls_snr'] = 0.0
                features['bls_depth'] = 0.0
                features['bls_duration'] = 0.0
                features['bls_significance'] = 0.0
            
            # Дополнительные признаки для реальных данных
            features['flux_ratio_95_5'] = float(np.percentile(flux, 95) / np.percentile(flux, 5))
            features['flux_beyond_1sigma'] = float(np.sum(np.abs(flux - np.mean(flux)) > np.std(flux)) / len(flux))
            features['flux_beyond_2sigma'] = float(np.sum(np.abs(flux - np.mean(flux)) > 2*np.std(flux)) / len(flux))
            
            # Тренды
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(time, flux)
                features['linear_trend_slope'] = float(slope)
                features['linear_trend_r2'] = float(r_value**2)
                features['linear_trend_p_value'] = float(p_value)
            except:
                features['linear_trend_slope'] = 0.0
                features['linear_trend_r2'] = 0.0
                features['linear_trend_p_value'] = 1.0
            
            logger.debug(f"Extracted {len(features)} features for real data classification")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def build_neural_network(self, input_dim: int, num_classes: int = 3) -> tf.keras.Model:
        """
        Создание нейронной сети для классификации реальных данных
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for neural network")
        
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_ensemble(self, 
                      features_list: List[Dict[str, float]],
                      labels: List[str],
                      test_size: float = 0.2) -> Dict[str, Any]:
        """
        Обучение ансамбля моделей на реальных данных
        
        Args:
            features_list: Список словарей с признаками
            labels: Метки классов
            test_size: Размер тестовой выборки
            
        Returns:
            Метрики обучения
        """
        logger.info("Training ensemble on real NASA data")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn required for ensemble training")
        
        # Подготовка данных
        feature_names = list(features_list[0].keys())
        X = np.array([[f.get(name, 0.0) for name in feature_names] for f in features_list])
        y = np.array(labels)
        
        # Кодирование меток
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Нормализация признаков
        X_scaled = self.scaler.fit_transform(X)
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, stratify=y_encoded, random_state=42
        )
        
        metrics = {}
        
        # 1. Random Forest
        logger.info("Training Random Forest")
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        
        rf_score = self.rf_model.score(X_test, y_test)
        metrics['random_forest_accuracy'] = rf_score
        
        # 2. XGBoost
        if 'xgboost' in globals():
            logger.info("Training XGBoost")
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss'
            )
            self.xgb_model.fit(X_train, y_train)
            
            xgb_score = self.xgb_model.score(X_test, y_test)
            metrics['xgboost_accuracy'] = xgb_score
        
        # 3. Neural Network
        if TENSORFLOW_AVAILABLE:
            logger.info("Training Neural Network")
            self.nn_model = self.build_neural_network(X_train.shape[1], len(np.unique(y_encoded)))
            
            callbacks = [
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6)
            ]
            
            history = self.nn_model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            nn_score = self.nn_model.evaluate(X_test, y_test, verbose=0)[1]
            metrics['neural_network_accuracy'] = nn_score
        
        # Ансамблевые предсказания
        ensemble_predictions = self.predict_ensemble(X_test)
        ensemble_accuracy = np.mean(ensemble_predictions == y_test)
        metrics['ensemble_accuracy'] = ensemble_accuracy
        
        # Сохранение моделей
        self.save_models()
        
        logger.info(f"Ensemble training completed. Accuracy: {ensemble_accuracy:.3f}")
        return metrics
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """
        Ансамблевое предсказание
        """
        predictions = []
        weights = []
        
        if self.rf_model is not None:
            rf_pred = self.rf_model.predict_proba(X)
            predictions.append(rf_pred)
            weights.append(self.ensemble_weights['random_forest'])
        
        if self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict_proba(X)
            predictions.append(xgb_pred)
            weights.append(self.ensemble_weights['xgboost'])
        
        if self.nn_model is not None:
            nn_pred = self.nn_model.predict(X)
            predictions.append(nn_pred)
            weights.append(self.ensemble_weights['neural_network'])
        
        if not predictions:
            raise ValueError("No trained models available")
        
        # Взвешенное усреднение
        weights = np.array(weights) / np.sum(weights)
        ensemble_proba = np.zeros_like(predictions[0])
        
        for pred, weight in zip(predictions, weights):
            ensemble_proba += weight * pred
        
        return np.argmax(ensemble_proba, axis=1)
    
    def predict_single_real_data(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Предсказание для одного объекта с реальными данными
        
        Args:
            features: Словарь с признаками
            
        Returns:
            Результат классификации
        """
        try:
            # Подготовка признаков
            if hasattr(self, 'feature_names'):
                feature_vector = np.array([[features.get(name, 0.0) for name in self.feature_names]])
            else:
                # Fallback: используем все доступные признаки
                feature_vector = np.array([list(features.values())]).reshape(1, -1)
            
            # Нормализация
            if hasattr(self.scaler, 'mean_'):
                feature_vector = self.scaler.transform(feature_vector)
            
            # Ансамблевое предсказание
            if self.rf_model is not None or self.xgb_model is not None or self.nn_model is not None:
                prediction_idx = self.predict_ensemble(feature_vector)[0]
                
                # Получение вероятностей
                probabilities = {}
                if self.rf_model is not None:
                    rf_proba = self.rf_model.predict_proba(feature_vector)[0]
                    for i, class_name in enumerate(self.label_encoder.classes_):
                        probabilities[class_name] = float(rf_proba[i])
                
                predicted_class = self.label_encoder.classes_[prediction_idx]
                confidence = max(probabilities.values()) if probabilities else 0.5
                
            else:
                # Улучшенная эвристика для реальных данных
                predicted_class, confidence = self._enhanced_heuristic_for_real_data(features)
                probabilities = {
                    'Confirmed': 0.3,
                    'Candidate': 0.4, 
                    'False Positive': 0.3
                }
                probabilities[predicted_class] = confidence
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_probabilities': probabilities,
                'model_status': 'ensemble' if self.rf_model else 'enhanced_heuristic',
                'data_type': 'real_nasa_data'
            }
            
        except Exception as e:
            logger.error(f"Error in real data prediction: {e}")
            return {
                'predicted_class': 'Unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _enhanced_heuristic_for_real_data(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Улучшенная эвристика специально для реальных данных NASA
        """
        # Ключевые признаки для реальных данных
        bls_snr = features.get('bls_snr', 0)
        bls_depth = features.get('bls_depth', 0)
        bls_significance = features.get('bls_significance', 0)
        bls_period = features.get('bls_period', 0)
        
        # Качество данных
        data_points = features.get('data_points', 0)
        time_span = features.get('time_span', 0)
        snr_photometric = features.get('snr_photometric', 0)
        
        # Комплексная оценка качества сигнала для реальных данных
        signal_quality = (
            bls_snr * 0.3 + 
            bls_significance * 0.25 + 
            snr_photometric * 0.2 +
            min(1.0, data_points / 1000) * 0.15 +  # Бонус за количество точек
            min(1.0, time_span / 30) * 0.1  # Бонус за длительность наблюдений
        )
        
        # Критерии для реальных данных NASA (более строгие)
        if (bls_snr > 12 and bls_depth > 0.003 and bls_significance > 0.85 and 
            1 < bls_period < 100 and data_points > 500):
            predicted_class = "Confirmed"
            confidence = min(0.95, 0.75 + signal_quality * 0.04)
            
        elif (bls_snr > 8 and bls_depth > 0.002 and bls_significance > 0.7 and 
              1 < bls_period < 200 and data_points > 300):
            predicted_class = "Candidate"
            confidence = min(0.85, 0.55 + signal_quality * 0.05)
            
        elif (bls_snr > 5 and bls_depth > 0.001 and bls_significance > 0.5 and 
              data_points > 200):
            predicted_class = "Candidate"
            confidence = min(0.75, 0.45 + signal_quality * 0.06)
            
        elif bls_depth > 0.005:  # Очень глубокие транзиты
            predicted_class = "Candidate"
            confidence = min(0.80, 0.50 + bls_depth * 30)
            
        else:
            predicted_class = "False Positive"
            confidence = max(0.60, min(0.90, 0.60 + bls_significance * 0.3))
        
        return predicted_class, confidence
    
    def save_models(self):
        """Сохранение обученных моделей"""
        try:
            if self.rf_model is not None:
                joblib.dump(self.rf_model, self.model_dir / "random_forest.pkl")
            
            if self.xgb_model is not None:
                joblib.dump(self.xgb_model, self.model_dir / "xgboost.pkl")
            
            if self.nn_model is not None:
                self.nn_model.save(self.model_dir / "neural_network.h5")
            
            # Сохранение препроцессоров
            joblib.dump(self.scaler, self.model_dir / "scaler.pkl")
            joblib.dump(self.label_encoder, self.model_dir / "label_encoder.pkl")
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Загрузка обученных моделей"""
        try:
            rf_path = self.model_dir / "random_forest.pkl"
            if rf_path.exists():
                self.rf_model = joblib.load(rf_path)
            
            xgb_path = self.model_dir / "xgboost.pkl"
            if xgb_path.exists():
                self.xgb_model = joblib.load(xgb_path)
            
            nn_path = self.model_dir / "neural_network.h5"
            if nn_path.exists() and TENSORFLOW_AVAILABLE:
                self.nn_model = tf.keras.models.load_model(nn_path)
            
            # Загрузка препроцессоров
            scaler_path = self.model_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            encoder_path = self.model_dir / "label_encoder.pkl"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")


# Глобальный экземпляр классификатора
_real_data_classifier = None

def get_real_data_classifier() -> RealDataClassifier:
    """Получить глобальный экземпляр классификатора для реальных данных"""
    global _real_data_classifier
    if _real_data_classifier is None:
        _real_data_classifier = RealDataClassifier()
        _real_data_classifier.load_models()  # Попытка загрузить существующие модели
    return _real_data_classifier
