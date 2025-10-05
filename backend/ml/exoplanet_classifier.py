"""
Ensemble Exoplanet Classification System
Ансамблевая система классификации экзопланет

Включает:
- XGBoost для табличных данных
- 1D-CNN для временных рядов
- Random Forest для устойчивости к шуму
- Ensemble voting для финального предсказания
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import json
from pathlib import Path
import logging

# ML библиотеки
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, 
        Dropout, BatchNormalization, Input, LSTM, Attention
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available, CNN models disabled")

logger = logging.getLogger(__name__)


class ExoplanetCNN:
    """
    1D-CNN модель для анализа кривых блеска
    """
    
    def __init__(self, sequence_length: int = 64, n_classes: int = 3):
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self):
        """
        Построение 1D-CNN архитектуры
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for CNN model")
        
        model = Sequential([
            # Первый блок свертки
            Conv1D(32, 3, activation='relu', input_shape=(self.sequence_length, 1)),
            BatchNormalization(),
            Conv1D(32, 3, activation='relu'),
            MaxPooling1D(2),
            Dropout(0.25),
            
            # Второй блок свертки
            Conv1D(64, 3, activation='relu'),
            BatchNormalization(),
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            Dropout(0.25),
            
            # Третий блок свертки
            Conv1D(128, 3, activation='relu'),
            BatchNormalization(),
            GlobalMaxPooling1D(),
            
            # Полносвязные слои
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Dict:
        """
        Обучение CNN модели
        
        Args:
            X: Входные данные (n_samples, sequence_length)
            y: Метки классов
            validation_split: Доля данных для валидации
            
        Returns:
            История обучения
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for CNN training")
        
        # Нормализация данных
        X_scaled = self.scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Построение модели
        self.model = self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
        ]
        
        # Обучение
        history = self.model.fit(
            X_scaled, y,
            validation_split=validation_split,
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание классов"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Предсказание вероятностей классов"""
        return self.predict(X)


class ExoplanetEnsembleClassifier:
    """
    Ансамблевый классификатор экзопланет
    """
    
    def __init__(self, sequence_length: int = 64):
        self.sequence_length = sequence_length
        self.n_classes = 3
        self.class_names = ['CANDIDATE', 'PC', 'FP']
        
        # Модели
        self.xgb_model = None
        self.rf_model = None
        self.cnn_model = None
        self.ensemble_model = None
        
        # Препроцессоры
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Метрики
        self.training_metrics = {}
        
    def _prepare_tabular_data(self, features: List[Dict[str, float]]) -> np.ndarray:
        """
        Подготовка табличных данных для XGBoost и Random Forest
        """
        # Преобразуем список словарей в DataFrame
        df = pd.DataFrame(features)
        
        # Заполняем пропуски нулями
        df = df.fillna(0)
        
        # Масштабируем признаки
        X_scaled = self.feature_scaler.fit_transform(df.values)
        
        return X_scaled
    
    def _prepare_sequence_data(self, sequences: List[np.ndarray]) -> np.ndarray:
        """
        Подготовка последовательностей для CNN
        """
        # Приводим все последовательности к одной длине
        X_sequences = []
        
        for seq in sequences:
            if len(seq) >= self.sequence_length:
                # Обрезаем до нужной длины
                X_sequences.append(seq[:self.sequence_length])
            else:
                # Дополняем нулями
                padded = np.zeros(self.sequence_length)
                padded[:len(seq)] = seq
                X_sequences.append(padded)
        
        return np.array(X_sequences)
    
    def train(self, 
              features: List[Dict[str, float]],
              sequences: List[np.ndarray],
              labels: List[str],
              test_size: float = 0.2,
              cv_folds: int = 5) -> Dict[str, Any]:
        """
        Обучение ансамбля моделей
        
        Args:
            features: Список словарей с признаками
            sequences: Список последовательностей (кривые блеска)
            labels: Метки классов
            test_size: Размер тестовой выборки
            cv_folds: Количество фолдов для кросс-валидации
            
        Returns:
            Метрики обучения
        """
        logger.info("Starting ensemble training")
        
        # Подготовка данных
        X_tabular = self._prepare_tabular_data(features)
        X_sequences = self._prepare_sequence_data(sequences)
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Разделение на train/test
        (X_tab_train, X_tab_test, 
         X_seq_train, X_seq_test, 
         y_train, y_test) = train_test_split(
            X_tabular, X_sequences, y_encoded,
            test_size=test_size, 
            stratify=y_encoded,
            random_state=42
        )
        
        metrics = {}
        
        # 1. Обучение XGBoost
        logger.info("Training XGBoost model")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        self.xgb_model.fit(X_tab_train, y_train)
        xgb_pred = self.xgb_model.predict(X_tab_test)
        xgb_proba = self.xgb_model.predict_proba(X_tab_test)
        
        metrics['xgboost'] = {
            'accuracy': np.mean(xgb_pred == y_test),
            'classification_report': classification_report(y_test, xgb_pred, output_dict=True),
            'auc_score': roc_auc_score(y_test, xgb_proba, multi_class='ovr')
        }
        
        # 2. Обучение Random Forest
        logger.info("Training Random Forest model")
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_tab_train, y_train)
        rf_pred = self.rf_model.predict(X_tab_test)
        rf_proba = self.rf_model.predict_proba(X_tab_test)
        
        metrics['random_forest'] = {
            'accuracy': np.mean(rf_pred == y_test),
            'classification_report': classification_report(y_test, rf_pred, output_dict=True),
            'auc_score': roc_auc_score(y_test, rf_proba, multi_class='ovr'),
            'feature_importance': self.rf_model.feature_importances_.tolist()
        }
        
        # 3. Обучение CNN (если доступен TensorFlow)
        if TENSORFLOW_AVAILABLE:
            logger.info("Training CNN model")
            self.cnn_model = ExoplanetCNN(self.sequence_length, self.n_classes)
            cnn_history = self.cnn_model.train(X_seq_train, y_train)
            
            cnn_proba = self.cnn_model.predict(X_seq_test)
            cnn_pred = np.argmax(cnn_proba, axis=1)
            
            metrics['cnn'] = {
                'accuracy': np.mean(cnn_pred == y_test),
                'classification_report': classification_report(y_test, cnn_pred, output_dict=True),
                'auc_score': roc_auc_score(y_test, cnn_proba, multi_class='ovr'),
                'training_history': cnn_history
            }
        
        # 4. Создание ансамбля
        logger.info("Creating ensemble model")
        estimators = [
            ('xgb', self.xgb_model),
            ('rf', self.rf_model)
        ]
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        
        self.ensemble_model.fit(X_tab_train, y_train)
        ensemble_pred = self.ensemble_model.predict(X_tab_test)
        ensemble_proba = self.ensemble_model.predict_proba(X_tab_test)
        
        metrics['ensemble'] = {
            'accuracy': np.mean(ensemble_pred == y_test),
            'classification_report': classification_report(y_test, ensemble_pred, output_dict=True),
            'auc_score': roc_auc_score(y_test, ensemble_proba, multi_class='ovr')
        }
        
        # 5. Кросс-валидация
        logger.info("Performing cross-validation")
        cv_scores = cross_val_score(
            self.ensemble_model, X_tabular, y_encoded, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        metrics['cross_validation'] = {
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'scores': cv_scores.tolist()
        }
        
        self.training_metrics = metrics
        logger.info("Ensemble training completed")
        
        return metrics
    
    def predict_single(self, 
                      features: Dict[str, float],
                      sequence: np.ndarray) -> Dict[str, Any]:
        """
        Предсказание для одного объекта
        
        Args:
            features: Словарь с признаками
            sequence: Последовательность (кривая блеска)
            
        Returns:
            Результат классификации
        """
        if self.ensemble_model is None:
            # Простой fallback для необученной модели
            logger.warning("Model not trained, using simple heuristic")
            
            # Простая эвристика на основе признаков
            logger.info(f"Features type: {type(features)}, content: {features}")
            
            if isinstance(features, dict):
                # Используем некоторые ключевые признаки для простой классификации
                snr = features.get('snr', 0)
                depth = features.get('transit_depth', 0)
                significance = features.get('significance', 0)
                
                logger.info(f"Extracted values: snr={snr}, depth={depth}, significance={significance}")
                
                # Улучшенная эвристика с учетом дополнительных факторов
                period_stability = features.get('period_stability', 0)
                data_quality = features.get('data_quality', 0.5)
                
                # Комбинированная оценка качества сигнала
                signal_quality = (snr * 0.4 + significance * 0.3 + period_stability * 0.2 + data_quality * 0.1)
                
                # Отладочная информация
                logger.info(f"ML Classification: SNR={snr:.3f}, depth={depth:.6f}, significance={significance:.3f}, signal_quality={signal_quality:.3f}")
                
                # Более чувствительные пороги для детекции планет
                # Сначала проверяем самые сильные сигналы
                logger.info(f"Checking conditions: depth={depth:.6f}, snr={snr:.3f}, significance={significance:.3f}")
                
                if depth > 0.015 and snr > 8:  # Очень глубокие транзиты с хорошим SNR
                    predicted_class = "Confirmed"
                    confidence = min(0.95, 0.70 + depth * 15 + snr * 0.02)
                    logger.info(f"Condition 1 matched: Very deep transits - Confirmed with {confidence:.3f}")
                elif depth > 0.01 and snr > 6:  # Глубокие транзиты
                    predicted_class = "Confirmed"
                    confidence = min(0.90, 0.65 + depth * 20 + snr * 0.025)
                    logger.info(f"Condition 2 matched: Deep transits - Confirmed with {confidence:.3f}")
                elif (snr > 10 or signal_quality > 8) and depth > 0.002 and significance > 0.8:
                    predicted_class = "Confirmed"
                    confidence = min(0.95, 0.65 + signal_quality * 0.03)
                    logger.info(f"Condition 3 matched: High SNR/quality - Confirmed with {confidence:.3f}")
                elif (snr > 8 or signal_quality > 6) and depth > 0.001 and significance > 0.6:
                    predicted_class = "Candidate" 
                    confidence = min(0.85, 0.45 + signal_quality * 0.04)
                    logger.info(f"Condition 4 matched: Good SNR - Candidate with {confidence:.3f}")
                elif (snr > 5 or signal_quality > 4) and depth > 0.0008:
                    predicted_class = "Candidate"
                    confidence = min(0.80, 0.40 + signal_quality * 0.05)
                    logger.info(f"Condition 5 matched: Moderate SNR - Candidate with {confidence:.3f}")
                elif depth > 0.005:  # Глубокие транзиты даже при низком SNR
                    predicted_class = "Candidate"
                    confidence = min(0.80, 0.50 + depth * 60)
                    logger.info(f"Condition 6 matched: Deep transits (low SNR) - Candidate with {confidence:.3f}")
                elif depth > 0.002 and snr > 3:  # Умеренно глубокие транзиты
                    predicted_class = "Candidate"
                    confidence = min(0.75, 0.40 + depth * 50 + snr * 0.03)
                    logger.info(f"Condition 7 matched: Moderate transits - Candidate with {confidence:.3f}")
                else:
                    predicted_class = "False Positive"
                    confidence = max(0.55, min(0.85, 0.55 + significance * 0.25))
                    logger.info(f"No conditions matched - False Positive with {confidence:.3f}")
            else:
                # Если features - массив numpy
                predicted_class = "Candidate"
                confidence = 0.6
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_probabilities': {
                    'Confirmed': 0.3 if predicted_class != 'Confirmed' else confidence,
                    'Candidate': 0.4 if predicted_class != 'Candidate' else confidence,
                    'False Positive': 0.3 if predicted_class != 'False Positive' else confidence
                },
                'model_status': 'fallback_heuristic'
            }
        
        # Подготовка табличных данных
        feature_df = pd.DataFrame([features]).fillna(0)
        X_tab = self.feature_scaler.transform(feature_df.values)
        
        # Предсказание ансамбля
        ensemble_proba = self.ensemble_model.predict_proba(X_tab)[0]
        ensemble_pred = np.argmax(ensemble_proba)
        
        result = {
            'predicted_class': self.class_names[ensemble_pred],
            'confidence': float(ensemble_proba[ensemble_pred]),
            'class_probabilities': {
                self.class_names[i]: float(prob) 
                for i, prob in enumerate(ensemble_proba)
            },
            'model_predictions': {}
        }
        
        # Индивидуальные предсказания моделей
        xgb_proba = self.xgb_model.predict_proba(X_tab)[0]
        rf_proba = self.rf_model.predict_proba(X_tab)[0]
        
        result['model_predictions']['xgboost'] = {
            'class': self.class_names[np.argmax(xgb_proba)],
            'confidence': float(np.max(xgb_proba)),
            'probabilities': {
                self.class_names[i]: float(prob) 
                for i, prob in enumerate(xgb_proba)
            }
        }
        
        result['model_predictions']['random_forest'] = {
            'class': self.class_names[np.argmax(rf_proba)],
            'confidence': float(np.max(rf_proba)),
            'probabilities': {
                self.class_names[i]: float(prob) 
                for i, prob in enumerate(rf_proba)
            }
        }
        
        # CNN предсказание (если доступно)
        if self.cnn_model is not None and TENSORFLOW_AVAILABLE:
            # Подготовка последовательности
            if len(sequence) >= self.sequence_length:
                seq_input = sequence[:self.sequence_length]
            else:
                seq_input = np.zeros(self.sequence_length)
                seq_input[:len(sequence)] = sequence
            
            cnn_proba = self.cnn_model.predict(seq_input.reshape(1, -1))[0]
            
            result['model_predictions']['cnn'] = {
                'class': self.class_names[np.argmax(cnn_proba)],
                'confidence': float(np.max(cnn_proba)),
                'probabilities': {
                    self.class_names[i]: float(prob) 
                    for i, prob in enumerate(cnn_proba)
                }
            }
        
        return result
    
    def get_feature_importance(self) -> Dict[str, List[float]]:
        """
        Получение важности признаков
        """
        importance = {}
        
        if self.rf_model is not None:
            importance['random_forest'] = self.rf_model.feature_importances_.tolist()
        
        if self.xgb_model is not None:
            importance['xgboost'] = self.xgb_model.feature_importances_.tolist()
        
        return importance
    
    def save_model(self, filepath: str):
        """
        Сохранение модели
        """
        model_data = {
            'ensemble_model': self.ensemble_model,
            'xgb_model': self.xgb_model,
            'rf_model': self.rf_model,
            'feature_scaler': self.feature_scaler,
            'label_encoder': self.label_encoder,
            'sequence_length': self.sequence_length,
            'n_classes': self.n_classes,
            'class_names': self.class_names,
            'training_metrics': self.training_metrics
        }
        
        # Сохраняем основные модели
        joblib.dump(model_data, filepath)
        
        # Сохраняем CNN отдельно (если есть)
        if self.cnn_model is not None and TENSORFLOW_AVAILABLE:
            cnn_path = filepath.replace('.joblib', '_cnn.h5')
            self.cnn_model.model.save(cnn_path)
            
            # Сохраняем scaler для CNN
            cnn_scaler_path = filepath.replace('.joblib', '_cnn_scaler.joblib')
            joblib.dump(self.cnn_model.scaler, cnn_scaler_path)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Загрузка модели
        """
        # Загружаем основные модели
        model_data = joblib.load(filepath)
        
        self.ensemble_model = model_data['ensemble_model']
        self.xgb_model = model_data['xgb_model']
        self.rf_model = model_data['rf_model']
        self.feature_scaler = model_data['feature_scaler']
        self.label_encoder = model_data['label_encoder']
        self.sequence_length = model_data['sequence_length']
        self.n_classes = model_data['n_classes']
        self.class_names = model_data['class_names']
        self.training_metrics = model_data['training_metrics']
        
        # Загружаем CNN (если есть)
        cnn_path = filepath.replace('.joblib', '_cnn.h5')
        cnn_scaler_path = filepath.replace('.joblib', '_cnn_scaler.joblib')
        
        if Path(cnn_path).exists() and TENSORFLOW_AVAILABLE:
            self.cnn_model = ExoplanetCNN(self.sequence_length, self.n_classes)
            self.cnn_model.model = tf.keras.models.load_model(cnn_path)
            
            if Path(cnn_scaler_path).exists():
                self.cnn_model.scaler = joblib.load(cnn_scaler_path)
        
        logger.info(f"Model loaded from {filepath}")


class ModelEvaluator:
    """
    Класс для оценки качества моделей
    """
    
    @staticmethod
    def evaluate_model(y_true: np.ndarray, 
                      y_pred: np.ndarray, 
                      y_proba: np.ndarray,
                      class_names: List[str]) -> Dict[str, Any]:
        """
        Комплексная оценка модели
        
        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            y_proba: Вероятности классов
            class_names: Названия классов
            
        Returns:
            Метрики качества
        """
        metrics = {}
        
        # Базовые метрики
        metrics['accuracy'] = float(np.mean(y_true == y_pred))
        
        # Classification report
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names, 
                                     output_dict=True)
        metrics['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # AUC score
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
            metrics['auc_score'] = float(auc)
        except:
            metrics['auc_score'] = None
        
        # Per-class metrics
        metrics['per_class_metrics'] = {}
        for i, class_name in enumerate(class_names):
            if class_name in report:
                metrics['per_class_metrics'][class_name] = {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1_score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                }
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, 
                            class_names: List[str],
                            save_path: Optional[str] = None):
        """
        Построение confusion matrix
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")


# Утилитарные функции
def create_synthetic_training_data(n_samples: int = 1000) -> Tuple[List[Dict], List[np.ndarray], List[str]]:
    """
    Создание синтетических данных для демонстрации
    (В реальности используются данные Kepler/TESS)
    """
    np.random.seed(42)
    
    features = []
    sequences = []
    labels = []
    
    class_distribution = [0.3, 0.4, 0.3]  # CANDIDATE, PC, FP
    
    for i in range(n_samples):
        # Выбираем класс
        class_idx = np.random.choice(3, p=class_distribution)
        class_name = ['CANDIDATE', 'PC', 'FP'][class_idx]
        
        # Генерируем признаки в зависимости от класса
        if class_name == 'CANDIDATE':
            # Настоящие экзопланеты
            feature_dict = {
                'transit_depth': np.random.normal(0.01, 0.005),
                'transit_snr': np.random.normal(15, 5),
                'transit_duration': np.random.normal(4, 1),
                'flux_std': np.random.normal(0.001, 0.0005),
                'v_shape_score': np.random.normal(0.8, 0.2),
                'secondary_eclipse_depth': np.random.normal(0.0001, 0.0001)
            }
            # Генерируем кривую блеска с четким транзитом
            sequence = np.random.normal(1.0, 0.001, 64)
            sequence[28:36] -= feature_dict['transit_depth']
            
        elif class_name == 'PC':
            # Планетарные кандидаты
            feature_dict = {
                'transit_depth': np.random.normal(0.005, 0.003),
                'transit_snr': np.random.normal(8, 3),
                'transit_duration': np.random.normal(3, 1.5),
                'flux_std': np.random.normal(0.002, 0.001),
                'v_shape_score': np.random.normal(0.6, 0.3),
                'secondary_eclipse_depth': np.random.normal(0.0001, 0.0002)
            }
            # Менее четкий транзит
            sequence = np.random.normal(1.0, 0.002, 64)
            sequence[30:34] -= feature_dict['transit_depth']
            
        else:  # FP
            # Ложные позитивы
            feature_dict = {
                'transit_depth': np.random.normal(0.008, 0.004),
                'transit_snr': np.random.normal(5, 2),
                'transit_duration': np.random.normal(2, 1),
                'flux_std': np.random.normal(0.005, 0.002),
                'v_shape_score': np.random.normal(0.3, 0.2),
                'secondary_eclipse_depth': np.random.normal(0.001, 0.0005)
            }
            # Шумный сигнал или систематика
            sequence = np.random.normal(1.0, 0.005, 64)
            if np.random.random() > 0.5:
                # Добавляем систематический тренд
                sequence += np.linspace(-0.01, 0.01, 64)
        
        features.append(feature_dict)
        sequences.append(sequence)
        labels.append(class_name)
    
    return features, sequences, labels
