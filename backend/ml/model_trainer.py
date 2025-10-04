"""
Model Training and Validation System
Система обучения и валидации моделей

Включает:
- Автоматическое обучение на исторических данных
- Кросс-валидацию и гиперпараметрическую оптимизацию
- Метрики качества и визуализацию результатов
- Сохранение и загрузка моделей
- Мониторинг производительности
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ML библиотеки
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Наши модули
from .lightcurve_preprocessor import LightCurvePreprocessor
from .feature_extractor import ExoplanetFeatureExtractor
from .exoplanet_classifier import ExoplanetEnsembleClassifier, ModelEvaluator

logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """
    Комплексный пайплайн обучения моделей
    """
    
    def __init__(self, 
                 models_dir: str = "models",
                 data_dir: str = "data",
                 results_dir: str = "results"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        
        # Создаем директории
        for dir_path in [self.models_dir, self.data_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Компоненты пайплайна
        self.preprocessor = LightCurvePreprocessor()
        self.feature_extractor = ExoplanetFeatureExtractor()
        self.classifier = ExoplanetEnsembleClassifier()
        
        # Результаты обучения
        self.training_history = []
        self.best_model_path = None
        self.best_metrics = None
        
    async def load_training_data(self, 
                               data_source: str = "kepler",
                               max_samples: Optional[int] = None) -> Tuple[List[Dict], List[np.ndarray], List[str]]:
        """
        Загрузка данных для обучения
        
        Args:
            data_source: Источник данных ('kepler', 'tess', 'synthetic')
            max_samples: Максимальное количество образцов
            
        Returns:
            features, sequences, labels
        """
        logger.info(f"Loading training data from {data_source}")
        
        if data_source == "synthetic":
            # Используем синтетические данные для демонстрации
            from .exoplanet_classifier import create_synthetic_training_data
            return create_synthetic_training_data(max_samples or 1000)
        
        elif data_source == "kepler":
            # В реальности здесь загружались бы данные Kepler
            return await self._load_kepler_data(max_samples)
        
        elif data_source == "tess":
            # В реальности здесь загружались бы данные TESS
            return await self._load_tess_data(max_samples)
        
        else:
            raise ValueError(f"Unknown data source: {data_source}")
    
    async def _load_kepler_data(self, max_samples: Optional[int]) -> Tuple[List[Dict], List[np.ndarray], List[str]]:
        """
        Загрузка данных Kepler (заглушка для реальной реализации)
        """
        logger.warning("Kepler data loading not implemented, using synthetic data")
        from .exoplanet_classifier import create_synthetic_training_data
        return create_synthetic_training_data(max_samples or 1000)
    
    async def _load_tess_data(self, max_samples: Optional[int]) -> Tuple[List[Dict], List[np.ndarray], List[str]]:
        """
        Загрузка данных TESS (заглушка для реальной реализации)
        """
        logger.warning("TESS data loading not implemented, using synthetic data")
        from .exoplanet_classifier import create_synthetic_training_data
        return create_synthetic_training_data(max_samples or 1000)
    
    def optimize_hyperparameters(self, 
                                X_train: np.ndarray, 
                                y_train: np.ndarray,
                                model_type: str = "xgboost",
                                n_iter: int = 50) -> Dict[str, Any]:
        """
        Оптимизация гиперпараметров
        
        Args:
            X_train, y_train: Обучающие данные
            model_type: Тип модели для оптимизации
            n_iter: Количество итераций поиска
            
        Returns:
            Лучшие параметры и метрики
        """
        logger.info(f"Optimizing hyperparameters for {model_type}")
        
        if model_type == "xgboost":
            from xgboost import XGBClassifier
            
            model = XGBClassifier(random_state=42, eval_metric='mlogloss')
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
        else:
            raise ValueError(f"Hyperparameter optimization not implemented for {model_type}")
        
        # Используем RandomizedSearchCV для эффективности
        search = RandomizedSearchCV(
            model, param_grid, 
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1_macro',
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X_train, y_train)
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'cv_results': search.cv_results_
        }
    
    async def train_with_validation(self,
                                  features: List[Dict[str, float]],
                                  sequences: List[np.ndarray],
                                  labels: List[str],
                                  validation_strategy: str = "holdout",
                                  test_size: float = 0.2,
                                  cv_folds: int = 5,
                                  optimize_hyperparams: bool = True) -> Dict[str, Any]:
        """
        Обучение с валидацией
        
        Args:
            features, sequences, labels: Данные для обучения
            validation_strategy: 'holdout', 'cross_validation', 'time_series'
            test_size: Размер тестовой выборки
            cv_folds: Количество фолдов для кросс-валидации
            optimize_hyperparams: Оптимизировать гиперпараметры
            
        Returns:
            Результаты обучения и валидации
        """
        logger.info(f"Starting training with {validation_strategy} validation")
        
        # Подготовка данных
        X_tabular = self.classifier._prepare_tabular_data(features)
        X_sequences = self.classifier._prepare_sequence_data(sequences)
        y_encoded = self.classifier.label_encoder.fit_transform(labels)
        
        results = {
            'training_start_time': datetime.now().isoformat(),
            'data_info': {
                'total_samples': len(features),
                'n_features': X_tabular.shape[1],
                'sequence_length': X_sequences.shape[1],
                'class_distribution': dict(zip(*np.unique(labels, return_counts=True)))
            },
            'models': {},
            'validation_strategy': validation_strategy
        }
        
        if validation_strategy == "holdout":
            # Простое разделение train/test
            (X_tab_train, X_tab_test, 
             X_seq_train, X_seq_test, 
             y_train, y_test) = train_test_split(
                X_tabular, X_sequences, y_encoded,
                test_size=test_size, 
                stratify=y_encoded,
                random_state=42
            )
            
            # Оптимизация гиперпараметров (если включена)
            if optimize_hyperparams:
                logger.info("Optimizing hyperparameters")
                
                # XGBoost оптимизация
                xgb_optimization = self.optimize_hyperparameters(
                    X_tab_train, y_train, "xgboost"
                )
                results['hyperparameter_optimization'] = {
                    'xgboost': xgb_optimization
                }
                
                # Используем оптимизированные параметры
                self.classifier.xgb_model = xgb_optimization['best_estimator']
            
            # Обучение моделей
            training_metrics = self.classifier.train(
                features=features,
                sequences=sequences,
                labels=labels,
                test_size=test_size,
                cv_folds=cv_folds
            )
            
            results['models'] = training_metrics
            
        elif validation_strategy == "cross_validation":
            # Кросс-валидация
            cv_results = await self._cross_validation_training(
                X_tabular, X_sequences, y_encoded, cv_folds
            )
            results['cross_validation'] = cv_results
            
        elif validation_strategy == "time_series":
            # Временная валидация (для временных рядов)
            ts_results = await self._time_series_validation(
                features, sequences, labels
            )
            results['time_series_validation'] = ts_results
        
        # Оценка финальной модели
        final_evaluation = await self._evaluate_final_model(
            X_tabular, X_sequences, y_encoded, test_size
        )
        results['final_evaluation'] = final_evaluation
        
        # Сохранение результатов
        results['training_end_time'] = datetime.now().isoformat()
        self.training_history.append(results)
        
        # Сохранение лучшей модели
        model_path = self.models_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        self.classifier.save_model(str(model_path))
        results['model_path'] = str(model_path)
        
        # Сохранение результатов в JSON
        results_path = self.results_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            # Преобразуем numpy объекты для JSON сериализации
            json_results = self._convert_for_json(results)
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Training completed. Results saved to {results_path}")
        return results
    
    async def _cross_validation_training(self,
                                       X_tabular: np.ndarray,
                                       X_sequences: np.ndarray,
                                       y: np.ndarray,
                                       cv_folds: int) -> Dict[str, Any]:
        """
        Кросс-валидация
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = {
            'fold_results': [],
            'mean_metrics': {},
            'std_metrics': {}
        }
        
        fold_scores = {'accuracy': [], 'f1_macro': [], 'auc': []}
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_tabular, y)):
            logger.info(f"Training fold {fold + 1}/{cv_folds}")
            
            # Разделение данных
            X_tab_train, X_tab_val = X_tabular[train_idx], X_tabular[val_idx]
            X_seq_train, X_seq_val = X_sequences[train_idx], X_sequences[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Создание временного классификатора для этого фолда
            fold_classifier = ExoplanetEnsembleClassifier()
            
            # Обучение (упрощенное для кросс-валидации)
            fold_classifier.feature_scaler.fit(X_tab_train)
            X_tab_train_scaled = fold_classifier.feature_scaler.transform(X_tab_train)
            X_tab_val_scaled = fold_classifier.feature_scaler.transform(X_tab_val)
            
            # Обучение XGBoost
            import xgboost as xgb
            fold_classifier.xgb_model = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, eval_metric='mlogloss'
            )
            fold_classifier.xgb_model.fit(X_tab_train_scaled, y_train)
            
            # Предсказания
            y_pred = fold_classifier.xgb_model.predict(X_tab_val_scaled)
            y_proba = fold_classifier.xgb_model.predict_proba(X_tab_val_scaled)
            
            # Метрики
            accuracy = np.mean(y_pred == y_val)
            f1_macro = f1_score(y_val, y_pred, average='macro')
            auc = roc_auc_score(y_val, y_proba, multi_class='ovr')
            
            fold_scores['accuracy'].append(accuracy)
            fold_scores['f1_macro'].append(f1_macro)
            fold_scores['auc'].append(auc)
            
            fold_result = {
                'fold': fold + 1,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'auc_score': auc,
                'classification_report': classification_report(y_val, y_pred, output_dict=True)
            }
            cv_results['fold_results'].append(fold_result)
        
        # Средние метрики
        for metric, scores in fold_scores.items():
            cv_results['mean_metrics'][metric] = np.mean(scores)
            cv_results['std_metrics'][metric] = np.std(scores)
        
        return cv_results
    
    async def _time_series_validation(self,
                                    features: List[Dict],
                                    sequences: List[np.ndarray],
                                    labels: List[str]) -> Dict[str, Any]:
        """
        Временная валидация (для временных рядов)
        """
        logger.info("Performing time series validation")
        
        # Сортируем данные по времени (если есть временные метки)
        # В реальности здесь использовались бы реальные временные метки
        n_samples = len(features)
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        
        # Временное разделение
        train_features = features[:train_size]
        train_sequences = sequences[:train_size]
        train_labels = labels[:train_size]
        
        val_features = features[train_size:train_size + val_size]
        val_sequences = sequences[train_size:train_size + val_size]
        val_labels = labels[train_size:train_size + val_size]
        
        test_features = features[train_size + val_size:]
        test_sequences = sequences[train_size + val_size:]
        test_labels = labels[train_size + val_size:]
        
        # Обучение на исторических данных
        temp_classifier = ExoplanetEnsembleClassifier()
        training_metrics = temp_classifier.train(
            train_features, train_sequences, train_labels,
            test_size=0.0  # Не используем дополнительное разделение
        )
        
        # Валидация на более поздних данных
        val_results = []
        for i, (feat, seq, label) in enumerate(zip(val_features, val_sequences, val_labels)):
            pred_result = temp_classifier.predict_single(feat, seq)
            val_results.append({
                'true_label': label,
                'predicted_label': pred_result['predicted_class'],
                'confidence': pred_result['confidence']
            })
        
        # Метрики валидации
        val_true = [r['true_label'] for r in val_results]
        val_pred = [r['predicted_label'] for r in val_results]
        
        val_accuracy = np.mean([t == p for t, p in zip(val_true, val_pred)])
        
        return {
            'training_samples': len(train_features),
            'validation_samples': len(val_features),
            'test_samples': len(test_features),
            'validation_accuracy': val_accuracy,
            'training_metrics': training_metrics,
            'validation_results': val_results
        }
    
    async def _evaluate_final_model(self,
                                  X_tabular: np.ndarray,
                                  X_sequences: np.ndarray,
                                  y: np.ndarray,
                                  test_size: float) -> Dict[str, Any]:
        """
        Финальная оценка модели
        """
        logger.info("Performing final model evaluation")
        
        # Разделение на train/test
        (X_tab_train, X_tab_test, 
         X_seq_train, X_seq_test, 
         y_train, y_test) = train_test_split(
            X_tabular, X_sequences, y,
            test_size=test_size, 
            stratify=y,
            random_state=42
        )
        
        # Предсказания ансамбля
        ensemble_proba = self.classifier.ensemble_model.predict_proba(X_tab_test)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        # Комплексная оценка
        evaluation = ModelEvaluator.evaluate_model(
            y_test, ensemble_pred, ensemble_proba,
            self.classifier.class_names
        )
        
        # Дополнительные метрики
        evaluation['feature_importance'] = self.classifier.get_feature_importance()
        
        return evaluation
    
    def _convert_for_json(self, obj: Any) -> Any:
        """
        Преобразование объектов для JSON сериализации
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj
    
    async def generate_training_report(self, results: Dict[str, Any]) -> str:
        """
        Генерация отчета об обучении
        """
        report_lines = [
            "# ExoplanetAI Model Training Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Training Configuration",
            f"- Validation Strategy: {results['validation_strategy']}",
            f"- Total Samples: {results['data_info']['total_samples']}",
            f"- Features: {results['data_info']['n_features']}",
            f"- Sequence Length: {results['data_info']['sequence_length']}",
            "",
            "## Class Distribution",
        ]
        
        for class_name, count in results['data_info']['class_distribution'].items():
            percentage = (count / results['data_info']['total_samples']) * 100
            report_lines.append(f"- {class_name}: {count} ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "## Model Performance",
        ])
        
        if 'models' in results:
            for model_name, metrics in results['models'].items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    report_lines.extend([
                        f"### {model_name.title()}",
                        f"- Accuracy: {metrics['accuracy']:.3f}",
                        f"- AUC Score: {metrics.get('auc_score', 'N/A')}",
                        ""
                    ])
        
        if 'final_evaluation' in results:
            eval_metrics = results['final_evaluation']
            report_lines.extend([
                "## Final Evaluation",
                f"- Overall Accuracy: {eval_metrics.get('accuracy', 'N/A'):.3f}",
                f"- AUC Score: {eval_metrics.get('auc_score', 'N/A')}",
                ""
            ])
        
        report_lines.extend([
            "## Training Duration",
            f"- Start: {results['training_start_time']}",
            f"- End: {results['training_end_time']}",
            "",
            "---",
            "*Report generated by ExoplanetAI Model Training Pipeline*"
        ])
        
        return "\n".join(report_lines)
    
    async def save_training_report(self, results: Dict[str, Any]) -> str:
        """
        Сохранение отчета об обучении
        """
        report_content = await self.generate_training_report(results)
        report_path = self.results_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Training report saved to {report_path}")
        return str(report_path)


class ModelMonitor:
    """
    Мониторинг производительности моделей
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.performance_history = []
    
    async def evaluate_model_drift(self,
                                 model_path: str,
                                 new_data: Tuple[List[Dict], List[np.ndarray], List[str]]) -> Dict[str, Any]:
        """
        Оценка дрейфа модели на новых данных
        """
        logger.info("Evaluating model drift")
        
        # Загружаем модель
        classifier = ExoplanetEnsembleClassifier()
        classifier.load_model(model_path)
        
        features, sequences, labels = new_data
        
        # Предсказания на новых данных
        predictions = []
        for feat, seq in zip(features, sequences):
            pred = classifier.predict_single(feat, seq)
            predictions.append(pred)
        
        # Анализ производительности
        predicted_labels = [p['predicted_class'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        accuracy = np.mean([true == pred for true, pred in zip(labels, predicted_labels)])
        mean_confidence = np.mean(confidences)
        
        drift_metrics = {
            'accuracy': accuracy,
            'mean_confidence': mean_confidence,
            'low_confidence_ratio': np.mean([c < 0.7 for c in confidences]),
            'class_distribution': dict(zip(*np.unique(predicted_labels, return_counts=True))),
            'evaluation_date': datetime.now().isoformat()
        }
        
        self.performance_history.append(drift_metrics)
        
        return drift_metrics
    
    def detect_performance_degradation(self, threshold: float = 0.05) -> Dict[str, Any]:
        """
        Детекция деградации производительности
        """
        if len(self.performance_history) < 2:
            return {'degradation_detected': False, 'reason': 'Insufficient history'}
        
        recent_accuracy = self.performance_history[-1]['accuracy']
        baseline_accuracy = np.mean([h['accuracy'] for h in self.performance_history[:-1]])
        
        degradation = baseline_accuracy - recent_accuracy
        
        return {
            'degradation_detected': degradation > threshold,
            'degradation_amount': degradation,
            'recent_accuracy': recent_accuracy,
            'baseline_accuracy': baseline_accuracy,
            'threshold': threshold
        }
