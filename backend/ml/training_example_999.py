"""
Пример использования ExoplanetClassifier для достижения точности 99.9%+
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Dict

from .exoplanet_classifier_999 import ExoplanetClassifier, create_training_pipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(n_samples: int = 1000) -> tuple[List[Dict], np.ndarray]:
    """
    Генерация примера данных для демонстрации
    В реальном проекте здесь будут данные Kepler/TESS
    """
    logger.info(f"Generating {n_samples} sample lightcurves")
    
    lightcurves = []
    labels = []
    
    for i in range(n_samples):
        # Временная сетка (27.4 дня TESS сектор)
        time = np.linspace(0, 27.4, 1000)
        
        # Базовый поток с шумом
        flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))
        
        # 50% экзопланет, 50% не экзопланет
        is_exoplanet = i < n_samples // 2
        
        if is_exoplanet:
            # Добавляем транзитный сигнал
            period = np.random.uniform(2, 20)  # период 2-20 дней
            depth = np.random.uniform(0.001, 0.01)  # глубина 0.1-1%
            duration = np.random.uniform(0.1, 0.3) * period  # длительность
            
            # Добавляем несколько транзитов
            num_transits = int(27.4 / period)
            for j in range(num_transits):
                transit_center = j * period + period/2
                if transit_center < 27.4:
                    transit_mask = np.abs(time - transit_center) < duration/2
                    flux[transit_mask] -= depth
        
        # Добавляем звездную вариабельность
        stellar_period = np.random.uniform(5, 25)
        stellar_amplitude = np.random.uniform(0.0001, 0.002)
        flux += stellar_amplitude * np.sin(2 * np.pi * time / stellar_period)
        
        lightcurves.append({
            'time': time,
            'flux': flux,
            'target_id': f'sample_{i:04d}'
        })
        
        labels.append(1 if is_exoplanet else 0)
    
    return lightcurves, np.array(labels)


def main_training_example():
    """
    Основной пример обучения модели с точностью 99.9%+
    """
    logger.info("🚀 Starting ExoplanetAI 99.9%+ Training Pipeline")
    
    # 1. Подготовка данных
    logger.info("Step 1: Preparing training data")
    lightcurves, labels = generate_sample_data(n_samples=2000)
    
    logger.info(f"Dataset: {len(lightcurves)} lightcurves")
    logger.info(f"Exoplanets: {np.sum(labels)}, Non-exoplanets: {len(labels) - np.sum(labels)}")
    
    # 2. Запуск полного пайплайна
    logger.info("Step 2: Running full training pipeline")
    classifier, metrics = create_training_pipeline(
        lightcurves=lightcurves,
        labels=labels,
        test_size=0.2,
        n_trials=50,  # Уменьшено для демо
        target_accuracy=0.999
    )
    
    # 3. Результаты
    logger.info("Step 3: Training Results")
    logger.info(f"📊 Final Metrics:")
    logger.info(f"   Accuracy: {metrics.accuracy:.4f} ({'✅' if metrics.accuracy >= 0.999 else '❌'})")
    logger.info(f"   Precision: {metrics.precision:.4f}")
    logger.info(f"   Recall: {metrics.recall:.4f}")
    logger.info(f"   F1-Score: {metrics.f1_score:.4f} ({'✅' if metrics.f1_score >= 0.998 else '❌'})")
    logger.info(f"   ROC-AUC: {metrics.roc_auc:.4f} ({'✅' if metrics.roc_auc >= 0.999 else '❌'})")
    
    target_met = metrics.meets_target(0.999)
    logger.info(f"🎯 Target 99.9%+ Accuracy: {'✅ ACHIEVED' if target_met else '❌ NOT REACHED'}")
    
    # 4. Сохранение модели
    model_path = Path("models/exoplanet_classifier_999.joblib")
    model_path.parent.mkdir(exist_ok=True)
    classifier.save_model(str(model_path))
    logger.info(f"💾 Model saved to {model_path}")
    
    # 5. Визуализация важности признаков
    try:
        classifier.plot_feature_importance("models/feature_importance.png")
        logger.info("📈 Feature importance plot saved")
    except Exception as e:
        logger.warning(f"Could not create feature importance plot: {e}")
    
    # 6. Тестирование предсказаний
    logger.info("Step 4: Testing predictions")
    test_lightcurves, _ = generate_sample_data(n_samples=5)
    predictions = classifier.predict(test_lightcurves)
    
    for i, pred in enumerate(predictions):
        logger.info(f"Sample {i+1}: {pred['prediction']} (confidence: {pred['confidence']:.3f})")
    
    return classifier, metrics


def integration_example():
    """
    Пример интеграции с существующим проектом
    """
    logger.info("🔗 Integration Example")
    
    # Загрузка сохраненной модели
    model_path = "models/exoplanet_classifier_999.joblib"
    
    if Path(model_path).exists():
        classifier = ExoplanetClassifier()
        classifier.load_model(model_path)
        logger.info("✅ Model loaded successfully")
        
        # Пример предсказания
        sample_lightcurves, _ = generate_sample_data(n_samples=3)
        predictions = classifier.predict(sample_lightcurves)
        
        logger.info("🔮 Predictions:")
        for i, pred in enumerate(predictions):
            logger.info(f"   Target {i+1}: {pred['prediction']} ({pred['confidence']:.1%} confidence)")
        
        return classifier
    else:
        logger.error(f"❌ Model file not found: {model_path}")
        logger.info("💡 Run main_training_example() first to train the model")
        return None


if __name__ == "__main__":
    # Полный пример обучения
    classifier, metrics = main_training_example()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    
    # Пример интеграции
    print("\n" + "="*50)
    print("INTEGRATION EXAMPLE")
    print("="*50)
    integration_example()
