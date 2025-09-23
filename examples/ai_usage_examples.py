"""
AI Module Usage Examples

Примеры использования AI модуля для анализа кривых блеска и обнаружения экзопланет.
"""

import asyncio
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Импорты AI модуля
from backend.ai.models import CNNClassifier, LSTMClassifier, TransformerClassifier
from backend.ai.ensemble import EnsembleClassifier, create_default_ensemble
from backend.ai.trainer import ModelTrainer, TransitDataset
from backend.ai.predictor import TransitPredictor, AIAssistant
from backend.ai.embeddings import EmbeddingManager
from backend.ai.database import DatabaseManager
from backend.ai.config import AIConfig
from backend.ai.utils import (
    normalize_lightcurve, remove_outliers, resample_lightcurve,
    fold_lightcurve, validate_lightcurve_data
)

def example_1_basic_model_usage():
    """
    Пример 1: Базовое использование моделей
    """
    print("=== Пример 1: Базовое использование моделей ===")
    
    # Создание синтетических данных
    np.random.seed(42)
    lightcurve_length = 1024
    time = np.linspace(0, 27.4, lightcurve_length)  # 27.4 дня (сектор TESS)
    
    # Создание кривой блеска с транзитом
    flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))
    
    # Добавляем транзит
    period = 3.5  # дни
    transit_depth = 0.01  # 1% глубина
    transit_duration = 0.1  # дни
    
    for i in range(int(27.4 / period)):
        transit_center = i * period + 1.0
        transit_mask = np.abs(time - transit_center) < transit_duration / 2
        flux[transit_mask] -= transit_depth
    
    # Создание и использование CNN модели
    print("Создание CNN модели...")
    cnn_model = CNNClassifier(
        input_size=lightcurve_length,
        num_classes=2,
        num_filters=(32, 64, 128),
        use_attention=True
    )
    
    # Предсказание
    cnn_model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(flux).unsqueeze(0)
        logits = cnn_model(input_tensor)
        probabilities = torch.softmax(logits, dim=-1)
        
    print(f"CNN предсказание: {probabilities[0].numpy()}")
    print(f"Вероятность транзита: {probabilities[0, 1].item():.3f}")
    
    # Извлечение признаков
    features = cnn_model.extract_features(input_tensor)
    print(f"Размер вектора признаков: {features.shape}")

def example_2_ensemble_usage():
    """
    Пример 2: Использование ансамбля моделей
    """
    print("\n=== Пример 2: Использование ансамбля моделей ===")
    
    # Создание ансамбля
    ensemble = create_default_ensemble(
        input_size=1024,
        num_classes=2,
        device='cpu'
    )
    
    # Синтетические данные
    batch_size = 4
    sequence_length = 1024
    test_data = torch.randn(batch_size, sequence_length)
    
    # Предсказание с оценкой неопределенности
    predictions, uncertainties, individual_preds = ensemble.predict_with_uncertainty(test_data)
    
    print(f"Ансамбль предсказания: {predictions}")
    print(f"Неопределенности: {uncertainties}")
    print(f"Индивидуальные модели: {list(individual_preds.keys())}")
    
    # Вклад каждой модели
    contributions = ensemble.get_model_contributions(test_data)
    print(f"Вклад моделей: {contributions}")

def example_3_training_workflow():
    """
    Пример 3: Процесс обучения модели
    """
    print("\n=== Пример 3: Процесс обучения модели ===")
    
    # Создание синтетического датасета
    def create_synthetic_dataset(n_samples=1000, sequence_length=1024):
        lightcurves = []
        labels = []
        
        for i in range(n_samples):
            # Базовая кривая блеска
            time = np.linspace(0, 27.4, sequence_length)
            flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))
            
            # 50% вероятность транзита
            has_transit = np.random.random() > 0.5
            
            if has_transit:
                # Добавляем транзит
                period = np.random.uniform(1.0, 10.0)
                depth = np.random.uniform(0.001, 0.02)
                duration = np.random.uniform(0.05, 0.3)
                
                n_transits = int(27.4 / period)
                for j in range(n_transits):
                    center = j * period + np.random.uniform(0.5, period - 0.5)
                    if center < 27.4:
                        mask = np.abs(time - center) < duration / 2
                        flux[mask] -= depth
                
                labels.append(1)  # Транзит
            else:
                labels.append(0)  # Нет транзита
            
            lightcurves.append(flux)
        
        return np.array(lightcurves), np.array(labels)
    
    # Создание данных
    print("Создание синтетического датасета...")
    train_lightcurves, train_labels = create_synthetic_dataset(800)
    val_lightcurves, val_labels = create_synthetic_dataset(200)
    
    # Создание DataLoader'ов
    train_dataset = TransitDataset(train_lightcurves, train_labels, augment=True)
    val_dataset = TransitDataset(val_lightcurves, val_labels, augment=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Создание модели и тренера
    model = CNNClassifier(input_size=1024, num_classes=2)
    trainer = ModelTrainer(
        model=model,
        device='cpu',
        experiment_name='synthetic_transit_detection',
        use_wandb=False,
        use_mlflow=False
    )
    
    # Обучение (короткое для примера)
    print("Начало обучения...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        learning_rate=1e-3,
        early_stopping_patience=3
    )
    
    print(f"Финальная точность: {history['val_acc'][-1]:.3f}")

def example_4_predictor_and_assistant():
    """
    Пример 4: Использование предиктора и AI ассистента
    """
    print("\n=== Пример 4: Предиктор и AI ассистент ===")
    
    # Создание компонентов
    model = CNNClassifier(input_size=1024, num_classes=2)
    embedding_manager = EmbeddingManager(embedding_dim=256)
    predictor = TransitPredictor(model, embedding_manager, device='cpu')
    assistant = AIAssistant()
    
    # Синтетическая кривая блеска с транзитом
    time = np.linspace(0, 27.4, 1024)
    flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))
    
    # Добавляем явный транзит
    period = 4.2
    depth = 0.015
    duration = 0.12
    
    for i in range(int(27.4 / period)):
        center = i * period + 2.1
        if center < 27.4:
            mask = np.abs(time - center) < duration / 2
            flux[mask] -= depth
    
    # Предсказание
    prediction = predictor.predict(
        lightcurve=flux,
        target_name="Synthetic Target 1",
        stellar_params={'radius': 1.0, 'temperature': 5778}
    )
    
    print(f"Обнаружен транзит: {prediction.is_transit}")
    print(f"Уверенность: {prediction.confidence:.3f}")
    print(f"Уровень уверенности: {prediction.confidence_level.value}")
    print(f"Объяснение: {prediction.explanation}")
    
    # AI ассистент
    beginner_explanation = assistant.explain_for_beginners(prediction, "Synthetic Target 1")
    print(f"\nОбъяснение для начинающих:\n{beginner_explanation}")
    
    comparison = assistant.compare_with_known_planets(prediction)
    print(f"\nСравнение с известными планетами:\n{comparison}")
    
    habitability = assistant.explain_habitability(prediction)
    print(f"\nОценка обитаемости:\n{habitability}")

async def example_5_database_integration():
    """
    Пример 5: Интеграция с базой данных
    """
    print("\n=== Пример 5: Интеграция с базой данных ===")
    
    # Примечание: Для работы требуется PostgreSQL
    try:
        db_manager = DatabaseManager(
            database_url="postgresql://user:password@localhost/exoplanet_ai_test"
        )
        
        await db_manager.initialize()
        print("База данных инициализирована")
        
        # Создание примера результата анализа
        from backend.ai.database import AnalysisResult
        
        analysis_result = AnalysisResult(
            target_name="Test Target",
            analysis_timestamp=datetime.now(),
            model_version="1.0.0",
            is_transit=True,
            confidence=0.85,
            transit_probability=0.85,
            physical_parameters={
                'period': 4.2,
                'depth': 1500,  # ppm
                'duration': 2.9,  # hours
                'planet_radius': 1.2  # Earth radii
            },
            bls_parameters={
                'best_period': 4.2,
                'best_power': 0.75,
                'snr': 9.2
            }
        )
        
        # Сохранение результата
        result_id = await db_manager.save_analysis_result(analysis_result)
        print(f"Результат сохранен с ID: {result_id}")
        
        # Получение статистики
        stats = await db_manager.get_statistics()
        print(f"Статистика БД: {stats}")
        
        await db_manager.close()
        
    except Exception as e:
        print(f"Ошибка подключения к БД: {e}")
        print("Убедитесь, что PostgreSQL запущен и настроен")

def example_6_data_preprocessing():
    """
    Пример 6: Предобработка данных
    """
    print("\n=== Пример 6: Предобработка данных ===")
    
    # Создание зашумленной кривой блеска
    time = np.linspace(0, 27.4, 2000)
    flux = np.ones_like(time) + np.random.normal(0, 0.002, len(time))
    
    # Добавляем выбросы
    outlier_indices = np.random.choice(len(flux), size=20, replace=False)
    flux[outlier_indices] += np.random.uniform(0.01, 0.05, 20)
    
    print(f"Исходные данные: {len(time)} точек")
    
    # Валидация данных
    validation = validate_lightcurve_data(time, flux)
    print(f"Валидация: {validation['is_valid']}")
    if validation['warnings']:
        print(f"Предупреждения: {validation['warnings']}")
    print(f"Статистика: {validation['statistics']}")
    
    # Нормализация
    flux_normalized = normalize_lightcurve(flux, method='median')
    print(f"Медиана после нормализации: {np.median(flux_normalized):.6f}")
    
    # Удаление выбросов
    flux_clean, outlier_mask = remove_outliers(flux_normalized, threshold=1.5)
    print(f"Удалено выбросов: {np.sum(~outlier_mask)}")
    
    # Ресэмплинг до стандартной длины
    time_resampled, flux_resampled = resample_lightcurve(
        time, flux_clean, target_length=1024
    )
    print(f"Данные после ресэмплинга: {len(time_resampled)} точек")
    
    # Фолдинг по периоду
    test_period = 3.5
    phase, flux_folded = fold_lightcurve(time_resampled, flux_resampled, test_period)
    print(f"Фолдинг по периоду {test_period} дней выполнен")

def example_7_embedding_management():
    """
    Пример 7: Управление embeddings
    """
    print("\n=== Пример 7: Управление embeddings ===")
    
    # Создание менеджера embeddings
    embedding_manager = EmbeddingManager(
        embedding_dim=256,
        similarity_threshold=0.9,
        max_cache_size=1000,
        use_faiss=False  # Используем sklearn для простоты
    )
    
    # Создание примеров embeddings
    np.random.seed(42)
    
    targets = [
        "TIC 441420236", "KIC 8462852", "EPIC 249622103",
        "TIC 307210830", "KIC 12557548"
    ]
    
    # Симуляция кэширования результатов
    for i, target in enumerate(targets):
        # Создаем случайный embedding
        embedding = np.random.randn(256)
        
        # Создаем mock предсказание
        mock_prediction = {
            'is_transit': i % 2 == 0,
            'confidence': np.random.uniform(0.6, 0.95),
            'explanation': f"Mock prediction for {target}"
        }
        
        # Кэшируем
        embedding_manager.cache_prediction(
            target_name=target,
            embedding=embedding,
            prediction_result=mock_prediction,
            model_version="1.0.0"
        )
    
    print(f"Закэшировано {len(targets)} результатов")
    
    # Поиск похожих целей
    query_embedding = np.random.randn(256)
    similar_targets = embedding_manager.find_similar_targets(
        embedding=query_embedding,
        top_k=3,
        min_similarity=0.1  # Низкий порог для демонстрации
    )
    
    print("Похожие цели:")
    for target_name, similarity, prediction in similar_targets:
        print(f"  {target_name}: схожесть {similarity:.3f}")
    
    # Статистика
    stats = embedding_manager.get_embedding_statistics()
    print(f"Статистика embeddings: {stats}")

def example_8_configuration():
    """
    Пример 8: Работа с конфигурацией
    """
    print("\n=== Пример 8: Работа с конфигурацией ===")
    
    # Просмотр текущей конфигурации
    print(f"Устройство: {AIConfig.DEVICE}")
    print(f"CNN конфигурация: {AIConfig.CNN_CONFIG}")
    print(f"Конфигурация обучения: {AIConfig.TRAINING_CONFIG}")
    
    # Обновление конфигурации
    AIConfig.update_config({
        'cnn_config': {
            'num_filters': (64, 128, 256, 512),
            'dropout': 0.2,
            'use_attention': True
        }
    })
    
    print(f"Обновленная CNN конфигурация: {AIConfig.CNN_CONFIG}")
    
    # Получение конфигурации для конкретной модели
    lstm_config = AIConfig.get_model_config('lstm')
    print(f"LSTM конфигурация: {lstm_config}")
    
    # Создание директорий
    AIConfig.create_directories()
    print("Необходимые директории созданы")

def example_9_model_comparison():
    """
    Пример 9: Сравнение производительности моделей
    """
    print("\n=== Пример 9: Сравнение производительности моделей ===")
    
    # Создание тестовых данных
    test_data = torch.randn(10, 1024)
    
    models = {
        'CNN': CNNClassifier(input_size=1024, num_classes=2),
        'LSTM': LSTMClassifier(input_size=1024, num_classes=2),
        'Transformer': TransformerClassifier(input_size=1024, num_classes=2)
    }
    
    # Тестирование каждой модели
    import time
    
    for model_name, model in models.items():
        model.eval()
        
        # Измерение времени инференса
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):  # 10 прогонов для усреднения
                predictions = model(test_data)
        
        avg_time = (time.time() - start_time) / 10
        
        # Подсчет параметров
        param_count = model.count_parameters()
        
        print(f"{model_name}:")
        print(f"  Время инференса: {avg_time*1000:.1f} мс")
        print(f"  Параметры: {param_count['total']:,}")
        print(f"  Размер модели: ~{param_count['total']*4/1024/1024:.1f} МБ")

def main():
    """
    Запуск всех примеров
    """
    print("🧠 AI Module Usage Examples")
    print("=" * 50)
    
    # Запуск примеров
    example_1_basic_model_usage()
    example_2_ensemble_usage()
    example_3_training_workflow()
    example_4_predictor_and_assistant()
    
    # Асинхронный пример
    asyncio.run(example_5_database_integration())
    
    example_6_data_preprocessing()
    example_7_embedding_management()
    example_8_configuration()
    example_9_model_comparison()
    
    print("\n✅ Все примеры выполнены!")
    print("\nДля использования в продакшене:")
    print("1. Настройте PostgreSQL базу данных")
    print("2. Обучите модели на реальных данных")
    print("3. Настройте мониторинг и логирование")
    print("4. Оптимизируйте для вашего hardware")

if __name__ == "__main__":
    main()
