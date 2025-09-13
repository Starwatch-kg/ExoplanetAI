#!/usr/bin/env python3
"""
Пример использования системы поиска экзопланет
Демонстрирует основные возможности системы
"""

import sys
import os
from pathlib import Path

# Добавляем src в путь для импортов
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from src.tess_data_loader import TESSDataLoader
from src.hybrid_transit_search import HybridTransitSearch
from src.representation_learning import SelfSupervisedRepresentationLearner, create_representation_dataset
from src.anomaly_ensemble import AnomalyEnsemble, create_anomaly_dataset
from src.results_exporter import ResultsExporter, ExoplanetCandidate

def create_synthetic_lightcurve_data():
    """
    Создает синтетические данные кривых блеска для демонстрации
    """
    print("Создание синтетических данных кривых блеска...")
    
    np.random.seed(42)
    lightcurves = []
    tic_ids = []
    
    # Параметры для генерации данных
    num_stars = 10
    time_length = 2000
    time_range = 30  # дней
    
    for i in range(num_stars):
        # Создание временной шкалы
        times = np.linspace(0, time_range, time_length)
        
        # Базовый поток с шумом
        base_flux = 1.0 + 0.01 * np.random.randn(time_length)
        
        # Добавление периодических вариаций звезды
        stellar_variation = 0.005 * np.sin(2 * np.pi * times / 5.0)  # 5-дневный период
        base_flux += stellar_variation
        
        # Случайно добавляем транзиты
        if np.random.rand() < 0.4:  # 40% вероятность транзита
            # Параметры транзита
            period = np.random.uniform(3, 20)  # период в днях
            depth = np.random.uniform(0.005, 0.03)  # глубина транзита
            duration = np.random.uniform(0.1, 0.5)  # длительность в днях
            t0 = np.random.uniform(0, period)  # время первого транзита
            
            # Создание транзитов
            for t in times:
                phase = (t - t0) % period
                if phase < duration or phase > (period - duration):
                    # Простая прямоугольная модель транзита
                    base_flux[int(t * time_length / time_range)] -= depth
        
        lightcurves.append((times, base_flux))
        tic_ids.append(f"TIC_{1000000 + i}")
    
    print(f"Создано {len(lightcurves)} синтетических кривых блеска")
    return lightcurves, tic_ids

def demonstrate_data_loading():
    """Демонстрация загрузки данных"""
    print("\n" + "="*50)
    print("ДЕМОНСТРАЦИЯ ЗАГРУЗКИ ДАННЫХ")
    print("="*50)
    
    # Создание загрузчика данных
    loader = TESSDataLoader(cache_dir="demo_cache")
    
    # Создание синтетических данных
    lightcurves, tic_ids = create_synthetic_lightcurve_data()
    
    # Сохранение данных в кэш
    for (times, fluxes), tic_id in zip(lightcurves, tic_ids):
        metadata = {
            'tic_id': tic_id,
            'ra': np.random.uniform(0, 360),
            'dec': np.random.uniform(-90, 90),
            'tmag': np.random.uniform(8, 15)
        }
        loader.save_lightcurve(times, fluxes, f"{tic_id}_lightcurve.csv", metadata)
    
    print(f"Данные сохранены в кэш для {len(tic_ids)} звезд")
    return lightcurves, tic_ids

def demonstrate_hybrid_search(lightcurves, tic_ids):
    """Демонстрация гибридного поиска транзитов"""
    print("\n" + "="*50)
    print("ДЕМОНСТРАЦИЯ ГИБРИДНОГО ПОИСКА ТРАНЗИТОВ")
    print("="*50)
    
    # Инициализация гибридного поиска
    hybrid_search = HybridTransitSearch()
    
    all_candidates = []
    
    for i, ((times, fluxes), tic_id) in enumerate(zip(lightcurves[:3], tic_ids[:3])):
        print(f"Анализ {i+1}/3: {tic_id}")
        
        try:
            # Поиск транзитов
            results = hybrid_search.search_transits(times, fluxes)
            
            print(f"  Найдено кандидатов: {len(results['candidates'])}")
            
            # Создание кандидатов
            for candidate_data in results['candidates']:
                candidate = ExoplanetCandidate(
                    tic_id=tic_id,
                    period=candidate_data['period'],
                    depth=candidate_data['depth'],
                    duration=candidate_data['duration'],
                    start_time=candidate_data['start_time'],
                    end_time=candidate_data['end_time'],
                    confidence=candidate_data['confidence'],
                    combined_score=candidate_data['combined_score'],
                    anomaly_probability=np.random.uniform(0.1, 0.8),  # Случайная оценка
                    star_info={
                        'ra': np.random.uniform(0, 360),
                        'dec': np.random.uniform(-90, 90),
                        'tmag': np.random.uniform(8, 15)
                    }
                )
                all_candidates.append(candidate)
                
        except Exception as e:
            print(f"  Ошибка анализа {tic_id}: {e}")
    
    print(f"Всего найдено кандидатов: {len(all_candidates)}")
    return all_candidates

def demonstrate_representation_learning(lightcurves):
    """Демонстрация обучения представлений"""
    print("\n" + "="*50)
    print("ДЕМОНСТРАЦИЯ ОБУЧЕНИЯ ПРЕДСТАВЛЕНИЙ")
    print("="*50)
    
    # Извлечение потоков
    fluxes_list = [fluxes for _, fluxes in lightcurves]
    
    # Создание DataLoader
    dataloader = create_representation_dataset(fluxes_list, batch_size=8)
    
    # Инициализация обучателя представлений
    learner = SelfSupervisedRepresentationLearner(
        input_length=2000,
        embedding_dim=64,  # Уменьшенный размер для демо
        hidden_dim=128,
        num_layers=2
    )
    
    print("Начинаем обучение представлений...")
    
    # Обучение (сокращенное для демо)
    loss_history = learner.train(dataloader, epochs=10, learning_rate=1e-3)
    
    print(f"Обучение завершено. Финальная потеря: {loss_history[-1]:.4f}")
    
    # Кодирование данных
    embeddings, metadata = learner.encode_dataset(dataloader)
    print(f"Размер представлений: {embeddings.shape}")
    
    return embeddings

def demonstrate_anomaly_detection(embeddings):
    """Демонстрация детекции аномалий"""
    print("\n" + "="*50)
    print("ДЕМОНСТРАЦИЯ ДЕТЕКЦИИ АНОМАЛИЙ")
    print("="*50)
    
    # Создание датасета для детекции аномалий
    normal_data, anomaly_labels = create_anomaly_dataset(embeddings, anomaly_ratio=0.2)
    
    # Инициализация ансамбля
    ensemble = AnomalyEnsemble(
        input_dim=embeddings.shape[1],
        latent_dim=16,
        hidden_dim=64
    )
    
    print("Начинаем обучение ансамбля детекции аномалий...")
    
    # Обучение (сокращенное для демо)
    import torch
    train_tensor = torch.tensor(normal_data, dtype=torch.float32)
    training_results = ensemble.train_ensemble(train_tensor, epochs=20)
    
    print("Обучение завершено")
    
    # Тестирование
    test_scores = ensemble.predict_combined_anomaly_score(embeddings)
    print(f"Средняя оценка аномальности: {np.mean(test_scores):.3f}")
    
    return test_scores

def demonstrate_results_export(candidates):
    """Демонстрация экспорта результатов"""
    print("\n" + "="*50)
    print("ДЕМОНСТРАЦИЯ ЭКСПОРТА РЕЗУЛЬТАТОВ")
    print("="*50)
    
    # Создание экспортера
    exporter = ResultsExporter(output_dir="demo_results")
    
    if not candidates:
        print("Нет кандидатов для экспорта")
        return
    
    # Экспорт всех кандидатов
    csv_file = exporter.save_candidates_csv(candidates, "demo_candidates.csv")
    json_file = exporter.save_candidates_json(candidates, "demo_candidates.json")
    
    print(f"Кандидаты сохранены в CSV: {csv_file}")
    print(f"Кандидаты сохранены в JSON: {json_file}")
    
    # Создание топ-кандидатов
    top_candidates = exporter.create_top_candidates_list(candidates, top_n=5)
    print(f"Топ-5 кандидатов:")
    for i, candidate in enumerate(top_candidates, 1):
        print(f"  {i}. {candidate.tic_id}: период={candidate.period:.3f}д, "
              f"глубина={candidate.depth:.4f}, уверенность={candidate.confidence:.3f}")
    
    # Создание визуализаций
    plot_files = exporter.create_visualization_plots(candidates, "demo_analysis")
    print(f"Создано графиков: {len(plot_files)}")
    
    # Создание отчета
    report_file = exporter._create_summary_report(candidates, "demo_report.txt")
    print(f"Отчет создан: {report_file}")

def create_demo_visualization(lightcurves, candidates):
    """Создание демонстрационной визуализации"""
    print("\n" + "="*50)
    print("СОЗДАНИЕ ДЕМОНСТРАЦИОННОЙ ВИЗУАЛИЗАЦИИ")
    print("="*50)
    
    # Создание графика с кривыми блеска
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Демонстрация системы поиска экзопланет', fontsize=16)
    
    # График 1: Пример кривой блеска
    times, fluxes = lightcurves[0]
    axes[0, 0].plot(times, fluxes, 'b-', alpha=0.7)
    axes[0, 0].set_title('Пример кривой блеска')
    axes[0, 0].set_xlabel('Время (дни)')
    axes[0, 0].set_ylabel('Поток')
    axes[0, 0].grid(True, alpha=0.3)
    
    # График 2: Распределение периодов кандидатов
    if candidates:
        periods = [c.period for c in candidates]
        axes[0, 1].hist(periods, bins=10, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Распределение периодов')
        axes[0, 1].set_xlabel('Период (дни)')
        axes[0, 1].set_ylabel('Количество')
        axes[0, 1].grid(True, alpha=0.3)
    
    # График 3: Распределение глубин
    if candidates:
        depths = [c.depth for c in candidates]
        axes[1, 0].hist(depths, bins=10, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Распределение глубин транзитов')
        axes[1, 0].set_xlabel('Глубина')
        axes[1, 0].set_ylabel('Количество')
        axes[1, 0].grid(True, alpha=0.3)
    
    # График 4: Уверенность vs качество
    if candidates:
        confidences = [c.confidence for c in candidates]
        quality_scores = [c.quality_score for c in candidates]
        scatter = axes[1, 1].scatter(confidences, quality_scores, 
                                    c=periods, cmap='viridis', alpha=0.7)
        axes[1, 1].set_title('Уверенность vs Качество')
        axes[1, 1].set_xlabel('Уверенность')
        axes[1, 1].set_ylabel('Показатель качества')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label='Период (дни)')
    
    plt.tight_layout()
    plt.savefig('demo_results/demo_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Демонстрационная визуализация сохранена: demo_results/demo_visualization.png")

def main():
    """Основная функция демонстрации"""
    print("🚀 ДЕМОНСТРАЦИЯ СИСТЕМЫ ПОИСКА ЭКЗОПЛАНЕТ")
    print("="*60)
    
    try:
        # 1. Демонстрация загрузки данных
        lightcurves, tic_ids = demonstrate_data_loading()
        
        # 2. Демонстрация гибридного поиска
        candidates = demonstrate_hybrid_search(lightcurves, tic_ids)
        
        # 3. Демонстрация обучения представлений
        embeddings = demonstrate_representation_learning(lightcurves)
        
        # 4. Демонстрация детекции аномалий
        anomaly_scores = demonstrate_anomaly_detection(embeddings)
        
        # 5. Демонстрация экспорта результатов
        demonstrate_results_export(candidates)
        
        # 6. Создание визуализации
        create_demo_visualization(lightcurves, candidates)
        
        print("\n" + "="*60)
        print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        print("="*60)
        print(f"📊 Проанализировано звезд: {len(lightcurves)}")
        print(f"🔍 Найдено кандидатов: {len(candidates)}")
        print(f"📁 Результаты сохранены в: demo_results/")
        print(f"📈 Графики созданы: demo_results/demo_visualization.png")
        
        if candidates:
            best_candidate = max(candidates, key=lambda x: x.quality_score)
            print(f"🏆 Лучший кандидат: {best_candidate.tic_id}")
            print(f"   Период: {best_candidate.period:.3f} дней")
            print(f"   Глубина: {best_candidate.depth:.4f}")
            print(f"   Уверенность: {best_candidate.confidence:.3f}")
        
    except Exception as e:
        print(f"\n❌ Ошибка в демонстрации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
