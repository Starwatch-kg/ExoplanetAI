#!/usr/bin/env python3
"""
Быстрый тест системы поиска экзопланет
Проверяет работоспособность всех компонентов
"""

import sys
import os
from pathlib import Path

# Добавляем src в путь для импортов
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Тест импортов всех модулей"""
    print("Тестирование импортов...")
    
    try:
        from src.tess_data_loader import TESSDataLoader
        print("✅ TESSDataLoader импортирован")
    except Exception as e:
        print(f"❌ Ошибка импорта TESSDataLoader: {e}")
        return False
    
    try:
        from src.hybrid_transit_search import HybridTransitSearch
        print("✅ HybridTransitSearch импортирован")
    except Exception as e:
        print(f"❌ Ошибка импорта HybridTransitSearch: {e}")
        return False
    
    try:
        from src.representation_learning import SelfSupervisedRepresentationLearner
        print("✅ SelfSupervisedRepresentationLearner импортирован")
    except Exception as e:
        print(f"❌ Ошибка импорта SelfSupervisedRepresentationLearner: {e}")
        return False
    
    try:
        from src.anomaly_ensemble import AnomalyEnsemble
        print("✅ AnomalyEnsemble импортирован")
    except Exception as e:
        print(f"❌ Ошибка импорта AnomalyEnsemble: {e}")
        return False
    
    try:
        from src.results_exporter import ResultsExporter, ExoplanetCandidate
        print("✅ ResultsExporter импортирован")
    except Exception as e:
        print(f"❌ Ошибка импорта ResultsExporter: {e}")
        return False
    
    try:
        from src.exoplanet_pipeline import ExoplanetSearchPipeline
        print("✅ ExoplanetSearchPipeline импортирован")
    except Exception as e:
        print(f"❌ Ошибка импорта ExoplanetSearchPipeline: {e}")
        return False
    
    try:
        from src.gpu_optimization import GPUManager
        print("✅ GPUManager импортирован")
    except Exception as e:
        print(f"❌ Ошибка импорта GPUManager: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Тест базовой функциональности"""
    print("\nТестирование базовой функциональности...")
    
    try:
        import numpy as np
        import torch
        
        # Тест создания синтетических данных
        times = np.linspace(0, 30, 1000)
        fluxes = np.ones_like(times) + 0.01 * np.random.randn(len(times))
        
        # Добавление простого транзита
        transit_mask = (times > 10) & (times < 10.5)
        fluxes[transit_mask] -= 0.02
        
        print("✅ Синтетические данные созданы")
        
        # Тест гибридного поиска
        from src.hybrid_transit_search import HybridTransitSearch
        hybrid_search = HybridTransitSearch()
        
        # Упрощенный тест BLS
        from src.hybrid_transit_search import BoxLeastSquares
        bls = BoxLeastSquares(period_range=(1, 20), nperiods=50)
        bls_results = bls.compute_periodogram(times, fluxes)
        
        print("✅ BLS периодограмма вычислена")
        print(f"   Лучший период: {bls_results['best_period']:.3f} дней")
        print(f"   Мощность: {bls_results['best_power']:.3f}")
        
        # Тест создания кандидата
        from src.results_exporter import ExoplanetCandidate
        candidate = ExoplanetCandidate(
            tic_id="TIC_TEST",
            period=10.0,
            depth=0.02,
            duration=0.5,
            start_time=10.0,
            end_time=10.5,
            confidence=0.8,
            combined_score=0.7,
            anomaly_probability=0.3
        )
        
        print("✅ Кандидат в экзопланеты создан")
        print(f"   TIC ID: {candidate.tic_id}")
        print(f"   Период: {candidate.period} дней")
        print(f"   Качество: {candidate.quality_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в базовой функциональности: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_availability():
    """Тест доступности GPU"""
    print("\nТестирование GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print("✅ CUDA доступна")
            print(f"   Количество GPU: {torch.cuda.device_count()}")
            print(f"   Текущее устройство: {torch.cuda.current_device()}")
            print(f"   Имя устройства: {torch.cuda.get_device_name()}")
            
            # Тест памяти GPU
            total_memory = torch.cuda.get_device_properties(0).total_memory
            print(f"   Общая память: {total_memory / 1024**3:.2f} GB")
            
            # Простой тест на GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.mm(x, x.t())
            print("✅ Тест вычислений на GPU прошел")
            
        else:
            print("⚠️  CUDA недоступна, будет использоваться CPU")
            
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования GPU: {e}")
        return False

def test_dependencies():
    """Тест зависимостей"""
    print("\nТестирование зависимостей...")
    
    dependencies = [
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('matplotlib', 'matplotlib'),
        ('pandas', 'pandas'),
        ('sklearn', 'sklearn'),
        ('scipy', 'scipy'),
        ('yaml', 'yaml')
    ]
    
    all_ok = True
    
    for dep_name, import_name in dependencies:
        try:
            exec(f"import {import_name}")
            print(f"✅ {dep_name} доступен")
        except ImportError as e:
            print(f"❌ {dep_name} недоступен: {e}")
            all_ok = False
    
    # Тест астрономических библиотек
    try:
        import lightkurve as lk
        print("✅ lightkurve доступен")
    except ImportError:
        print("⚠️  lightkurve недоступен (необходим для загрузки данных TESS)")
        all_ok = False
    
    try:
        import astropy
        print("✅ astropy доступен")
    except ImportError:
        print("⚠️  astropy недоступен (необходим для астрономических вычислений)")
        all_ok = False
    
    return all_ok

def test_configuration():
    """Тест конфигурации"""
    print("\nТестирование конфигурации...")
    
    try:
        import yaml
        
        config_path = Path("src/config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print("✅ Конфигурация загружена")
            print(f"   Размер входа: {config.get('input_length', 'N/A')}")
            print(f"   Размер батча: {config.get('batch_size', 'N/A')}")
            print(f"   Скорость обучения: {config.get('learning_rate', 'N/A')}")
            
            return True
        else:
            print("❌ Файл конфигурации не найден")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("🧪 БЫСТРЫЙ ТЕСТ СИСТЕМЫ ПОИСКА ЭКЗОПЛАНЕТ")
    print("="*60)
    
    tests = [
        ("Импорты модулей", test_imports),
        ("Базовая функциональность", test_basic_functionality),
        ("Доступность GPU", test_gpu_availability),
        ("Зависимости", test_dependencies),
        ("Конфигурация", test_configuration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Критическая ошибка в тесте '{test_name}': {e}")
            results.append((test_name, False))
    
    # Итоговый отчет
    print("\n" + "="*60)
    print("ИТОГОВЫЙ ОТЧЕТ ТЕСТИРОВАНИЯ")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ ПРОШЕЛ" if result else "❌ НЕ ПРОШЕЛ"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nРезультат: {passed}/{total} тестов прошли успешно")
    
    if passed == total:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print("Система готова к использованию.")
        print("\nДля запуска демонстрации выполните:")
        print("python demo_exoplanet_search.py")
        print("\nДля поиска экзопланет выполните:")
        print("python exoplanet_search.py --tic-ids 261136679")
    else:
        print(f"\n⚠️  {total - passed} тестов не прошли.")
        print("Проверьте установку зависимостей и конфигурацию.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
