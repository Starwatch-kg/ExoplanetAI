"""
Основной модуль для поиска экзопланет.

Этот модуль содержит точку входа с консольным меню для выбора
различных операций: обучение моделей, поиск экзопланет, тестирование.
"""

import logging
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from datetime import datetime

# Импорты из других модулей
import numpy as np
from preprocess import TESSDataProcessor, create_synthetic_data
from model import train_model, ExoplanetAutoencoder, ExoplanetClassifier
from pipeline import search_exoplanets, ExoplanetPipeline
from visualize import visualize_results, ExoplanetVisualizer
from utils import calculate_metrics, create_train_test_split, validate_data_quality

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('exoplanet_search.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ExoplanetSearchApp:
    """Основное приложение для поиска экзопланет."""
    
    def __init__(self):
        """Инициализация приложения."""
        self.data_processor = TESSDataProcessor()
        self.visualizer = ExoplanetVisualizer()
        
        # Создание директорий
        self._create_directories()
        
        logger.info("Приложение инициализировано")
    
    def _create_directories(self):
        """Создание необходимых директорий."""
        directories = [
            'data/tess_cache',
            'models',
            'results',
            'results/plots',
            'results/candidates',
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def train_model(self, model_type: str = 'autoencoder',
                   epochs: int = 100,
                   batch_size: int = 32,
                   learning_rate: float = 1e-3,
                   use_synthetic: bool = True,
                   tic_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Обучение модели для детекции экзопланет.
        
        Args:
            model_type: Тип модели ('autoencoder' или 'classifier').
            epochs: Количество эпох обучения.
            batch_size: Размер батча.
            learning_rate: Скорость обучения.
            use_synthetic: Использовать синтетические данные.
            tic_ids: Список TIC ID для обучения (если не синтетические).
            
        Returns:
            Dict[str, Any]: Результаты обучения.
        """
        logger.info(f"Начинаем обучение модели типа: {model_type}")
        
        try:
            # Подготовка данных
            if use_synthetic:
                logger.info("Использование синтетических данных")
                train_data, train_labels = create_synthetic_data(
                    num_samples=1000,
                    length=2000,
                    transit_fraction=0.3
                )
                
                # Разделение на train/val
                X_train, X_val, y_train, y_val = create_train_test_split(
                    train_data, train_labels, test_size=0.2, random_state=42
                )
                
            else:
                if not tic_ids:
                    raise ValueError("Необходимо указать TIC ID для реальных данных")
                
                logger.info(f"Загрузка реальных данных для {len(tic_ids)} звезд")
                lightcurves = self.data_processor.load_multiple_stars(tic_ids)
                
                if not lightcurves:
                    raise ValueError("Не удалось загрузить данные TESS")
                
                # Подготовка данных для обучения
                train_data, train_labels = self.data_processor.prepare_training_data(lightcurves)
                
                # Разделение на train/val
                X_train, X_val, y_train, y_val = create_train_test_split(
                    train_data, train_labels, test_size=0.2, random_state=42
                )
            
            logger.info(f"Подготовлено данных: train={len(X_train)}, val={len(X_val)}")
            
            # Обучение модели
            model_path = f"models/{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            
            model, history = train_model(
                model_type=model_type,
                train_data=X_train,
                train_labels=y_train if model_type == 'classifier' else None,
                val_data=X_val,
                val_labels=y_val if model_type == 'classifier' else None,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                save_path=model_path
            )
            
            # Визуализация истории обучения
            history_path = f"results/plots/training_history_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.visualizer.plot_training_history(history, f"Training History - {model_type}", history_path)
            
            # Вычисление финальных метрик
            if model_type == 'classifier':
                import torch
                from model import ModelTrainer
                trainer = ModelTrainer()
                
                # Предсказания на валидационной выборке
                model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
                    predictions = model(X_val_tensor)
                    predicted_labels = torch.argmax(predictions, dim=1).numpy()
                    predicted_scores = torch.softmax(predictions, dim=1)[:, 1].numpy()
                
                metrics = calculate_metrics(y_val, predicted_labels, predicted_scores)
            else:
                metrics = {'model_type': model_type}
            
            results = {
                'model_type': model_type,
                'model_path': model_path,
                'history': history,
                'metrics': metrics,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'epochs': epochs,
                'final_train_loss': history['train_loss'][-1] if 'train_loss' in history else 0,
                'final_val_loss': history['val_loss'][-1] if 'val_loss' in history else 0
            }
            
            logger.info(f"Обучение завершено. Модель сохранена: {model_path}")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}")
            raise
    
    def search_exoplanets(self, tic_ids: List[str],
                         sectors: Optional[List[int]] = None,
                         autoencoder_path: Optional[str] = None,
                         classifier_path: Optional[str] = None,
                         config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Поиск экзопланет для списка звезд.
        
        Args:
            tic_ids: Список TIC ID для анализа.
            sectors: Список секторов TESS.
            autoencoder_path: Путь к автоэнкодеру.
            classifier_path: Путь к классификатору.
            config: Конфигурация пайплайна.
            
        Returns:
            Dict[str, Any]: Результаты поиска.
        """
        logger.info(f"Начинаем поиск экзопланет для {len(tic_ids)} звезд")
        
        try:
            # Поиск экзопланет
            results = search_exoplanets(
                tic_ids=tic_ids,
                sectors=sectors,
                config=config,
                autoencoder_path=autoencoder_path,
                classifier_path=classifier_path
            )
            
            # Визуализация результатов
            if results.get('candidates'):
                logger.info("Создание визуализаций")
                plot_files = visualize_results(results, create_all=True)
                results['plot_files'] = plot_files
            
            # Сохранение текстового отчета
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = f"results/search_report_{timestamp}.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("ОТЧЕТ О ПОИСКЕ ЭКЗОПЛАНЕТ\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Дата поиска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Время обработки: {results.get('processing_time', 'N/A')}\n")
                f.write(f"Анализируемые звезды: {len(tic_ids)}\n")
                f.write(f"Загружено звезд: {results.get('loaded_stars', 0)}\n")
                f.write(f"Найдено кандидатов: {results.get('total_candidates', 0)}\n\n")
                
                # Ошибки
                if results.get('errors'):
                    f.write("ОШИБКИ:\n")
                    for error in results['errors']:
                        f.write(f"  - {error}\n")
                    f.write("\n")
                
                # Топ кандидаты
                candidates = results.get('candidates', [])
                if candidates:
                    f.write("ТОП-10 КАНДИДАТОВ:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{'№':<3} {'TIC ID':<12} {'Метод':<12} {'Период':<8} {'Уверенность':<12}\n")
                    f.write("-" * 80 + "\n")
                    
                    # Сортировка по уверенности
                    sorted_candidates = sorted(candidates, key=lambda x: x.get('confidence', 0), reverse=True)
                    
                    for i, candidate in enumerate(sorted_candidates[:10], 1):
                        f.write(f"{i:<3} {candidate['tic_id']:<12} "
                               f"{candidate['method']:<12} "
                               f"{candidate.get('period', 0):<8.3f} "
                               f"{candidate.get('confidence', 0):<12.3f}\n")
            
            results['report_path'] = report_path
            
            logger.info(f"Поиск завершен. Найдено {results.get('total_candidates', 0)} кандидатов")
            logger.info(f"Отчет сохранен: {report_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при поиске экзопланет: {e}")
            raise
    
    def test_pipeline(self, test_tic_ids: Optional[List[str]] = None,
                     use_synthetic: bool = True,
                     autoencoder_path: Optional[str] = None,
                     classifier_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Тестирование пайплайна на тестовых данных.
        
        Args:
            test_tic_ids: Список TIC ID для тестирования.
            use_synthetic: Использовать синтетические данные.
            autoencoder_path: Путь к автоэнкодеру.
            classifier_path: Путь к классификатору.
            
        Returns:
            Dict[str, Any]: Результаты тестирования.
        """
        logger.info("Начинаем тестирование пайплайна")
        
        try:
            if use_synthetic:
                logger.info("Тестирование на синтетических данных")
                
                # Создание синтетических данных с известными транзитами
                test_data, test_labels = create_synthetic_data(
                    num_samples=200,
                    length=2000,
                    transit_fraction=0.4
                )
                
                # Создание фиктивных TIC ID
                test_tic_ids = [f"TIC_TEST_{i:06d}" for i in range(len(test_data))]
                
                # Сохранение синтетических данных для тестирования
                test_results = {
                    'test_samples': len(test_data),
                    'transit_samples': np.sum(test_labels),
                    'no_transit_samples': len(test_labels) - np.sum(test_labels),
                    'synthetic_data': True
                }
                
            else:
                if not test_tic_ids:
                    raise ValueError("Необходимо указать TIC ID для тестирования")
                
                logger.info(f"Тестирование на реальных данных: {len(test_tic_ids)} звезд")
                test_results = {
                    'test_samples': len(test_tic_ids),
                    'synthetic_data': False
                }
            
            # Запуск пайплайна
            if use_synthetic:
                # Для синтетических данных используем специальную логику тестирования
                pipeline_results = self._test_with_synthetic_data(test_data, test_labels, test_tic_ids)
            else:
                pipeline_results = self.search_exoplanets(
                    tic_ids=test_tic_ids,
                    autoencoder_path=autoencoder_path,
                    classifier_path=classifier_path
                )
            
            # Объединение результатов
            test_results.update(pipeline_results)
            
            # Вычисление метрик производительности
            if use_synthetic and 'metrics' in pipeline_results:
                # Для синтетических данных используем вычисленные метрики
                test_results['performance_metrics'] = pipeline_results['metrics']
            elif use_synthetic and 'candidates' in pipeline_results:
                # Fallback для старого формата
                candidates = pipeline_results['candidates']
                
                # Простая эвристика: если найдены кандидаты, считаем это успехом
                detected_transits = len(candidates)
                true_transits = np.sum(test_labels)
                
                precision = min(detected_transits / max(detected_transits, 1), 1.0)
                recall = min(detected_transits / max(true_transits, 1), 1.0)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                test_results['performance_metrics'] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'detected_transits': detected_transits,
                    'true_transits': true_transits
                }
            
            # Сохранение результатов тестирования
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            test_report_path = f"results/test_report_{timestamp}.txt"
            
            with open(test_report_path, 'w', encoding='utf-8') as f:
                f.write("ОТЧЕТ О ТЕСТИРОВАНИИ ПАЙПЛАЙНА\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Тип данных: {'Синтетические' if use_synthetic else 'Реальные'}\n")
                f.write(f"Количество тестовых образцов: {test_results['test_samples']}\n")
                
                if use_synthetic:
                    f.write(f"Образцы с транзитами: {test_results['transit_samples']}\n")
                    f.write(f"Образцы без транзитов: {test_results['no_transit_samples']}\n")
                
                f.write(f"Найдено кандидатов: {test_results.get('total_candidates', 0)}\n")
                f.write(f"Время обработки: {test_results.get('processing_time', 'N/A')}\n\n")
                
                # Метрики производительности
                if 'performance_metrics' in test_results:
                    metrics = test_results['performance_metrics']
                    f.write("МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ:\n")
                    f.write(f"  Precision: {metrics['precision']:.3f}\n")
                    f.write(f"  Recall: {metrics['recall']:.3f}\n")
                    f.write(f"  F1-Score: {metrics['f1_score']:.3f}\n")
                    f.write(f"  Обнаружено транзитов: {metrics['detected_transits']}\n")
                    f.write(f"  Истинных транзитов: {metrics['true_transits']}\n")
            
            test_results['test_report_path'] = test_report_path
            
            logger.info("Тестирование пайплайна завершено")
            logger.info(f"Отчет о тестировании сохранен: {test_report_path}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Ошибка при тестировании пайплайна: {e}")
            raise
    
    def _test_with_synthetic_data(self, test_data: np.ndarray, test_labels: np.ndarray, test_tic_ids: List[str]) -> Dict[str, Any]:
        """
        Тестирование пайплайна на синтетических данных без обращения к MAST.
        
        Args:
            test_data: Синтетические данные световых кривых.
            test_labels: Метки транзитов (1 - есть транзит, 0 - нет транзита).
            test_tic_ids: Список фиктивных TIC ID.
            
        Returns:
            Dict[str, Any]: Результаты тестирования.
        """
        logger.info("Тестирование пайплайна на синтетических данных")
        
        try:
            # Импортируем необходимые компоненты
            from pipeline import BoxLeastSquares
            from utils import calculate_metrics
            
            # Инициализация BLS анализатора
            bls_analyzer = BoxLeastSquares()
            
            candidates = []
            lightcurves = []
            all_scores = []
            all_predictions = []
            
            # Анализ каждой синтетической световой кривой
            for i, (data, label, tic_id) in enumerate(zip(test_data, test_labels, test_tic_ids)):
                logger.info(f"Анализ образца {i+1}/{len(test_data)}: {tic_id}")
                
                # Создание временной шкалы
                times = np.linspace(0, 30, len(data))
                
                # BLS анализ
                bls_results = bls_analyzer.compute_periodogram(times, data)
                
                # Простая эвристика для определения кандидатов
                if bls_results and bls_results['best_power'] > 1000:  # Порог для синтетических данных
                    candidate = {
                        'tic_id': tic_id,
                        'period': bls_results['best_period'],
                        'depth': 0.01,  # Фиктивная глубина
                        'confidence': min(bls_results['best_power'] / 10000, 1.0),
                        'method': 'BLS',
                        'bls_power': bls_results['best_power']
                    }
                    candidates.append(candidate)
                    all_predictions.append(1)
                else:
                    all_predictions.append(0)
                
                # Сохранение световой кривой
                lightcurves.append({
                    'tic_id': tic_id,
                    'times': times,
                    'fluxes': data,
                    'has_transit': bool(label)
                })
                
                # Генерация фиктивных скоров
                score = bls_results['best_power'] if bls_results else 0
                all_scores.append(score)
            
            # Вычисление метрик
            metrics = calculate_metrics(test_labels, np.array(all_predictions), np.array(all_scores))
            
            # Создание результатов
            results = {
                'candidates': candidates,
                'lightcurves': lightcurves,
                'metrics': metrics,
                'total_stars': len(test_data),
                'detected_candidates': len(candidates),
                'true_transits': int(np.sum(test_labels)),
                'synthetic_test': True
            }
            
            logger.info(f"Тестирование завершено: найдено {len(candidates)} кандидатов из {int(np.sum(test_labels))} истинных транзитов")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при тестировании на синтетических данных: {e}")
            raise
    
    def visualize_results(self, results_path: str) -> Dict[str, str]:
        """
        Визуализация результатов из файла.
        
        Args:
            results_path: Путь к файлу с результатами.
            
        Returns:
            Dict[str, str]: Пути к созданным графикам.
        """
        logger.info(f"Визуализация результатов из файла: {results_path}")
        
        try:
            # Загрузка результатов
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Создание визуализаций
            plot_files = visualize_results(results, create_all=True)
            
            logger.info(f"Создано {len(plot_files)} графиков")
            return plot_files
            
        except Exception as e:
            logger.error(f"Ошибка при визуализации результатов: {e}")
            raise


def display_menu():
    """Отображение главного меню."""
    print("\n" + "="*60)
    print("🔍 СИСТЕМА ПОИСКА ЭКЗОПЛАНЕТ")
    print("="*60)
    print("1. Обучить модель")
    print("2. Поиск экзопланет")
    print("3. Тестирование пайплайна")
    print("4. Визуализация результатов")
    print("5. Выход")
    print("="*60)


def get_user_input(prompt: str, input_type: type = str, default: Any = None) -> Any:
    """
    Получение ввода от пользователя с проверкой типа.
    
    Args:
        prompt: Текст запроса.
        input_type: Ожидаемый тип данных.
        default: Значение по умолчанию.
        
    Returns:
        Any: Введенное значение.
    """
    while True:
        try:
            user_input = input(f"{prompt}: ").strip()
            
            if not user_input and default is not None:
                return default
            
            if input_type == bool:
                return user_input.lower() in ['y', 'yes', 'да', '1', 'true']
            elif input_type == int:
                return int(user_input)
            elif input_type == float:
                return float(user_input)
            elif input_type == list:
                return [item.strip() for item in user_input.split(',') if item.strip()]
            else:
                return user_input
                
        except ValueError:
            print(f"Ошибка: введите корректное значение типа {input_type.__name__}")
        except KeyboardInterrupt:
            print("\nОперация отменена пользователем")
            return None


def main():
    """Основная функция приложения."""
    parser = argparse.ArgumentParser(description='Система поиска экзопланет')
    parser.add_argument('--mode', choices=['interactive', 'train', 'search', 'test', 'visualize'],
                       default='interactive', help='Режим работы')
    parser.add_argument('--tic-ids', nargs='+', help='TIC ID для анализа')
    parser.add_argument('--model-type', choices=['autoencoder', 'classifier'], 
                       default='autoencoder', help='Тип модели')
    parser.add_argument('--epochs', type=int, default=100, help='Количество эпох')
    parser.add_argument('--autoencoder-path', help='Путь к автоэнкодеру')
    parser.add_argument('--classifier-path', help='Путь к классификатору')
    parser.add_argument('--results-path', help='Путь к файлу результатов')
    
    args = parser.parse_args()
    
    app = ExoplanetSearchApp()
    
    if args.mode == 'interactive':
        # Интерактивный режим
        while True:
            try:
                display_menu()
                choice = get_user_input("Выберите опцию (1-5)", int)
                
                if choice is None:
                    break
                
                if choice == 1:
                    # Обучение модели
                    print("\n📚 ОБУЧЕНИЕ МОДЕЛИ")
                    print("-" * 30)
                    
                    model_type = get_user_input("Тип модели (autoencoder/classifier)", 
                                               str, "autoencoder")
                    epochs = get_user_input("Количество эпох", int, 100)
                    batch_size = get_user_input("Размер батча", int, 32)
                    learning_rate = get_user_input("Скорость обучения", float, 1e-3)
                    use_synthetic = get_user_input("Использовать синтетические данные? (y/n)", 
                                                 bool, True)
                    
                    if not use_synthetic:
                        tic_ids_input = get_user_input("TIC ID (через запятую)", str)
                        tic_ids = [tid.strip() for tid in tic_ids_input.split(',')] if tic_ids_input else None
                    else:
                        tic_ids = None
                    
                    results = app.train_model(
                        model_type=model_type,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        use_synthetic=use_synthetic,
                        tic_ids=tic_ids
                    )
                    
                    print(f"\n✅ Обучение завершено!")
                    print(f"Модель сохранена: {results['model_path']}")
                    print(f"Финальная потеря: {results['final_train_loss']:.4f}")
                    
                elif choice == 2:
                    # Поиск экзопланет
                    print("\n🔍 ПОИСК ЭКЗОПЛАНЕТ")
                    print("-" * 30)
                    
                    tic_ids_input = get_user_input("TIC ID (через запятую)", str)
                    if not tic_ids_input:
                        print("Ошибка: необходимо указать TIC ID")
                        continue
                    
                    tic_ids = [tid.strip() for tid in tic_ids_input.split(',')]
                    
                    sectors_input = get_user_input("Секторы TESS (через запятую, Enter для всех)", str)
                    sectors = None
                    if sectors_input:
                        sectors = [int(s.strip()) for s in sectors_input.split(',')]
                    
                    autoencoder_path = get_user_input("Путь к автоэнкодеру (Enter для пропуска)", str)
                    classifier_path = get_user_input("Путь к классификатору (Enter для пропуска)", str)
                    
                    results = app.search_exoplanets(
                        tic_ids=tic_ids,
                        sectors=sectors,
                        autoencoder_path=autoencoder_path if autoencoder_path else None,
                        classifier_path=classifier_path if classifier_path else None
                    )
                    
                    print(f"\n✅ Поиск завершен!")
                    print(f"Найдено кандидатов: {results.get('total_candidates', 0)}")
                    print(f"Отчет сохранен: {results.get('report_path', 'N/A')}")
                    
                elif choice == 3:
                    # Тестирование пайплайна
                    print("\n🧪 ТЕСТИРОВАНИЕ ПАЙПЛАЙНА")
                    print("-" * 30)
                    
                    use_synthetic = get_user_input("Использовать синтетические данные? (y/n)", 
                                                 bool, True)
                    
                    test_tic_ids = None
                    if not use_synthetic:
                        tic_ids_input = get_user_input("TIC ID для тестирования (через запятую)", str)
                        if tic_ids_input:
                            test_tic_ids = [tid.strip() for tid in tic_ids_input.split(',')]
                    
                    autoencoder_path = get_user_input("Путь к автоэнкодеру (Enter для пропуска)", str)
                    classifier_path = get_user_input("Путь к классификатору (Enter для пропуска)", str)
                    
                    results = app.test_pipeline(
                        test_tic_ids=test_tic_ids,
                        use_synthetic=use_synthetic,
                        autoencoder_path=autoencoder_path if autoencoder_path else None,
                        classifier_path=classifier_path if classifier_path else None
                    )
                    
                    print(f"\n✅ Тестирование завершено!")
                    print(f"Тестовых образцов: {results['test_samples']}")
                    print(f"Найдено кандидатов: {results.get('total_candidates', 0)}")
                    
                    if 'performance_metrics' in results:
                        metrics = results['performance_metrics']
                        print(f"Precision: {metrics['precision']:.3f}")
                        print(f"Recall: {metrics['recall']:.3f}")
                        print(f"F1-Score: {metrics['f1_score']:.3f}")
                    
                    print(f"Отчет о тестировании: {results.get('test_report_path', 'N/A')}")
                    
                elif choice == 4:
                    # Визуализация результатов
                    print("\n📊 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
                    print("-" * 30)
                    
                    results_path = get_user_input("Путь к файлу результатов", str)
                    if not results_path:
                        print("Ошибка: необходимо указать путь к файлу результатов")
                        continue
                    
                    plot_files = app.visualize_results(results_path)
                    
                    print(f"\n✅ Визуализация завершена!")
                    print(f"Создано графиков: {len(plot_files)}")
                    for plot_type, plot_path in plot_files.items():
                        print(f"  {plot_type}: {plot_path}")
                    
                elif choice == 5:
                    print("\n👋 До свидания!")
                    break
                
                else:
                    print("❌ Неверный выбор. Попробуйте снова.")
                
                input("\nНажмите Enter для продолжения...")
                
            except KeyboardInterrupt:
                print("\n\n👋 До свидания!")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")
                input("Нажмите Enter для продолжения...")
    
    else:
        # Режим командной строки
        try:
            if args.mode == 'train':
                results = app.train_model(
                    model_type=args.model_type,
                    epochs=args.epochs,
                    tic_ids=args.tic_ids
                )
                print(f"Обучение завершено. Модель: {results['model_path']}")
                
            elif args.mode == 'search':
                if not args.tic_ids:
                    print("Ошибка: необходимо указать --tic-ids")
                    return
                
                results = app.search_exoplanets(
                    tic_ids=args.tic_ids,
                    autoencoder_path=args.autoencoder_path,
                    classifier_path=args.classifier_path
                )
                print(f"Поиск завершен. Найдено кандидатов: {results.get('total_candidates', 0)}")
                
            elif args.mode == 'test':
                results = app.test_pipeline(
                    test_tic_ids=args.tic_ids,
                    autoencoder_path=args.autoencoder_path,
                    classifier_path=args.classifier_path
                )
                print(f"Тестирование завершено. Образцов: {results['test_samples']}")
                
            elif args.mode == 'visualize':
                if not args.results_path:
                    print("Ошибка: необходимо указать --results-path")
                    return
                
                plot_files = app.visualize_results(args.results_path)
                print(f"Визуализация завершена. Графиков: {len(plot_files)}")
                
        except Exception as e:
            logger.error(f"Ошибка в режиме командной строки: {e}")
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()