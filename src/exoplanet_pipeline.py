"""
Основной pipeline для поиска экзопланет по данным TESS/MAST
Объединяет все компоненты системы в единый рабочий процесс
"""

import os
import sys
import logging
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import warnings
from datetime import datetime
import argparse

# Локальные импорты
from src.tess_data_loader import TESSDataLoader, create_tess_dataset
from src.hybrid_transit_search import HybridTransitSearch
from src.representation_learning import SelfSupervisedRepresentationLearner, create_representation_dataset
from src.anomaly_ensemble import AnomalyEnsemble, create_anomaly_dataset
from src.results_exporter import ResultsExporter, ExoplanetCandidate

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('exoplanet_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Подавление предупреждений
warnings.filterwarnings('ignore')


class ExoplanetSearchPipeline:
    """
    Основной класс для поиска экзопланет
    Объединяет все компоненты системы
    """
    
    def __init__(self, config_path: str = "src/config.yaml"):
        """
        Инициализация pipeline
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Инициализация ExoplanetSearchPipeline на устройстве: {self.device}")
        
        # Инициализация компонентов
        self.data_loader = TESSDataLoader(cache_dir=self.config.get('cache_dir', 'data/tess_cache'))
        self.hybrid_search = HybridTransitSearch(
            bls_config=self.config.get('bls_config', {}),
            neural_config=self.config.get('neural_config', {})
        )
        self.representation_learner = SelfSupervisedRepresentationLearner(
            input_length=self.config.get('input_length', 2000),
            embedding_dim=self.config.get('embedding_dim', 128),
            hidden_dim=self.config.get('hidden_dim', 256),
            num_layers=self.config.get('num_layers', 4)
        )
        self.anomaly_ensemble = AnomalyEnsemble(
            input_dim=self.config.get('embedding_dim', 128),
            latent_dim=self.config.get('latent_dim', 32),
            hidden_dim=self.config.get('hidden_dim', 256),
            device=str(self.device)
        )
        self.results_exporter = ResultsExporter(
            output_dir=self.config.get('output_dir', 'results')
        )
        
        # Создание директорий
        self._create_directories()
        
        logger.info("Pipeline инициализирован успешно")
    
    def _load_config(self, config_path: str) -> Dict:
        """Загружает конфигурацию из файла"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Конфигурация загружена из {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Файл конфигурации {config_path} не найден, используем значения по умолчанию")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Возвращает конфигурацию по умолчанию"""
        return {
            'input_length': 2000,
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 4,
            'latent_dim': 32,
            'cache_dir': 'data/tess_cache',
            'output_dir': 'results',
            'bls_config': {
                'period_range': [0.5, 50.0],
                'nperiods': 1000,
                'oversample_factor': 5
            },
            'neural_config': {
                'input_length': 2000,
                'hidden_dim': 256,
                'num_layers': 3,
                'dropout': 0.3
            },
            'training': {
                'representation_epochs': 100,
                'anomaly_epochs': 100,
                'learning_rate': 1e-3,
                'batch_size': 32
            },
            'search': {
                'confidence_threshold': 0.3,
                'top_n_candidates': 50
            }
        }
    
    def _create_directories(self):
        """Создает необходимые директории"""
        directories = [
            self.config.get('cache_dir', 'data/tess_cache'),
            self.config.get('output_dir', 'results'),
            'logs',
            'models'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_tess_data(self, tic_ids: List[Union[int, str]], 
                       sectors: Optional[List[int]] = None,
                       use_cache: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Загружает данные TESS для списка звезд
        
        Args:
            tic_ids: Список TIC ID
            sectors: Список секторов TESS
            use_cache: Использовать ли кэшированные данные
            
        Returns:
            List[Tuple[times, fluxes]]: Список кривых блеска
        """
        logger.info(f"Загружаем данные TESS для {len(tic_ids)} звезд")
        
        lightcurves = []
        successful_loads = 0
        
        for i, tic_id in enumerate(tic_ids):
            try:
                logger.info(f"Загрузка {i+1}/{len(tic_ids)}: TIC {tic_id}")
                
                # Проверка кэша
                cache_file = f"tic_{tic_id}_lightcurve.csv"
                if use_cache and (self.data_loader.cache_dir / cache_file).exists():
                    times, fluxes, metadata = self.data_loader.load_lightcurve(cache_file)
                    logger.info(f"Загружено из кэша: TIC {tic_id}")
                else:
                    # Загрузка из MAST
                    times, fluxes = self.data_loader.load_by_tic_id(tic_id, sectors)
                    
                    # Получение информации о звезде
                    star_info = self.data_loader.get_star_info(tic_id)
                    
                    # Сохранение в кэш
                    metadata = {'tic_id': tic_id, **star_info}
                    self.data_loader.save_lightcurve(times, fluxes, cache_file, metadata)
                
                lightcurves.append((times, fluxes))
                successful_loads += 1
                
            except Exception as e:
                logger.error(f"Ошибка загрузки TIC {tic_id}: {e}")
                continue
        
        logger.info(f"Успешно загружено {successful_loads}/{len(tic_ids)} кривых блеска")
        return lightcurves
    
    def train_representation_model(self, lightcurves: List[Tuple[np.ndarray, np.ndarray]],
                                 epochs: Optional[int] = None) -> Dict:
        """
        Обучает модель представлений
        
        Args:
            lightcurves: Список кривых блеска
            epochs: Количество эпох (если None, берется из конфигурации)
            
        Returns:
            Dict: Результаты обучения
        """
        logger.info("Начинаем обучение модели представлений")
        
        # Извлечение только потоков
        fluxes_list = [fluxes for _, fluxes in lightcurves]
        
        # Создание DataLoader
        dataloader = create_representation_dataset(
            fluxes_list, 
            batch_size=self.config['training']['batch_size']
        )
        
        # Обучение
        epochs = epochs or self.config['training']['representation_epochs']
        loss_history = self.representation_learner.train(
            dataloader, 
            epochs=epochs,
            learning_rate=self.config['training']['learning_rate']
        )
        
        # Сохранение модели
        model_path = f"models/contrastive_encoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        self.representation_learner.save_model(model_path)
        
        logger.info("Обучение модели представлений завершено")
        
        return {
            'loss_history': loss_history,
            'model_path': model_path,
            'epochs': epochs
        }
    
    def train_anomaly_detection(self, lightcurves: List[Tuple[np.ndarray, np.ndarray]],
                               epochs: Optional[int] = None) -> Dict:
        """
        Обучает ансамбль детекции аномалий
        
        Args:
            lightcurves: Список кривых блеска
            epochs: Количество эпох
            
        Returns:
            Dict: Результаты обучения
        """
        logger.info("Начинаем обучение ансамбля детекции аномалий")
        
        # Кодирование кривых блеска в представления
        fluxes_list = [fluxes for _, fluxes in lightcurves]
        dataloader = create_representation_dataset(
            fluxes_list,
            batch_size=self.config['training']['batch_size']
        )
        
        embeddings, metadata = self.representation_learner.encode_dataset(dataloader)
        
        # Создание датасета для детекции аномалий
        normal_data, anomaly_labels = create_anomaly_dataset(
            embeddings, 
            anomaly_ratio=0.1
        )
        
        # Обучение ансамбля
        epochs = epochs or self.config['training']['anomaly_epochs']
        train_tensor = torch.tensor(normal_data, dtype=torch.float32)
        
        training_results = self.anomaly_ensemble.train_ensemble(
            train_tensor,
            epochs=epochs,
            learning_rate=self.config['training']['learning_rate']
        )
        
        # Сохранение модели
        ensemble_path = f"models/anomaly_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        self.anomaly_ensemble.save_ensemble(ensemble_path)
        
        logger.info("Обучение ансамбля детекции аномалий завершено")
        
        return {
            'training_results': training_results,
            'ensemble_path': ensemble_path,
            'embeddings_shape': embeddings.shape,
            'epochs': epochs
        }
    
    def search_exoplanets(self, lightcurves: List[Tuple[np.ndarray, np.ndarray]],
                         tic_ids: List[Union[int, str]]) -> List[ExoplanetCandidate]:
        """
        Выполняет поиск экзопланет для списка кривых блеска
        
        Args:
            lightcurves: Список кривых блеска
            tic_ids: Список TIC ID
            
        Returns:
            List[ExoplanetCandidate]: Список найденных кандидатов
        """
        logger.info(f"Начинаем поиск экзопланет для {len(lightcurves)} кривых блеска")
        
        all_candidates = []
        
        for i, ((times, fluxes), tic_id) in enumerate(zip(lightcurves, tic_ids)):
            try:
                logger.info(f"Обработка {i+1}/{len(lightcurves)}: TIC {tic_id}")
                
                # Гибридный поиск транзитов
                search_results = self.hybrid_search.search_transits(
                    times, fluxes, device=str(self.device)
                )
                
                # Получение представлений для детекции аномалий
                fluxes_tensor = torch.tensor(fluxes, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                embeddings = self.representation_learner.encode(fluxes_tensor.to(self.device))
                
                # Оценка аномальности
                anomaly_probability = self.anomaly_ensemble.predict_anomaly_probability(
                    embeddings.cpu().numpy()
                )[0]  # Берем только вероятность
                
                # Создание кандидатов
                for candidate_data in search_results['candidates']:
                    # Получение информации о звезде
                    star_info = self.data_loader.get_star_info(tic_id)
                    
                    candidate = ExoplanetCandidate(
                        tic_id=tic_id,
                        period=candidate_data['period'],
                        depth=candidate_data['depth'],
                        duration=candidate_data['duration'],
                        start_time=candidate_data['start_time'],
                        end_time=candidate_data['end_time'],
                        confidence=candidate_data['confidence'],
                        combined_score=candidate_data['combined_score'],
                        anomaly_probability=anomaly_probability,
                        star_info=star_info,
                        metadata={
                            'search_method': 'hybrid',
                            'processing_time': datetime.now().isoformat(),
                            'sectors': search_results.get('sectors', [])
                        }
                    )
                    
                    all_candidates.append(candidate)
                
                logger.info(f"Найдено {len(search_results['candidates'])} кандидатов для TIC {tic_id}")
                
            except Exception as e:
                logger.error(f"Ошибка обработки TIC {tic_id}: {e}")
                continue
        
        logger.info(f"Всего найдено {len(all_candidates)} кандидатов")
        return all_candidates
    
    def run_full_pipeline(self, tic_ids: List[Union[int, str]],
                         sectors: Optional[List[int]] = None,
                         train_models: bool = True,
                         top_n: Optional[int] = None) -> Dict:
        """
        Запускает полный pipeline поиска экзопланет
        
        Args:
            tic_ids: Список TIC ID для анализа
            sectors: Список секторов TESS
            train_models: Обучать ли модели с нуля
            top_n: Количество топ кандидатов для экспорта
            
        Returns:
            Dict: Результаты выполнения pipeline
        """
        logger.info("Запуск полного pipeline поиска экзопланет")
        start_time = datetime.now()
        
        results = {
            'start_time': start_time.isoformat(),
            'tic_ids': tic_ids,
            'sectors': sectors,
            'total_candidates': 0,
            'top_candidates': 0,
            'exported_files': {}
        }
        
        try:
            # 1. Загрузка данных
            logger.info("Этап 1: Загрузка данных TESS")
            lightcurves = self.load_tess_data(tic_ids, sectors)
            
            if not lightcurves:
                raise ValueError("Не удалось загрузить данные TESS")
            
            results['loaded_lightcurves'] = len(lightcurves)
            
            # 2. Обучение моделей (если требуется)
            if train_models:
                logger.info("Этап 2: Обучение моделей")
                
                # Обучение модели представлений
                representation_results = self.train_representation_model(lightcurves)
                results['representation_training'] = representation_results
                
                # Обучение ансамбля детекции аномалий
                anomaly_results = self.train_anomaly_detection(lightcurves)
                results['anomaly_training'] = anomaly_results
            
            # 3. Поиск экзопланет
            logger.info("Этап 3: Поиск экзопланет")
            candidates = self.search_exoplanets(lightcurves, tic_ids)
            results['total_candidates'] = len(candidates)
            
            # 4. Экспорт результатов
            logger.info("Этап 4: Экспорт результатов")
            top_n = top_n or self.config['search']['top_n_candidates']
            
            exported_files = self.results_exporter.export_complete_results(
                candidates, 
                top_n=top_n,
                create_plots=True
            )
            results['exported_files'] = exported_files
            results['top_candidates'] = min(top_n, len(candidates))
            
            # 5. Создание итогового отчета
            logger.info("Этап 5: Создание итогового отчета")
            final_report = self._create_final_report(results, candidates)
            results['final_report'] = final_report
            
            end_time = datetime.now()
            results['end_time'] = end_time.isoformat()
            results['duration'] = str(end_time - start_time)
            
            logger.info(f"Pipeline завершен успешно за {results['duration']}")
            logger.info(f"Найдено {results['total_candidates']} кандидатов")
            logger.info(f"Экспортировано {len(exported_files)} файлов")
            
        except Exception as e:
            logger.error(f"Ошибка в pipeline: {e}")
            results['error'] = str(e)
            results['end_time'] = datetime.now().isoformat()
        
        return results
    
    def _create_final_report(self, results: Dict, candidates: List[ExoplanetCandidate]) -> str:
        """
        Создает итоговый отчет о выполнении pipeline
        
        Args:
            results: Результаты выполнения
            candidates: Список кандидатов
            
        Returns:
            str: Путь к файлу отчета
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"pipeline_report_{timestamp}.txt"
        filepath = Path(self.config['output_dir']) / "summaries" / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("ИТОГОВЫЙ ОТЧЕТ О ПОИСКЕ ЭКЗОПЛАНЕТ\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Дата выполнения: {results['start_time']}\n")
            f.write(f"Время выполнения: {results.get('duration', 'N/A')}\n")
            f.write(f"Анализируемые звезды: {len(results['tic_ids'])}\n")
            f.write(f"Загружено кривых блеска: {results.get('loaded_lightcurves', 0)}\n")
            f.write(f"Найдено кандидатов: {results['total_candidates']}\n")
            f.write(f"Топ кандидатов: {results['top_candidates']}\n\n")
            
            if 'error' in results:
                f.write(f"ОШИБКА: {results['error']}\n\n")
            
            # Статистика по кандидатам
            if candidates:
                periods = [c.period for c in candidates]
                depths = [c.depth for c in candidates]
                confidences = [c.confidence for c in candidates]
                
                f.write("СТАТИСТИКА КАНДИДАТОВ:\n")
                f.write(f"  Периоды: {min(periods):.3f} - {max(periods):.3f} дней\n")
                f.write(f"  Глубины: {min(depths):.4f} - {max(depths):.4f}\n")
                f.write(f"  Уверенность: {min(confidences):.3f} - {max(confidences):.3f}\n\n")
                
                # Топ-5 кандидатов
                top_candidates = sorted(candidates, key=lambda x: x.quality_score, reverse=True)[:5]
                f.write("ТОП-5 КАНДИДАТОВ:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'№':<3} {'TIC ID':<12} {'Период':<8} {'Глубина':<8} {'Уверенность':<12} {'Качество':<8}\n")
                f.write("-" * 80 + "\n")
                
                for i, candidate in enumerate(top_candidates, 1):
                    f.write(f"{i:<3} {candidate.tic_id:<12} "
                           f"{candidate.period:<8.3f} {candidate.depth:<8.4f} "
                           f"{candidate.confidence:<12.3f} {candidate.quality_score:<8.3f}\n")
            
            # Список экспортированных файлов
            f.write(f"\nЭКСПОРТИРОВАННЫЕ ФАЙЛЫ:\n")
            for file_type, filepath in results.get('exported_files', {}).items():
                f.write(f"  {file_type}: {filepath}\n")
        
        logger.info(f"Итоговый отчет создан: {filepath}")
        return str(filepath)


def main():
    """Основная функция для запуска pipeline из командной строки"""
    parser = argparse.ArgumentParser(description='Поиск экзопланет по данным TESS')
    parser.add_argument('--tic-ids', nargs='+', required=True,
                       help='Список TIC ID для анализа')
    parser.add_argument('--sectors', nargs='+', type=int,
                       help='Список секторов TESS')
    parser.add_argument('--config', default='src/config.yaml',
                       help='Путь к файлу конфигурации')
    parser.add_argument('--top-n', type=int, default=50,
                       help='Количество топ кандидатов для экспорта')
    parser.add_argument('--no-train', action='store_true',
                       help='Не обучать модели (использовать предобученные)')
    parser.add_argument('--output-dir', default='results',
                       help='Директория для сохранения результатов')
    
    args = parser.parse_args()
    
    # Инициализация pipeline
    pipeline = ExoplanetSearchPipeline(args.config)
    
    # Обновление конфигурации
    if args.output_dir:
        pipeline.config['output_dir'] = args.output_dir
        pipeline.results_exporter = ResultsExporter(output_dir=args.output_dir)
    
    # Запуск pipeline
    results = pipeline.run_full_pipeline(
        tic_ids=args.tic_ids,
        sectors=args.sectors,
        train_models=not args.no_train,
        top_n=args.top_n
    )
    
    # Вывод результатов
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ПОИСКА ЭКЗОПЛАНЕТ")
    print("="*60)
    print(f"Время выполнения: {results.get('duration', 'N/A')}")
    print(f"Анализируемые звезды: {len(results['tic_ids'])}")
    print(f"Найдено кандидатов: {results['total_candidates']}")
    print(f"Топ кандидатов: {results['top_candidates']}")
    
    if 'error' in results:
        print(f"ОШИБКА: {results['error']}")
        sys.exit(1)
    
    print(f"\nЭкспортированные файлы:")
    for file_type, filepath in results.get('exported_files', {}).items():
        print(f"  {file_type}: {filepath}")
    
    print(f"\nИтоговый отчет: {results.get('final_report', 'N/A')}")


if __name__ == "__main__":
    # Пример использования для тестирования
    logger.info("Запуск тестового примера ExoplanetSearchPipeline")
    
    # Тестовые TIC ID (известные звезды с экзопланетами)
    test_tic_ids = [
        261136679,  # TOI-700
        38846515,   # TOI-715
        142802581,  # TOI-715
    ]
    
    try:
        # Инициализация pipeline
        pipeline = ExoplanetSearchPipeline()
        
        # Запуск полного pipeline
        results = pipeline.run_full_pipeline(
            tic_ids=test_tic_ids,
            sectors=[1, 2, 3],  # Первые три сектора
            train_models=True,
            top_n=20
        )
        
        # Вывод результатов
        print(f"\nТест завершен:")
        print(f"Найдено кандидатов: {results['total_candidates']}")
        print(f"Время выполнения: {results.get('duration', 'N/A')}")
        
        if 'error' in results:
            print(f"Ошибка: {results['error']}")
        
    except Exception as e:
        logger.error(f"Ошибка в тестовом примере: {e}")
        print(f"Ошибка: {e}")
    
    logger.info("Тестовый пример завершен")
