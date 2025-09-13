"""
Модуль пайплайна поиска экзопланет.

Этот модуль содержит основной пайплайн: Hybrid BLS → Representation → Anomaly Score
для поиска кандидатов в экзопланеты.
"""

import logging
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import json
from datetime import datetime

# Импорты из других модулей
from preprocess import TESSDataProcessor, load_multiple_stars
from model import ExoplanetAutoencoder, ExoplanetClassifier, ModelTrainer
from utils import BoxLeastSquares, calculate_metrics

# Настройка логирования
logger = logging.getLogger(__name__)


class ExoplanetPipeline:
    """Основной пайплайн поиска экзопланет."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Инициализация пайплайна.
        
        Args:
            config: Конфигурация пайплайна.
        """
        self.config = config or self._get_default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Инициализация компонентов
        self.data_processor = TESSDataProcessor()
        self.bls = BoxLeastSquares(
            period_range=self.config['bls']['period_range'],
            nperiods=self.config['bls']['nperiods']
        )
        self.model_trainer = ModelTrainer(device=str(self.device))
        
        # Модели (будут загружены при необходимости)
        self.autoencoder = None
        self.classifier = None
        
        logger.info(f"Пайплайн инициализирован на устройстве: {self.device}")
    
    def _get_default_config(self) -> Dict:
        """Возвращает конфигурацию по умолчанию."""
        return {
            'bls': {
                'period_range': [0.5, 50.0],
                'nperiods': 1000
            },
            'model': {
                'type': 'autoencoder',
                'latent_dim': 64,
                'hidden_dim': 128
            },
            'anomaly': {
                'threshold': 0.5,
                'min_period': 0.5,
                'max_period': 50.0
            },
            'output': {
                'save_candidates': True,
                'save_plots': True,
                'output_dir': 'results'
            }
        }
    
    def load_models(self, autoencoder_path: Optional[str] = None,
                   classifier_path: Optional[str] = None):
        """
        Загрузка предобученных моделей.
        
        Args:
            autoencoder_path: Путь к автоэнкодеру.
            classifier_path: Путь к классификатору.
        """
        if autoencoder_path and Path(autoencoder_path).exists():
            self.autoencoder = ExoplanetAutoencoder(
                input_length=self.config['model']['input_length'],
                latent_dim=self.config['model']['latent_dim'],
                hidden_dim=self.config['model']['hidden_dim']
            )
            self.model_trainer.load_model(self.autoencoder, autoencoder_path)
            logger.info("Автоэнкодер загружен")
        
        if classifier_path and Path(classifier_path).exists():
            self.classifier = ExoplanetClassifier(
                input_length=self.config['model']['input_length'],
                num_classes=2,
                hidden_dim=self.config['model']['hidden_dim']
            )
            self.model_trainer.load_model(self.classifier, classifier_path)
            logger.info("Классификатор загружен")
    
    def search_exoplanets(self, tic_ids: List[Union[int, str]],
                         sectors: Optional[List[int]] = None,
                         use_models: bool = True) -> Dict:
        """
        Поиск экзопланет для списка звезд.
        
        Args:
            tic_ids: Список TIC ID для анализа.
            sectors: Список секторов TESS.
            use_models: Использовать ли нейронные модели.
            
        Returns:
            Dict: Результаты поиска с кандидатами и метриками.
        """
        logger.info(f"Начинаем поиск экзопланет для {len(tic_ids)} звезд")
        
        results = {
            'tic_ids': tic_ids,
            'candidates': [],
            'metrics': {},
            'processing_time': None,
            'errors': []
        }
        
        start_time = datetime.now()
        
        try:
            # 1. Загрузка данных
            logger.info("Этап 1: Загрузка данных TESS")
            lightcurves = load_multiple_stars(tic_ids, sectors)
            
            if not lightcurves:
                raise ValueError("Не удалось загрузить данные TESS")
            
            results['loaded_stars'] = len(lightcurves)
            
            # 2. Поиск кандидатов для каждой звезды
            logger.info("Этап 2: Поиск кандидатов")
            all_candidates = []
            
            for i, ((times, fluxes), tic_id) in enumerate(zip(lightcurves, tic_ids)):
                try:
                    logger.info(f"Обработка {i+1}/{len(lightcurves)}: TIC {tic_id}")
                    
                    # Гибридный поиск: BLS + Neural Models
                    candidates = self._find_candidates_for_star(
                        tic_id, times, fluxes, use_models
                    )
                    
                    all_candidates.extend(candidates)
                    logger.info(f"Найдено {len(candidates)} кандидатов для TIC {tic_id}")
                    
                except Exception as e:
                    error_msg = f"Ошибка обработки TIC {tic_id}: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
                    continue
            
            results['candidates'] = all_candidates
            results['total_candidates'] = len(all_candidates)
            
            # 3. Вычисление метрик
            logger.info("Этап 3: Вычисление метрик")
            if all_candidates:
                results['metrics'] = self._calculate_search_metrics(all_candidates)
            
            # 4. Сохранение результатов
            logger.info("Этап 4: Сохранение результатов")
            if self.config['output']['save_candidates']:
                self._save_results(results)
            
            end_time = datetime.now()
            results['processing_time'] = str(end_time - start_time)
            
            logger.info(f"Поиск завершен. Найдено {len(all_candidates)} кандидатов")
            
        except Exception as e:
            logger.error(f"Критическая ошибка в пайплайне: {e}")
            results['errors'].append(f"Критическая ошибка: {e}")
        
        return results
    
    def _find_candidates_for_star(self, tic_id: Union[int, str],
                                times: np.ndarray,
                                fluxes: np.ndarray,
                                use_models: bool) -> List[Dict]:
        """
        Поиск кандидатов для одной звезды.
        
        Args:
            tic_id: TIC ID звезды.
            times: Временные метки.
            fluxes: Потоки.
            use_models: Использовать ли нейронные модели.
            
        Returns:
            List[Dict]: Список кандидатов.
        """
        candidates = []
        
        # 1. BLS поиск
        logger.debug(f"BLS поиск для TIC {tic_id}")
        bls_results = self.bls.compute_periodogram(times, fluxes)
        
        # Обработка результатов BLS
        if bls_results['best_power'] > 0.1:  # Порог значимости
            bls_candidate = {
                'tic_id': tic_id,
                'method': 'BLS',
                'period': bls_results['best_period'],
                'power': bls_results['best_power'],
                'depth': bls_results['best_params'].get('depth', 0.0),
                'duration': bls_results['best_params'].get('duration', 0.0),
                'confidence': min(bls_results['best_power'] * 2, 1.0),
                'anomaly_score': 0.0
            }
            candidates.append(bls_candidate)
        
        # 2. Neural Models поиск (если модели загружены)
        if use_models and (self.autoencoder is not None or self.classifier is not None):
            logger.debug(f"Neural поиск для TIC {tic_id}")
            neural_candidates = self._neural_search(times, fluxes, tic_id)
            candidates.extend(neural_candidates)
        
        return candidates
    
    def _neural_search(self, times: np.ndarray,
                      fluxes: np.ndarray,
                      tic_id: Union[int, str]) -> List[Dict]:
        """
        Поиск кандидатов с использованием нейронных моделей.
        
        Args:
            times: Временные метки.
            fluxes: Потоки.
            tic_id: TIC ID звезды.
            
        Returns:
            List[Dict]: Список кандидатов от нейронных моделей.
        """
        candidates = []
        
        # Нормализация данных
        normalized_fluxes = self.data_processor.normalize_data(fluxes)
        
        # Подготовка данных для модели
        window_size = 2000
        if len(normalized_fluxes) < window_size:
            # Дополнение данных если слишком короткие
            padding = window_size - len(normalized_fluxes)
            normalized_fluxes = np.pad(normalized_fluxes, (0, padding), mode='edge')
        
        # Создание окон для анализа
        stride = window_size // 4
        windows = []
        window_times = []
        
        for start in range(0, len(normalized_fluxes) - window_size + 1, stride):
            window = normalized_fluxes[start:start + window_size]
            windows.append(window)
            window_times.append(times[start + window_size // 2] if start + window_size // 2 < len(times) else times[-1])
        
        if not windows:
            return candidates
        
        windows_array = np.array(windows)
        
        # Анализ с автоэнкодером
        if self.autoencoder is not None:
            try:
                anomaly_scores = self._compute_anomaly_scores(windows_array)
                
                # Поиск аномальных окон
                threshold = self.config['anomaly']['threshold']
                anomalous_indices = np.where(anomaly_scores > threshold)[0]
                
                for idx in anomalous_indices:
                    candidate = {
                        'tic_id': tic_id,
                        'method': 'Autoencoder',
                        'period': 0.0,  # Будет определен позже
                        'power': anomaly_scores[idx],
                        'depth': 0.0,
                        'duration': 0.0,
                        'confidence': min(anomaly_scores[idx] * 2, 1.0),
                        'anomaly_score': anomaly_scores[idx],
                        'window_time': window_times[idx]
                    }
                    candidates.append(candidate)
                    
            except Exception as e:
                logger.error(f"Ошибка автоэнкодера для TIC {tic_id}: {e}")
        
        # Анализ с классификатором
        if self.classifier is not None:
            try:
                predictions = self._compute_classifier_predictions(windows_array)
                
                # Поиск окон с предсказанными транзитами
                transit_indices = np.where(predictions[:, 1] > 0.5)[0]
                
                for idx in transit_indices:
                    candidate = {
                        'tic_id': tic_id,
                        'method': 'Classifier',
                        'period': 0.0,  # Будет определен позже
                        'power': predictions[idx, 1],
                        'depth': 0.0,
                        'duration': 0.0,
                        'confidence': predictions[idx, 1],
                        'anomaly_score': predictions[idx, 1],
                        'window_time': window_times[idx]
                    }
                    candidates.append(candidate)
                    
            except Exception as e:
                logger.error(f"Ошибка классификатора для TIC {tic_id}: {e}")
        
        return candidates
    
    def _compute_anomaly_scores(self, windows: np.ndarray) -> np.ndarray:
        """
        Вычисление оценок аномальности с помощью автоэнкодера.
        
        Args:
            windows: Окна данных для анализа.
            
        Returns:
            np.ndarray: Оценки аномальности.
        """
        self.autoencoder.eval()
        
        with torch.no_grad():
            windows_tensor = torch.tensor(windows, dtype=torch.float32).unsqueeze(1).to(self.device)
            reconstructed, latent = self.autoencoder(windows_tensor)
            
            # Вычисление ошибки реконструкции
            reconstruction_error = torch.mean((windows_tensor - reconstructed) ** 2, dim=(1, 2))
            
            # Нормализация ошибки
            anomaly_scores = reconstruction_error.cpu().numpy()
            anomaly_scores = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores) + 1e-8)
            
        return anomaly_scores
    
    def _compute_classifier_predictions(self, windows: np.ndarray) -> np.ndarray:
        """
        Вычисление предсказаний классификатора.
        
        Args:
            windows: Окна данных для анализа.
            
        Returns:
            np.ndarray: Предсказания классов.
        """
        self.classifier.eval()
        
        with torch.no_grad():
            windows_tensor = torch.tensor(windows, dtype=torch.float32).unsqueeze(1).to(self.device)
            predictions = self.classifier(windows_tensor)
            probabilities = torch.softmax(predictions, dim=1)
            
        return probabilities.cpu().numpy()
    
    def _calculate_search_metrics(self, candidates: List[Dict]) -> Dict:
        """
        Вычисление метрик поиска.
        
        Args:
            candidates: Список кандидатов.
            
        Returns:
            Dict: Метрики поиска.
        """
        if not candidates:
            return {}
        
        # Статистика по методам
        methods = [c['method'] for c in candidates]
        method_counts = {method: methods.count(method) for method in set(methods)}
        
        # Статистика по периодам
        periods = [c['period'] for c in candidates if c['period'] > 0]
        period_stats = {
            'min_period': min(periods) if periods else 0,
            'max_period': max(periods) if periods else 0,
            'mean_period': np.mean(periods) if periods else 0,
            'median_period': np.median(periods) if periods else 0
        }
        
        # Статистика по уверенности
        confidences = [c['confidence'] for c in candidates]
        confidence_stats = {
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'mean_confidence': np.mean(confidences),
            'median_confidence': np.median(confidences)
        }
        
        # Топ кандидаты
        top_candidates = sorted(candidates, key=lambda x: x['confidence'], reverse=True)[:10]
        
        metrics = {
            'total_candidates': len(candidates),
            'method_distribution': method_counts,
            'period_statistics': period_stats,
            'confidence_statistics': confidence_stats,
            'top_candidates': top_candidates
        }
        
        return metrics
    
    def _save_results(self, results: Dict):
        """
        Сохранение результатов поиска.
        
        Args:
            results: Результаты поиска.
        """
        output_dir = Path(self.config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Сохранение кандидатов в JSON
        candidates_file = output_dir / f"candidates_{timestamp}.json"
        with open(candidates_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Сохранение текстового отчета
        report_file = output_dir / f"search_report_{timestamp}.txt"
        self._save_text_report(results, report_file)
        
        logger.info(f"Результаты сохранены в {output_dir}")
    
    def _save_text_report(self, results: Dict, filepath: Path):
        """
        Сохранение текстового отчета.
        
        Args:
            results: Результаты поиска.
            filepath: Путь к файлу отчета.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ О ПОИСКЕ ЭКЗОПЛАНЕТ\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Дата поиска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Время обработки: {results.get('processing_time', 'N/A')}\n")
            f.write(f"Анализируемые звезды: {len(results['tic_ids'])}\n")
            f.write(f"Загружено звезд: {results.get('loaded_stars', 0)}\n")
            f.write(f"Найдено кандидатов: {results.get('total_candidates', 0)}\n\n")
            
            # Ошибки
            if results.get('errors'):
                f.write("ОШИБКИ:\n")
                for error in results['errors']:
                    f.write(f"  - {error}\n")
                f.write("\n")
            
            # Метрики
            if results.get('metrics'):
                metrics = results['metrics']
                f.write("МЕТРИКИ ПОИСКА:\n")
                f.write(f"  Общее количество кандидатов: {metrics.get('total_candidates', 0)}\n")
                
                # Распределение по методам
                method_dist = metrics.get('method_distribution', {})
                f.write("  Распределение по методам:\n")
                for method, count in method_dist.items():
                    f.write(f"    {method}: {count}\n")
                
                # Статистика периодов
                period_stats = metrics.get('period_statistics', {})
                f.write("  Статистика периодов:\n")
                f.write(f"    Минимальный: {period_stats.get('min_period', 0):.3f} дней\n")
                f.write(f"    Максимальный: {period_stats.get('max_period', 0):.3f} дней\n")
                f.write(f"    Средний: {period_stats.get('mean_period', 0):.3f} дней\n")
                
                # Статистика уверенности
                conf_stats = metrics.get('confidence_statistics', {})
                f.write("  Статистика уверенности:\n")
                f.write(f"    Минимальная: {conf_stats.get('min_confidence', 0):.3f}\n")
                f.write(f"    Максимальная: {conf_stats.get('max_confidence', 0):.3f}\n")
                f.write(f"    Средняя: {conf_stats.get('mean_confidence', 0):.3f}\n")
                
                # Топ кандидаты
                top_candidates = metrics.get('top_candidates', [])
                if top_candidates:
                    f.write("\nТОП-10 КАНДИДАТОВ:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{'№':<3} {'TIC ID':<12} {'Метод':<12} {'Период':<8} {'Уверенность':<12}\n")
                    f.write("-" * 80 + "\n")
                    
                    for i, candidate in enumerate(top_candidates[:10], 1):
                        f.write(f"{i:<3} {candidate['tic_id']:<12} "
                               f"{candidate['method']:<12} "
                               f"{candidate['period']:<8.3f} "
                               f"{candidate['confidence']:<12.3f}\n")


def search_exoplanets(tic_ids: List[Union[int, str]],
                     sectors: Optional[List[int]] = None,
                     config: Optional[Dict] = None,
                     autoencoder_path: Optional[str] = None,
                     classifier_path: Optional[str] = None) -> Dict:
    """
    Основная функция поиска экзопланет.
    
    Args:
        tic_ids: Список TIC ID для анализа.
        sectors: Список секторов TESS.
        config: Конфигурация пайплайна.
        autoencoder_path: Путь к автоэнкодеру.
        classifier_path: Путь к классификатору.
        
    Returns:
        Dict: Результаты поиска.
    """
    logger.info(f"Запуск поиска экзопланет для {len(tic_ids)} звезд")
    
    # Создание пайплайна
    pipeline = ExoplanetPipeline(config)
    
    # Загрузка моделей
    pipeline.load_models(autoencoder_path, classifier_path)
    
    # Поиск экзопланет
    results = pipeline.search_exoplanets(tic_ids, sectors)
    
    logger.info("Поиск экзопланет завершен")
    return results
