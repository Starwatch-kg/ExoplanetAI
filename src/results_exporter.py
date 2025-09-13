"""
Модуль вывода результатов поиска экзопланет
Сохраняет кандидатов в CSV/JSON и создает топ-N список наиболее вероятных экзопланет
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ExoplanetCandidate:
    """
    Класс для представления кандидата в экзопланеты
    """
    
    def __init__(self, tic_id: Union[int, str], 
                 period: float,
                 depth: float,
                 duration: float,
                 start_time: float,
                 end_time: float,
                 confidence: float,
                 combined_score: float,
                 anomaly_probability: float,
                 star_info: Optional[Dict] = None,
                 transit_model: Optional[np.ndarray] = None,
                 metadata: Optional[Dict] = None):
        """
        Инициализация кандидата
        
        Args:
            tic_id: TIC ID звезды
            period: Период транзита (дни)
            depth: Глубина транзита
            duration: Длительность транзита (дни)
            start_time: Время начала транзита
            end_time: Время окончания транзита
            confidence: Уверенность в обнаружении
            combined_score: Комбинированная оценка
            anomaly_probability: Вероятность аномалии
            star_info: Информация о звезде
            transit_model: Модель транзита
            metadata: Дополнительные метаданные
        """
        self.tic_id = tic_id
        self.period = period
        self.depth = depth
        self.duration = duration
        self.start_time = start_time
        self.end_time = end_time
        self.confidence = confidence
        self.combined_score = combined_score
        self.anomaly_probability = anomaly_probability
        self.star_info = star_info or {}
        self.transit_model = transit_model
        self.metadata = metadata or {}
        
        # Вычисление дополнительных параметров
        self.transit_depth_mag = -2.5 * np.log10(1 - depth)  # Глубина в звездных величинах
        self.snr = self._calculate_snr()
        self.quality_score = self._calculate_quality_score()
    
    def _calculate_snr(self) -> float:
        """Вычисляет отношение сигнал/шум"""
        if self.depth > 0:
            # Простая оценка SNR на основе глубины транзита
            snr = self.depth / 0.01  # Предполагаем шум ~1%
            return min(snr, 100)  # Ограничиваем максимальное значение
        return 0.0
    
    def _calculate_quality_score(self) -> float:
        """Вычисляет общий показатель качества кандидата"""
        # Комбинируем различные факторы
        factors = [
            self.confidence * 0.3,
            self.combined_score * 0.3,
            self.anomaly_probability * 0.2,
            min(self.snr / 10, 1.0) * 0.2  # Нормализованный SNR
        ]
        
        return sum(factors)
    
    def to_dict(self) -> Dict:
        """Преобразует кандидата в словарь"""
        return {
            'tic_id': self.tic_id,
            'period': self.period,
            'depth': self.depth,
            'duration': self.duration,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'confidence': self.confidence,
            'combined_score': self.combined_score,
            'anomaly_probability': self.anomaly_probability,
            'transit_depth_mag': self.transit_depth_mag,
            'snr': self.snr,
            'quality_score': self.quality_score,
            'star_info': self.star_info,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        return (f"ExoplanetCandidate(TIC={self.tic_id}, "
                f"P={self.period:.3f}d, "
                f"depth={self.depth:.4f}, "
                f"confidence={self.confidence:.3f})")


class ResultsExporter:
    """
    Класс для экспорта результатов поиска экзопланет
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Инициализация экспортера
        
        Args:
            output_dir: Директория для сохранения результатов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Создание поддиректорий
        self.candidates_dir = self.output_dir / "candidates"
        self.plots_dir = self.output_dir / "plots"
        self.summaries_dir = self.output_dir / "summaries"
        
        for dir_path in [self.candidates_dir, self.plots_dir, self.summaries_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def save_candidates_csv(self, candidates: List[ExoplanetCandidate], 
                           filename: str = "exoplanet_candidates.csv") -> str:
        """
        Сохраняет кандидатов в CSV файл
        
        Args:
            candidates: Список кандидатов
            filename: Имя файла
            
        Returns:
            str: Путь к сохраненному файлу
        """
        logger.info(f"Сохраняем {len(candidates)} кандидатов в CSV")
        
        # Преобразование в DataFrame
        data = []
        for candidate in candidates:
            row = candidate.to_dict()
            
            # Разворачиваем star_info в отдельные колонки
            star_info = row.pop('star_info', {})
            for key, value in star_info.items():
                row[f'star_{key}'] = value
            
            # Разворачиваем metadata в отдельные колонки
            metadata = row.pop('metadata', {})
            for key, value in metadata.items():
                row[f'meta_{key}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Сортировка по качеству
        df = df.sort_values('quality_score', ascending=False)
        
        # Сохранение
        filepath = self.candidates_dir / filename
        df.to_csv(filepath, index=False)
        
        logger.info(f"Кандидаты сохранены: {filepath}")
        return str(filepath)
    
    def save_candidates_json(self, candidates: List[ExoplanetCandidate],
                            filename: str = "exoplanet_candidates.json") -> str:
        """
        Сохраняет кандидатов в JSON файл
        
        Args:
            candidates: Список кандидатов
            filename: Имя файла
            
        Returns:
            str: Путь к сохраненному файлу
        """
        logger.info(f"Сохраняем {len(candidates)} кандидатов в JSON")
        
        # Преобразование в список словарей
        data = [candidate.to_dict() for candidate in candidates]
        
        # Добавление метаданных файла
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_candidates': len(candidates),
            'candidates': data
        }
        
        # Сохранение
        filepath = self.candidates_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Кандидаты сохранены: {filepath}")
        return str(filepath)
    
    def create_top_candidates_list(self, candidates: List[ExoplanetCandidate],
                                  top_n: int = 50) -> List[ExoplanetCandidate]:
        """
        Создает список топ-N наиболее вероятных кандидатов
        
        Args:
            candidates: Список всех кандидатов
            top_n: Количество топ кандидатов
            
        Returns:
            List[ExoplanetCandidate]: Список топ кандидатов
        """
        logger.info(f"Создаем список топ-{top_n} кандидатов")
        
        # Сортировка по качеству
        sorted_candidates = sorted(candidates, key=lambda x: x.quality_score, reverse=True)
        
        # Отбор топ-N
        top_candidates = sorted_candidates[:top_n]
        
        logger.info(f"Отобрано {len(top_candidates)} топ кандидатов")
        return top_candidates
    
    def save_top_candidates(self, candidates: List[ExoplanetCandidate],
                          top_n: int = 50,
                          filename_prefix: str = "top_candidates") -> Dict[str, str]:
        """
        Сохраняет топ кандидатов в различных форматах
        
        Args:
            candidates: Список всех кандидатов
            top_n: Количество топ кандидатов
            filename_prefix: Префикс имени файла
            
        Returns:
            Dict[str, str]: Пути к сохраненным файлам
        """
        top_candidates = self.create_top_candidates_list(candidates, top_n)
        
        saved_files = {}
        
        # Сохранение в CSV
        csv_file = f"{filename_prefix}_{top_n}.csv"
        saved_files['csv'] = self.save_candidates_csv(top_candidates, csv_file)
        
        # Сохранение в JSON
        json_file = f"{filename_prefix}_{top_n}.json"
        saved_files['json'] = self.save_candidates_json(top_candidates, json_file)
        
        # Создание краткого отчета
        report_file = f"{filename_prefix}_{top_n}_report.txt"
        saved_files['report'] = self._create_summary_report(top_candidates, report_file)
        
        return saved_files
    
    def _create_summary_report(self, candidates: List[ExoplanetCandidate],
                             filename: str) -> str:
        """
        Создает краткий текстовый отчет
        
        Args:
            candidates: Список кандидатов
            filename: Имя файла
            
        Returns:
            str: Путь к файлу отчета
        """
        filepath = self.summaries_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ О ПОИСКЕ ЭКЗОПЛАНЕТ\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Общее количество кандидатов: {len(candidates)}\n\n")
            
            # Статистика по периодам
            periods = [c.period for c in candidates]
            f.write("СТАТИСТИКА ПО ПЕРИОДАМ:\n")
            f.write(f"  Минимальный период: {min(periods):.3f} дней\n")
            f.write(f"  Максимальный период: {max(periods):.3f} дней\n")
            f.write(f"  Средний период: {np.mean(periods):.3f} дней\n")
            f.write(f"  Медианный период: {np.median(periods):.3f} дней\n\n")
            
            # Статистика по глубинам
            depths = [c.depth for c in candidates]
            f.write("СТАТИСТИКА ПО ГЛУБИНАМ ТРАНЗИТОВ:\n")
            f.write(f"  Минимальная глубина: {min(depths):.4f}\n")
            f.write(f"  Максимальная глубина: {max(depths):.4f}\n")
            f.write(f"  Средняя глубина: {np.mean(depths):.4f}\n")
            f.write(f"  Медианная глубина: {np.median(depths):.4f}\n\n")
            
            # Статистика по уверенности
            confidences = [c.confidence for c in candidates]
            f.write("СТАТИСТИКА ПО УВЕРЕННОСТИ:\n")
            f.write(f"  Минимальная уверенность: {min(confidences):.3f}\n")
            f.write(f"  Максимальная уверенность: {max(confidences):.3f}\n")
            f.write(f"  Средняя уверенность: {np.mean(confidences):.3f}\n")
            f.write(f"  Медианная уверенность: {np.median(confidences):.3f}\n\n")
            
            # Топ-10 кандидатов
            f.write("ТОП-10 КАНДИДАТОВ:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'№':<3} {'TIC ID':<12} {'Период':<8} {'Глубина':<8} {'Уверенность':<12} {'Качество':<8}\n")
            f.write("-" * 80 + "\n")
            
            for i, candidate in enumerate(candidates[:10], 1):
                f.write(f"{i:<3} {candidate.tic_id:<12} "
                       f"{candidate.period:<8.3f} {candidate.depth:<8.4f} "
                       f"{candidate.confidence:<12.3f} {candidate.quality_score:<8.3f}\n")
        
        logger.info(f"Отчет создан: {filepath}")
        return str(filepath)
    
    def create_visualization_plots(self, candidates: List[ExoplanetCandidate],
                                 filename_prefix: str = "candidates_analysis") -> Dict[str, str]:
        """
        Создает визуализации для анализа кандидатов
        
        Args:
            candidates: Список кандидатов
            filename_prefix: Префикс имени файла
            
        Returns:
            Dict[str, str]: Пути к созданным графикам
        """
        logger.info("Создаем визуализации для анализа кандидатов")
        
        plot_files = {}
        
        # График распределения периодов
        periods = [c.period for c in candidates]
        plt.figure(figsize=(10, 6))
        plt.hist(periods, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Период (дни)')
        plt.ylabel('Количество кандидатов')
        plt.title('Распределение периодов транзитов')
        plt.grid(True, alpha=0.3)
        
        period_plot = self.plots_dir / f"{filename_prefix}_periods.png"
        plt.savefig(period_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['periods'] = str(period_plot)
        
        # График распределения глубин
        depths = [c.depth for c in candidates]
        plt.figure(figsize=(10, 6))
        plt.hist(depths, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Глубина транзита')
        plt.ylabel('Количество кандидатов')
        plt.title('Распределение глубин транзитов')
        plt.grid(True, alpha=0.3)
        
        depth_plot = self.plots_dir / f"{filename_prefix}_depths.png"
        plt.savefig(depth_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['depths'] = str(depth_plot)
        
        # График уверенности vs качества
        confidences = [c.confidence for c in candidates]
        quality_scores = [c.quality_score for c in candidates]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(confidences, quality_scores, alpha=0.6)
        plt.xlabel('Уверенность')
        plt.ylabel('Показатель качества')
        plt.title('Уверенность vs Показатель качества')
        plt.grid(True, alpha=0.3)
        
        # Добавляем цветовую кодировку по периоду
        scatter = plt.scatter(confidences, quality_scores, c=periods, 
                            cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Период (дни)')
        
        quality_plot = self.plots_dir / f"{filename_prefix}_quality.png"
        plt.savefig(quality_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['quality'] = str(quality_plot)
        
        # Корреляционная матрица
        df = pd.DataFrame([c.to_dict() for c in candidates])
        numeric_cols = ['period', 'depth', 'duration', 'confidence', 
                        'combined_score', 'anomaly_probability', 'snr', 'quality_score']
        
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        if HAS_SEABORN:
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.3f')
        else:
            # Fallback без seaborn
            plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
            plt.yticks(range(len(numeric_cols)), numeric_cols)
            # Добавляем значения на график
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.3f}', 
                            ha='center', va='center', fontsize=8)
        plt.title('Корреляционная матрица параметров кандидатов')
        
        correlation_plot = self.plots_dir / f"{filename_prefix}_correlation.png"
        plt.savefig(correlation_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['correlation'] = str(correlation_plot)
        
        logger.info(f"Создано {len(plot_files)} графиков")
        return plot_files
    
    def export_complete_results(self, candidates: List[ExoplanetCandidate],
                              top_n: int = 50,
                              create_plots: bool = True) -> Dict[str, str]:
        """
        Экспортирует полные результаты поиска
        
        Args:
            candidates: Список всех кандидатов
            top_n: Количество топ кандидатов
            create_plots: Создавать ли графики
            
        Returns:
            Dict[str, str]: Пути к созданным файлам
        """
        logger.info("Начинаем полный экспорт результатов")
        
        exported_files = {}
        
        # Сохранение всех кандидатов
        exported_files['all_candidates_csv'] = self.save_candidates_csv(candidates)
        exported_files['all_candidates_json'] = self.save_candidates_json(candidates)
        
        # Сохранение топ кандидатов
        top_files = self.save_top_candidates(candidates, top_n)
        exported_files.update(top_files)
        
        # Создание визуализаций
        if create_plots:
            plot_files = self.create_visualization_plots(candidates)
            exported_files.update(plot_files)
        
        # Создание общего отчета
        summary_file = self._create_general_summary(candidates, top_n)
        exported_files['general_summary'] = summary_file
        
        logger.info(f"Экспорт завершен. Создано {len(exported_files)} файлов")
        return exported_files
    
    def _create_general_summary(self, candidates: List[ExoplanetCandidate],
                              top_n: int) -> str:
        """
        Создает общий сводный отчет
        
        Args:
            candidates: Список кандидатов
            top_n: Количество топ кандидатов
            
        Returns:
            str: Путь к файлу отчета
        """
        filename = f"general_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = self.summaries_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("ОБЩИЙ ОТЧЕТ О ПОИСКЕ ЭКЗОПЛАНЕТ\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Общее количество кандидатов: {len(candidates)}\n")
            f.write(f"Количество топ кандидатов: {min(top_n, len(candidates))}\n\n")
            
            # Общая статистика
            if candidates:
                periods = [c.period for c in candidates]
                depths = [c.depth for c in candidates]
                confidences = [c.confidence for c in candidates]
                quality_scores = [c.quality_score for c in candidates]
                
                f.write("ОБЩАЯ СТАТИСТИКА:\n")
                f.write(f"  Периоды: {min(periods):.3f} - {max(periods):.3f} дней "
                       f"(среднее: {np.mean(periods):.3f})\n")
                f.write(f"  Глубины: {min(depths):.4f} - {max(depths):.4f} "
                       f"(среднее: {np.mean(depths):.4f})\n")
                f.write(f"  Уверенность: {min(confidences):.3f} - {max(confidences):.3f} "
                       f"(среднее: {np.mean(confidences):.3f})\n")
                f.write(f"  Качество: {min(quality_scores):.3f} - {max(quality_scores):.3f} "
                       f"(среднее: {np.mean(quality_scores):.3f})\n\n")
                
                # Рекомендации
                f.write("РЕКОМЕНДАЦИИ:\n")
                high_confidence = [c for c in candidates if c.confidence > 0.8]
                if high_confidence:
                    f.write(f"  - {len(high_confidence)} кандидатов с высокой уверенностью (>0.8)\n")
                
                short_period = [c for c in candidates if c.period < 10]
                if short_period:
                    f.write(f"  - {len(short_period)} кандидатов с коротким периодом (<10 дней)\n")
                
                deep_transits = [c for c in candidates if c.depth > 0.01]
                if deep_transits:
                    f.write(f"  - {len(deep_transits)} кандидатов с глубокими транзитами (>1%)\n")
        
        logger.info(f"Общий отчет создан: {filepath}")
        return str(filepath)


def load_candidates_from_csv(filepath: str) -> List[ExoplanetCandidate]:
    """
    Загружает кандидатов из CSV файла
    
    Args:
        filepath: Путь к CSV файлу
        
    Returns:
        List[ExoplanetCandidate]: Список кандидатов
    """
    logger.info(f"Загружаем кандидатов из {filepath}")
    
    df = pd.read_csv(filepath)
    candidates = []
    
    for _, row in df.iterrows():
        # Извлечение star_info
        star_info = {}
        for col in df.columns:
            if col.startswith('star_'):
                star_info[col[5:]] = row[col]
        
        # Извлечение metadata
        metadata = {}
        for col in df.columns:
            if col.startswith('meta_'):
                metadata[col[5:]] = row[col]
        
        candidate = ExoplanetCandidate(
            tic_id=row['tic_id'],
            period=row['period'],
            depth=row['depth'],
            duration=row['duration'],
            start_time=row['start_time'],
            end_time=row['end_time'],
            confidence=row['confidence'],
            combined_score=row['combined_score'],
            anomaly_probability=row['anomaly_probability'],
            star_info=star_info,
            metadata=metadata
        )
        
        candidates.append(candidate)
    
    logger.info(f"Загружено {len(candidates)} кандидатов")
    return candidates


def load_candidates_from_json(filepath: str) -> List[ExoplanetCandidate]:
    """
    Загружает кандидатов из JSON файла
    
    Args:
        filepath: Путь к JSON файлу
        
    Returns:
        List[ExoplanetCandidate]: Список кандидатов
    """
    logger.info(f"Загружаем кандидатов из {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    candidates = []
    for candidate_data in data['candidates']:
        candidate = ExoplanetCandidate(
            tic_id=candidate_data['tic_id'],
            period=candidate_data['period'],
            depth=candidate_data['depth'],
            duration=candidate_data['duration'],
            start_time=candidate_data['start_time'],
            end_time=candidate_data['end_time'],
            confidence=candidate_data['confidence'],
            combined_score=candidate_data['combined_score'],
            anomaly_probability=candidate_data['anomaly_probability'],
            star_info=candidate_data.get('star_info', {}),
            metadata=candidate_data.get('metadata', {})
        )
        candidates.append(candidate)
    
    logger.info(f"Загружено {len(candidates)} кандидатов")
    return candidates


if __name__ == "__main__":
    # Пример использования
    logger.info("Тестирование модуля экспорта результатов")
    
    # Создание тестовых кандидатов
    test_candidates = []
    
    for i in range(20):
        candidate = ExoplanetCandidate(
            tic_id=f"TIC_{1000000 + i}",
            period=np.random.uniform(1, 50),
            depth=np.random.uniform(0.001, 0.05),
            duration=np.random.uniform(0.1, 2.0),
            start_time=np.random.uniform(0, 100),
            end_time=np.random.uniform(100, 200),
            confidence=np.random.uniform(0.3, 0.95),
            combined_score=np.random.uniform(0.2, 0.9),
            anomaly_probability=np.random.uniform(0.1, 0.8),
            star_info={
                'ra': np.random.uniform(0, 360),
                'dec': np.random.uniform(-90, 90),
                'tmag': np.random.uniform(8, 15)
            },
            metadata={
                'sector': np.random.randint(1, 30),
                'method': 'hybrid_search'
            }
        )
        test_candidates.append(candidate)
    
    # Создание экспортера
    exporter = ResultsExporter()
    
    # Экспорт результатов
    exported_files = exporter.export_complete_results(test_candidates, top_n=10)
    
    # Вывод результатов
    print("Созданные файлы:")
    for file_type, filepath in exported_files.items():
        print(f"  {file_type}: {filepath}")
    
    logger.info("Тест завершен")
