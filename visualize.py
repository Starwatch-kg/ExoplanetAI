"""
Модуль визуализации результатов поиска экзопланет.

Этот модуль содержит функции для создания графиков световых кривых,
периодограмм, ROC-кривых и других визуализаций.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import pandas as pd

# Настройка логирования
logger = logging.getLogger(__name__)

# Настройка стиля графиков
plt.style.use('default')
sns.set_palette("husl")


class ExoplanetVisualizer:
    """Класс для визуализации результатов поиска экзопланет."""
    
    def __init__(self, output_dir: str = "results/plots"):
        """
        Инициализация визуализатора.
        
        Args:
            output_dir: Директория для сохранения графиков.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройки графиков
        self.figure_size = (12, 8)
        self.dpi = 300
        
    def plot_lightcurve(self, times: np.ndarray,
                       fluxes: np.ndarray,
                       title: str = "Light Curve",
                       candidates: Optional[List[Dict]] = None,
                       save_path: Optional[str] = None) -> None:
        """
        Построение графика световой кривой.
        
        Args:
            times: Временные метки.
            fluxes: Потоки.
            title: Заголовок графика.
            candidates: Список кандидатов для выделения.
            save_path: Путь для сохранения графика.
        """
        logger.info(f"Создание графика световой кривой: {title}")
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Основная кривая блеска
        ax.plot(times, fluxes, 'b-', alpha=0.7, linewidth=0.8, label='Light Curve')
        
        # Выделение кандидатов
        if candidates:
            for i, candidate in enumerate(candidates):
                if 'window_time' in candidate:
                    # Для нейронных моделей
                    ax.axvline(candidate['window_time'], color='red', 
                             alpha=0.7, linestyle='--', 
                             label=f"Candidate {i+1}" if i < 5 else "")
                elif candidate.get('period', 0) > 0:
                    # Для BLS кандидатов - показываем несколько периодов
                    period = candidate['period']
                    start_time = times[0]
                    for cycle in range(3):  # Показываем 3 цикла
                        transit_time = start_time + cycle * period
                        if transit_time <= times[-1]:
                            ax.axvline(transit_time, color='red', alpha=0.7, 
                                     linestyle='--', linewidth=2)
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Normalized Flux')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Сохранение
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"График сохранен: {save_path}")
        
        plt.close()
    
    def plot_periodogram(self, periods: np.ndarray,
                        powers: np.ndarray,
                        title: str = "Periodogram",
                        best_period: Optional[float] = None,
                        save_path: Optional[str] = None) -> None:
        """
        Построение графика периодограммы.
        
        Args:
            periods: Периоды.
            powers: Мощности.
            title: Заголовок графика.
            best_period: Лучший период для выделения.
            save_path: Путь для сохранения графика.
        """
        logger.info(f"Создание графика периодограммы: {title}")
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Периодограмма
        ax.semilogx(periods, powers, 'b-', linewidth=1.5, label='Power')
        
        # Выделение лучшего периода
        if best_period is not None:
            best_power = np.interp(best_period, periods, powers)
            ax.semilogx(best_period, best_power, 'ro', markersize=10, 
                       label=f'Best Period: {best_period:.3f} days')
        
        ax.set_xlabel('Period (days)')
        ax.set_ylabel('Power')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Сохранение
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"График сохранен: {save_path}")
        
        plt.close()
    
    def plot_candidates_distribution(self, candidates: List[Dict],
                                   title: str = "Candidates Distribution",
                                   save_path: Optional[str] = None) -> None:
        """
        Построение графика распределения кандидатов.
        
        Args:
            candidates: Список кандидатов.
            title: Заголовок графика.
            save_path: Путь для сохранения графика.
        """
        logger.info(f"Создание графика распределения кандидатов: {title}")
        
        if not candidates:
            logger.warning("Нет кандидатов для визуализации")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Подготовка данных
        df = pd.DataFrame(candidates)
        
        # 1. Распределение по методам
        if 'method' in df.columns:
            method_counts = df['method'].value_counts()
            axes[0, 0].pie(method_counts.values, labels=method_counts.index, 
                          autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Distribution by Method')
        
        # 2. Распределение периодов
        if 'period' in df.columns:
            periods = df['period'][df['period'] > 0]
            if len(periods) > 0:
                axes[0, 1].hist(periods, bins=20, alpha=0.7, edgecolor='black')
                axes[0, 1].set_xlabel('Period (days)')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].set_title('Period Distribution')
                axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Распределение уверенности
        if 'confidence' in df.columns:
            axes[1, 0].hist(df['confidence'], bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Confidence')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Confidence Distribution')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Корреляция период-уверенность
        if 'period' in df.columns and 'confidence' in df.columns:
            valid_data = df[(df['period'] > 0) & (df['confidence'] > 0)]
            if len(valid_data) > 0:
                scatter = axes[1, 1].scatter(valid_data['period'], valid_data['confidence'], 
                                          alpha=0.6, c=range(len(valid_data)), cmap='viridis')
                axes[1, 1].set_xlabel('Period (days)')
                axes[1, 1].set_ylabel('Confidence')
                axes[1, 1].set_title('Period vs Confidence')
                axes[1, 1].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[1, 1], label='Candidate Index')
        
        plt.tight_layout()
        
        # Сохранение
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"График сохранен: {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray,
                      y_scores: np.ndarray,
                      title: str = "ROC Curve",
                      save_path: Optional[str] = None) -> None:
        """
        Построение ROC-кривой.
        
        Args:
            y_true: Истинные метки.
            y_scores: Предсказанные оценки.
            title: Заголовок графика.
            save_path: Путь для сохранения графика.
        """
        logger.info(f"Создание ROC-кривой: {title}")
        
        from sklearn.metrics import roc_curve, auc
        
        # Вычисление ROC-кривой
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # ROC-кривая
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.2f})')
        
        # Диагональная линия (случайный классификатор)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # Сохранение
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"График сохранен: {save_path}")
        
        plt.close()
    
    def plot_training_history(self, history: Dict,
                            title: str = "Training History",
                            save_path: Optional[str] = None) -> None:
        """
        Построение графика истории обучения.
        
        Args:
            history: История обучения модели.
            title: Заголовок графика.
            save_path: Путь для сохранения графика.
        """
        logger.info(f"Создание графика истории обучения: {title}")
        
        # Определяем количество подграфиков
        metrics = list(history.keys())
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            logger.warning("Нет данных для визуализации истории обучения")
            return
        
        # Создаем подграфики
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle(title, fontsize=16)
        
        for i, metric in enumerate(metrics):
            epochs = range(1, len(history[metric]) + 1)
            axes[i].plot(epochs, history[metric], 'b-', linewidth=2)
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохранение
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"График сохранен: {save_path}")
        
        plt.close()
    
    def plot_anomaly_scores(self, scores: np.ndarray,
                          threshold: float = 0.5,
                          title: str = "Anomaly Scores",
                          save_path: Optional[str] = None) -> None:
        """
        Построение графика оценок аномальности.
        
        Args:
            scores: Оценки аномальности.
            threshold: Порог для выделения аномалий.
            title: Заголовок графика.
            save_path: Путь для сохранения графика.
        """
        logger.info(f"Создание графика оценок аномальности: {title}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)
        
        # Гистограмма оценок
        ax1.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {threshold}')
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Anomaly Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Временной ряд оценок
        ax2.plot(scores, 'b-', alpha=0.7, linewidth=1)
        ax2.axhline(threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {threshold}')
        
        # Выделение аномалий
        anomalies = scores > threshold
        ax2.scatter(np.where(anomalies)[0], scores[anomalies], 
                   color='red', s=20, alpha=0.8, label='Anomalies')
        
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Anomaly Score')
        ax2.set_title('Anomaly Scores Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохранение
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"График сохранен: {save_path}")
        
        plt.close()
    
    def create_summary_plot(self, results: Dict,
                          save_path: Optional[str] = None) -> None:
        """
        Создание сводного графика результатов.
        
        Args:
            results: Результаты поиска экзопланет.
            save_path: Путь для сохранения графика.
        """
        logger.info("Создание сводного графика результатов")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Exoplanet Search Results Summary', fontsize=16)
        
        candidates = results.get('candidates', [])
        
        if not candidates:
            # Если нет кандидатов, показываем пустые графики
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No candidates found', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('No Data')
        else:
            df = pd.DataFrame(candidates)
            
            # 1. Распределение по методам
            if 'method' in df.columns:
                method_counts = df['method'].value_counts()
                axes[0, 0].bar(method_counts.index, method_counts.values)
                axes[0, 0].set_title('Candidates by Method')
                axes[0, 0].set_ylabel('Count')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Распределение периодов
            if 'period' in df.columns:
                periods = df['period'][df['period'] > 0]
                if len(periods) > 0:
                    axes[0, 1].hist(periods, bins=20, alpha=0.7, edgecolor='black')
                    axes[0, 1].set_title('Period Distribution')
                    axes[0, 1].set_xlabel('Period (days)')
                    axes[0, 1].set_ylabel('Count')
            
            # 3. Распределение уверенности
            if 'confidence' in df.columns:
                axes[1, 0].hist(df['confidence'], bins=20, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Confidence Distribution')
                axes[1, 0].set_xlabel('Confidence')
                axes[1, 0].set_ylabel('Count')
            
            # 4. Корреляция период-уверенность
            if 'period' in df.columns and 'confidence' in df.columns:
                valid_data = df[(df['period'] > 0) & (df['confidence'] > 0)]
                if len(valid_data) > 0:
                    scatter = axes[1, 1].scatter(valid_data['period'], valid_data['confidence'], 
                                                alpha=0.6, c=range(len(valid_data)), cmap='viridis')
                    axes[1, 1].set_title('Period vs Confidence')
                    axes[1, 1].set_xlabel('Period (days)')
                    axes[1, 1].set_ylabel('Confidence')
                    plt.colorbar(scatter, ax=axes[1, 1], label='Candidate Index')
        
        plt.tight_layout()
        
        # Сохранение
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Сводный график сохранен: {save_path}")
        
        plt.close()


def visualize_results(results: Dict,
                     output_dir: str = "results/plots",
                     create_all: bool = True) -> Dict[str, str]:
    """
    Основная функция визуализации результатов.
    
    Args:
        results: Результаты поиска экзопланет.
        output_dir: Директория для сохранения графиков.
        create_all: Создавать ли все типы графиков.
        
    Returns:
        Dict[str, str]: Пути к созданным графикам.
    """
    logger.info("Начинаем визуализацию результатов")
    
    visualizer = ExoplanetVisualizer(output_dir)
    plot_files = {}
    
    candidates = results.get('candidates', [])
    
    if not candidates:
        logger.warning("Нет кандидатов для визуализации")
        return plot_files
    
    # Сводный график
    summary_path = visualizer.output_dir / "summary_plot.png"
    visualizer.create_summary_plot(results, str(summary_path))
    plot_files['summary'] = str(summary_path)
    
    if create_all:
        # Распределение кандидатов
        candidates_path = visualizer.output_dir / "candidates_distribution.png"
        visualizer.plot_candidates_distribution(candidates, 
                                              "Candidates Distribution",
                                              str(candidates_path))
        plot_files['candidates'] = str(candidates_path)
    
    logger.info(f"Создано {len(plot_files)} графиков")
    return plot_files
