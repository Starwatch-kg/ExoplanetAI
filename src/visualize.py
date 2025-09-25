import os
import matplotlib
matplotlib.use('Agg')  # Используем неинтерактивный бэкенд для серверной среды
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

# --- Setup ---
# Устанавливаем стиль для графиков
sns.set_theme(style="whitegrid", palette="muted")
# Директория для сохранения результатов
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_history(history: dict, metric: str, save_path: str, title: str = None, 
                 xlabel: str = "Epochs", ylabel: str = None):
    """
    Строит и сохраняет график метрики для тренировочной и валидационной выборок.

    Args:
        history (dict): Словарь с историей обучения, должен содержать ключи 
                        'train_{metric}' и 'val_{metric}'.
        metric (str): Название метрики (например, 'loss' или 'accuracy').
        save_path (str): Путь для сохранения графика.
        title (str, optional): Заголовок графика.
        xlabel (str, optional): Подпись оси X.
        ylabel (str, optional): Подпись оси Y.
    """
    plt.figure(figsize=(12, 7))
    
    train_metric_key = f'train_{metric}'
    val_metric_key = f'val_{metric}'

    if train_metric_key not in history or val_metric_key not in history:
        raise ValueError(f"History dictionary must contain '{train_metric_key}' and '{val_metric_key}' keys.")

    plt.plot(history[train_metric_key], label=f'Training {metric.capitalize()}', color='royalblue', marker='o', linestyle='-')
    plt.plot(history[val_metric_key], label=f'Validation {metric.capitalize()}', color='darkorange', marker='x', linestyle='--')
    
    plt.title(title or f'Training and Validation {metric.capitalize()}', fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel or metric.capitalize(), fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    # Всегда сохраняем в файл, поскольку используем неинтерактивный бэкенд
    if save_path:
        full_save_path = os.path.join(RESULTS_DIR, save_path)
    else:
        # Генерируем уникальное имя файла
        import time
        timestamp = int(time.time())
        full_save_path = os.path.join(RESULTS_DIR, f'history_{metric}_{timestamp}.png')

    plt.savefig(full_save_path)
    print(f"Saved {metric} plot to {full_save_path}")
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], 
                          save_path: str, normalize: bool = False, title: str = 'Confusion Matrix'):
    """
    Строит и сохраняет матрицу ошибок.

    Args:
        y_true (np.ndarray): Истинные метки.
        y_pred (np.ndarray): Предсказанные метки.
        classes (list[str]): Список названий классов.
        save_path (str): Путь для сохранения графика.
        normalize (bool, optional): Нормализовать ли матрицу. Defaults to False.
        title (str, optional): Заголовок графика.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = f'Normalized {title}'
    else:
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=classes, yticklabels=classes)
    
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Всегда сохраняем в файл, поскольку используем неинтерактивный бэкенд
    if save_path:
        full_save_path = os.path.join(RESULTS_DIR, save_path)
    else:
        # Генерируем уникальное имя файла
        import time
        timestamp = int(time.time())
        full_save_path = os.path.join(RESULTS_DIR, f'confusion_matrix_{timestamp}.png')

    plt.savefig(full_save_path)
    print(f"Saved confusion matrix to {full_save_path}")
    plt.close()

def plot_regression(y_true: np.ndarray, y_pred: np.ndarray, save_path: str, 
                    title: str = 'Predicted vs. Actual Values', 
                    xlabel: str = 'Actual Values', ylabel: str = 'Predicted Values'):
    """
    Строит scatter plot для сравнения предсказанных и реальных значений в задаче регрессии.

    Args:
        y_true (np.ndarray): Реальные значения.
        y_pred (np.ndarray): Предсказанные значения.
        save_path (str): Путь для сохранения графика.
        title (str, optional): Заголовок графика.
        xlabel (str, optional): Подпись оси X.
        ylabel (str, optional): Подпись оси Y.
    """
    plt.figure(figsize=(10, 10))
    
    # Scatter plot
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor='k', s=80)
    
    # Идеальная линия регрессии (y=x)
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label='Ideal Line (y=x)')
    
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.axis('equal') # Делаем оси одинаковыми для правильного восприятия
    plt.grid(True)
    plt.tight_layout()
    
    # Всегда сохраняем в файл, поскольку используем неинтерактивный бэкенд
    if save_path:
        full_save_path = os.path.join(RESULTS_DIR, save_path)
    else:
        # Генерируем уникальное имя файла
        import time
        timestamp = int(time.time())
        full_save_path = os.path.join(RESULTS_DIR, f'regression_{timestamp}.png')

    plt.savefig(full_save_path)
    print(f"Saved regression plot to {full_save_path}")
    plt.close()

def plot_lightcurve(time, flux, probs=None, candidates=None, save_path=None):
    """
    Строит и сохраняет кривую блеска с кандидатами в транзиты.

    Args:
        time (np.ndarray): Временные метки.
        flux (np.ndarray): Значения потока.
        probs (np.ndarray, optional): Вероятности транзита.
        candidates (list, optional): Список словарей с кандидатами.
        save_path (str, optional): Путь для сохранения графика. Если None, используется имя по умолчанию.
    """
    plt.figure(figsize=(15, 5))
    plt.plot(time, flux, label='Flux', linewidth=1, color='royalblue')

    if probs is not None:
        # Масштабируем вероятности для наглядности
        p_scaled = np.interp(probs, (probs.min(), probs.max()), (np.percentile(flux, 5), np.percentile(flux, 95)))
        plt.plot(time[:len(p_scaled)], p_scaled, color='darkorange', alpha=0.7, label='Transit Probability')

    if candidates:
        for cand in candidates:
            plt.axvspan(cand['start_time'], cand['end_time'], color='red', alpha=0.3)
            plt.text(
                (cand['start_time'] + cand['end_time']) / 2,
                np.max(flux),
                f"{cand['mean_prob']:.2f}",
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=9
            )

    plt.title('Light Curve with Transit Candidates', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Normalized Flux', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Всегда сохраняем в файл, поскольку используем неинтерактивный бэкенд
    if save_path:
        full_save_path = os.path.join(RESULTS_DIR, save_path)
    else:
        # Генерируем уникальное имя файла
        import time
        timestamp = int(time.time())
        full_save_path = os.path.join(RESULTS_DIR, f'lightcurve_{timestamp}.png')

    plt.savefig(full_save_path)
    print(f"Saved light curve plot to {full_save_path}")
    plt.close()

    return full_save_path


# Пример использования (можно раскомментировать для теста)
if __name__ == '__main__':
    print("Running visualization script examples...")

    # --- Пример для plot_history ---
    mock_history = {
        'train_loss': np.exp(-np.arange(10) * 0.3) + np.random.rand(10) * 0.05,
        'val_loss': np.exp(-np.arange(10) * 0.25) + np.random.rand(10) * 0.1,
        'train_accuracy': 0.6 + (1 - np.exp(-np.arange(10) * 0.3)),
        'val_accuracy': 0.55 + (1 - np.exp(-np.arange(10) * 0.25)),
    }
    plot_history(mock_history, 'loss', 'sample_loss_plot.png')
    plot_history(mock_history, 'accuracy', 'sample_accuracy_plot.png')

    # --- Пример для plot_confusion_matrix ---
    mock_y_true = np.random.randint(0, 2, 100)
    mock_y_pred = np.random.randint(0, 2, 100)
    class_names = ['No Exoplanet', 'Exoplanet']
    plot_confusion_matrix(mock_y_true, mock_y_pred, class_names, 'sample_cm_plot.png')
    plot_confusion_matrix(mock_y_true, mock_y_pred, class_names, 'sample_cm_normalized_plot.png', normalize=True)

    # --- Пример для plot_regression ---
    mock_reg_true = np.linspace(0, 100, 100)
    mock_reg_pred = mock_reg_true + np.random.normal(0, 10, 100)
    plot_regression(mock_reg_true, mock_reg_pred, 'sample_regression_plot.png')

    # --- Пример для plot_lightcurve ---
    mock_time = np.linspace(0, 1, 100)
    mock_flux = np.sin(2 * np.pi * mock_time) + np.random.normal(0, 0.1, 100)
    mock_probs = np.random.rand(100)
    mock_candidates = [{'start_time': 0.2, 'end_time': 0.3, 'mean_prob': 0.85}]
    plot_lightcurve(mock_time, mock_flux, mock_probs, mock_candidates, 'sample_lightcurve_plot.png')
    
    print("All example plots have been generated in the 'results/' directory.")
