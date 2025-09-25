"""
Utilities for AI Module

Вспомогательные функции для AI модуля анализа кривых блеска.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from pathlib import Path
import json
import pickle
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

def normalize_lightcurve(flux: np.ndarray, method: str = 'median') -> np.ndarray:
    """
    Нормализация кривой блеска
    
    Args:
        flux: Массив значений потока
        method: Метод нормализации ('median', 'mean', 'minmax')
        
    Returns:
        Нормализованный массив
    """
    flux = np.array(flux, dtype=np.float32)
    
    if method == 'median':
        return flux / np.median(flux)
    elif method == 'mean':
        return flux / np.mean(flux)
    elif method == 'minmax':
        min_val, max_val = np.min(flux), np.max(flux)
        return (flux - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def remove_outliers(data: np.ndarray, threshold: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Удаление выбросов методом IQR
    
    Args:
        data: Входные данные
        threshold: Множитель для IQR
        
    Returns:
        Tuple из (очищенные данные, маска валидных точек)
    """
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    mask = (data >= lower_bound) & (data <= upper_bound)
    cleaned_data = data.copy()
    cleaned_data[~mask] = np.median(data[mask])
    
    return cleaned_data, mask

def resample_lightcurve(time: np.ndarray, flux: np.ndarray, 
                       target_length: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ресэмплинг кривой блеска до целевой длины
    
    Args:
        time: Массив времени
        flux: Массив потока
        target_length: Целевая длина
        
    Returns:
        Tuple из (новое время, новый поток)
    """
    if len(flux) == target_length:
        return time, flux
    
    if len(flux) > target_length:
        # Даунсэмплинг
        indices = np.linspace(0, len(flux) - 1, target_length).astype(int)
        return time[indices], flux[indices]
    else:
        # Апсэмплинг с интерполяцией
        try:
            from scipy.interpolate import interp1d
            f_flux = interp1d(time, flux, kind='linear', fill_value='extrapolate')
            new_time = np.linspace(time[0], time[-1], target_length)
            new_flux = f_flux(new_time)
            return new_time, new_flux
        except ImportError:
            # Fallback без scipy
            new_time = np.linspace(time[0], time[-1], target_length)
            new_flux = np.interp(new_time, time, flux)
            return new_time, new_flux

def compute_data_hash(data: Union[np.ndarray, List, str]) -> str:
    """
    Вычисление хэша данных для кэширования
    
    Args:
        data: Данные для хэширования
        
    Returns:
        Хэш строка
    """
    if isinstance(data, np.ndarray):
        data_bytes = data.tobytes()
    elif isinstance(data, (list, tuple)):
        data_bytes = str(data).encode('utf-8')
    elif isinstance(data, str):
        data_bytes = data.encode('utf-8')
    else:
        data_bytes = str(data).encode('utf-8')
    
    return hashlib.md5(data_bytes).hexdigest()

def create_sliding_windows(data: np.ndarray, window_size: int, 
                          stride: int = 1) -> np.ndarray:
    """
    Создание скользящих окон из данных
    
    Args:
        data: Входные данные
        window_size: Размер окна
        stride: Шаг скольжения
        
    Returns:
        Массив окон [num_windows, window_size]
    """
    if len(data) < window_size:
        return np.array([data])
    
    num_windows = (len(data) - window_size) // stride + 1
    windows = np.zeros((num_windows, window_size))
    
    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        windows[i] = data[start_idx:end_idx]
    
    return windows

def calculate_snr(signal: np.ndarray, noise_region: Optional[np.ndarray] = None) -> float:
    """
    Вычисление отношения сигнал/шум
    
    Args:
        signal: Сигнал
        noise_region: Область шума (если None, используется весь сигнал)
        
    Returns:
        SNR значение
    """
    if noise_region is None:
        noise_region = signal
    
    signal_power = np.mean(signal**2)
    noise_power = np.var(noise_region)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)

def fold_lightcurve(time: np.ndarray, flux: np.ndarray, 
                   period: float, t0: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Фолдинг кривой блеска по периоду
    
    Args:
        time: Массив времени
        flux: Массив потока
        period: Период фолдинга
        t0: Начальная эпоха
        
    Returns:
        Tuple из (фаза, поток)
    """
    phase = ((time - t0) % period) / period
    
    # Сортируем по фазе
    sort_indices = np.argsort(phase)
    
    return phase[sort_indices], flux[sort_indices]

def bin_lightcurve(time: np.ndarray, flux: np.ndarray, 
                  bin_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Биннинг кривой блеска
    
    Args:
        time: Массив времени
        flux: Массив потока
        bin_size: Размер бина во времени
        
    Returns:
        Tuple из (время бинов, поток бинов, ошибки бинов)
    """
    time_min, time_max = np.min(time), np.max(time)
    num_bins = int((time_max - time_min) / bin_size) + 1
    
    bin_centers = np.linspace(time_min, time_max, num_bins)
    bin_flux = np.zeros(num_bins)
    bin_errors = np.zeros(num_bins)
    
    for i in range(num_bins):
        bin_start = bin_centers[i] - bin_size / 2
        bin_end = bin_centers[i] + bin_size / 2
        
        mask = (time >= bin_start) & (time < bin_end)
        
        if np.sum(mask) > 0:
            bin_flux[i] = np.mean(flux[mask])
            bin_errors[i] = np.std(flux[mask]) / np.sqrt(np.sum(mask))
        else:
            bin_flux[i] = np.nan
            bin_errors[i] = np.nan
    
    # Удаляем пустые бины
    valid_mask = ~np.isnan(bin_flux)
    
    return bin_centers[valid_mask], bin_flux[valid_mask], bin_errors[valid_mask]

def detect_gaps(time: np.ndarray, threshold: float = None) -> List[Tuple[int, int]]:
    """
    Обнаружение пропусков в данных
    
    Args:
        time: Массив времени
        threshold: Порог для определения пропуска (если None, автоматически)
        
    Returns:
        Список кортежей (начало_пропуска, конец_пропуска)
    """
    if threshold is None:
        # Автоматическое определение порога
        time_diffs = np.diff(time)
        threshold = np.median(time_diffs) * 5
    
    time_diffs = np.diff(time)
    gap_indices = np.where(time_diffs > threshold)[0]
    
    gaps = []
    for idx in gap_indices:
        gaps.append((idx, idx + 1))
    
    return gaps

def estimate_noise_level(flux: np.ndarray, method: str = 'mad') -> float:
    """
    Оценка уровня шума в кривой блеска
    
    Args:
        flux: Массив потока
        method: Метод оценки ('std', 'mad', 'iqr')
        
    Returns:
        Уровень шума
    """
    if method == 'std':
        return np.std(flux)
    elif method == 'mad':
        # Median Absolute Deviation
        median = np.median(flux)
        mad = np.median(np.abs(flux - median))
        return 1.4826 * mad  # Нормализация для гауссовского распределения
    elif method == 'iqr':
        q1, q3 = np.percentile(flux, [25, 75])
        return (q3 - q1) / 1.349  # Нормализация для гауссовского распределения
    else:
        raise ValueError(f"Unknown noise estimation method: {method}")

def save_model_metadata(model_path: Path, metadata: Dict[str, Any]):
    """
    Сохранение метаданных модели
    
    Args:
        model_path: Путь к модели
        metadata: Метаданные для сохранения
    """
    metadata_path = model_path.with_suffix('.json')
    
    # Добавляем временную метку
    metadata['saved_at'] = datetime.now().isoformat()
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_model_metadata(model_path: Path) -> Optional[Dict[str, Any]]:
    """
    Загрузка метаданных модели
    
    Args:
        model_path: Путь к модели
        
    Returns:
        Метаданные или None если файл не найден
    """
    metadata_path = model_path.with_suffix('.json')
    
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_path}: {e}")
        return None

def convert_to_tensor(data: Union[np.ndarray, List], device: str = 'cpu') -> torch.Tensor:
    """
    Конвертация данных в PyTorch тензор
    
    Args:
        data: Данные для конвертации
        device: Устройство для размещения тензора
        
    Returns:
        PyTorch тензор
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    
    tensor = torch.FloatTensor(data)
    return tensor.to(device)

def ensure_batch_dimension(tensor: torch.Tensor) -> torch.Tensor:
    """
    Добавление batch размерности если её нет
    
    Args:
        tensor: Входной тензор
        
    Returns:
        Тензор с batch размерностью
    """
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    elif tensor.dim() == 2 and tensor.shape[0] == 1:
        return tensor
    elif tensor.dim() == 2:
        return tensor.unsqueeze(0)
    else:
        return tensor

def create_attention_mask(sequence_length: int, 
                         valid_length: int) -> torch.Tensor:
    """
    Создание маски внимания для последовательностей с padding
    
    Args:
        sequence_length: Полная длина последовательности
        valid_length: Валидная длина (без padding)
        
    Returns:
        Маска внимания
    """
    mask = torch.zeros(sequence_length, dtype=torch.bool)
    if valid_length < sequence_length:
        mask[valid_length:] = True
    
    return mask

def log_model_performance(model_name: str, metrics: Dict[str, float], 
                         log_file: Optional[Path] = None):
    """
    Логирование производительности модели
    
    Args:
        model_name: Имя модели
        metrics: Метрики производительности
        log_file: Файл для логирования (если None, используется стандартный логгер)
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'metrics': metrics
    }
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Читаем существующие логи
        logs = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        # Добавляем новый лог
        logs.append(log_entry)
        
        # Сохраняем
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    else:
        logger.info(f"Model {model_name} performance: {metrics}")

def validate_lightcurve_data(time: np.ndarray, flux: np.ndarray) -> Dict[str, Any]:
    """
    Валидация данных кривой блеска
    
    Args:
        time: Массив времени
        flux: Массив потока
        
    Returns:
        Словарь с результатами валидации
    """
    validation_result = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'statistics': {}
    }
    
    # Проверка базовых требований
    if len(time) != len(flux):
        validation_result['is_valid'] = False
        validation_result['errors'].append("Time and flux arrays have different lengths")
    
    if len(time) < 10:
        validation_result['is_valid'] = False
        validation_result['errors'].append("Insufficient data points (< 10)")
    
    # Проверка на NaN и inf
    if np.any(~np.isfinite(time)):
        validation_result['warnings'].append("Non-finite values in time array")
    
    if np.any(~np.isfinite(flux)):
        validation_result['warnings'].append("Non-finite values in flux array")
    
    # Проверка монотонности времени
    if not np.all(np.diff(time) > 0):
        validation_result['warnings'].append("Time array is not monotonically increasing")
    
    # Статистика
    if len(flux) > 0:
        validation_result['statistics'] = {
            'duration_days': float(np.max(time) - np.min(time)),
            'median_cadence_days': float(np.median(np.diff(time))),
            'data_points': len(flux),
            'flux_median': float(np.median(flux)),
            'flux_std': float(np.std(flux)),
            'noise_estimate': estimate_noise_level(flux)
        }
    
    return validation_result
