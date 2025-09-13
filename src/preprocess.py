import numpy as np
import torch
from scipy import signal
from lightkurve import search_lightcurvefile

# Placeholder imports, replace with your actual modules
# Assuming detect.py and visualize.py are in the same directory (src)
try:
    from .. import detect
    from .. import visualize
except ImportError:
    print("Warning: Could not import 'detect' and 'visualize' modules. Using placeholder objects.")
    class Placeholder:
        def __getattr__(self, name):
            def _dummy(*args, **kwargs):
                print(f"Placeholder function {name} called with args: {args}, kwargs: {kwargs}")
                if name == 'train_on_windows':
                    # Return a dummy model and history
                    return torch.nn.Sequential(torch.nn.Linear(1,1)), {}
                if name == 'sliding_prediction_full':
                    # Return dummy probabilities
                    return np.random.rand(len(args[1]))
                if name == 'extract_candidates':
                    return []
                return None
            return _dummy
    detect = Placeholder()
    visualize = Placeholder()

from .preprocessing_utils import (
    normalize_array, remove_outliers, smooth_lightcurve, detrend
)
from .cache import LightCurveCache
from .data_loading import (
    load_lightcurve, load_kepler_realtime, DataSource, DataQuality, DataValidationError,
    validate_lightcurve, check_data_quality
)
from .synthetic import generate_synthetic_transit, generate_training_dataset


torch.backends.quantized.engine = 'fbgemm'  # или 'qnnpack' для CPU

def main():
    """Main execution block for demonstration and testing."""
    # Генерация синтетических данных для обучения
    print("Generating synthetic training data...")
    X_train, y_train = generate_training_dataset()

    # Обучение модели (CNN или LSTM на выбор)
    print("Training model...")
    # Assuming detect.py has train_on_windows
    model, history = detect.train_on_windows(X_train, y_train, model_type='cnn')

    # Визуализация процесса обучения
    print("Plotting training history...")
    visualize.plot_training_history(history)

    # Сохранение модели
    print("Saving model...")
    detect.save_model(model, 'models/exoplanet_detector.pth')

    # Загрузка реальных данных
    print("Loading real data for Kepler-10...")
    try:
        times, flux = load_kepler_realtime("Kepler-10")
    except Exception as e:
        print(f"Error loading real data: {e}")
        return

    # Предобработка
    print("Preprocessing real data...")
    flux = remove_outliers(flux)
    flux = smooth_lightcurve(flux)
    flux = normalize_array(flux)

    # Предсказание
    print("Making predictions...")
    probs = detect.sliding_prediction_full(model, flux)
    candidates = detect.extract_candidates(times, probs)

    # Визуализация результатов
    print("Visualizing results...")
    visualize.plot_lightcurve(times, flux, probs, candidates)

    # Детальный просмотр кандидата
    if candidates:
        print("Plotting candidate details...")
        visualize.plot_candidate_details(times, flux, candidates[0])
    else:
        print("No candidates found.")

if __name__ == '__main__':
    main()
    np.ndarray
        
    """
    cleaned_data = np.copy(data)
    
    if method == 'sigma':
        mean = np.mean(data)
        std = np.std(data)
        mask = np.abs(data - mean) <= threshold * std
    
    elif method == 'mad':
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        mask = np.abs(data - median) <= threshold * 1.4826 * mad
    
    elif method == 'local':
        if window_size is None:
            window_size = len(data) // 20  # 5% от длины данных
        
        mask = np.ones_like(data, dtype=bool)
        for i in range(len(data)):
            start = max(0, i - window_size//2)
            end = min(len(data), i + window_size//2)
            window = data[start:end]
            local_med = np.median(window)
            local_std = np.std(window)
            if np.abs(data[i] - local_med) > threshold * local_std:
                mask[i] = False
    
    elif method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        mask = (data >= q1 - threshold * iqr) & (data <= q3 + threshold * iqr)
    
    else:
        raise ValueError(f"Unknown outlier removal method: {method}")
    
    # Заполняем выбросы интерполированными значениями
    cleaned_data[~mask] = np.nan
    nans, x = np.isnan(cleaned_data), lambda z: z.nonzero()[0]
    cleaned_data[nans] = np.interp(x(nans), x(~nans), cleaned_data[~nans])
    
    return cleaned_data

def smooth_lightcurve(
    data: np.ndarray,
    method: str = 'savgol',
    window_length: int = 31,
    polyorder: int = 3
) -> np.ndarray:
    """
    #Сглаживает кривую блеска различными методами
    
    Parameters
    ----------
    data : np.ndarray
        Входной массив данных
    method : str
        Метод сглаживания:
        - 'savgol': фильтр Савицкого-Голея
        - 'gaussian': гауссово сглаживание
        - 'median': медианный фильтр
        - 'lowess': LOWESS сглаживание
    window_length : int
        Длина окна (должна быть нечетной)
    polyorder : int
        Порядок полинома (для метода Савицкого-Голея)
        
    Returns
    -------
    np.ndarray
        #Сглаженный массив
    """
    if method == 'savgol':
        return signal.savgol_filter(data, window_length, polyorder)
    
    elif method == 'gaussian':
        window = signal.gaussian(window_length, std=window_length/5.0)
        window = window / window.sum()
        return signal.convolve(data, window, mode='same')
    
    elif method == 'median':
        return signal.medfilt(data, kernel_size=window_length)
    
    elif method == 'lowess':
        from statsmodels.nonparametric.smoothers_lowess import lowess
        x = np.arange(len(data))
        frac = window_length / len(data)
        smooth_data = lowess(data, x, frac=frac, return_sorted=False)
        return smooth_data
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

def detrend(
    data: np.ndarray,
    time: np.ndarray,
    method: str = 'polynomial',
    polyorder: int = 2
) -> np.ndarray:
    """
    Удаляет тренд из кривой блеска
    
    Parameters
    ----------
    data : np.ndarray
        Входной массив данных
    time : np.ndarray
        Временные точки
    method : str
        Метод удаления тренда:
        - 'polynomial': полиномиальная подгонка
        - 'spline': сплайн
        - 'median': вычитание скользящей медианы
    polyorder : int
        Порядок полинома (для polynomial метода)
        
    Returns
    -------
    np.ndarray
        Массив без тренда
    """
    if method == 'polynomial':
        coeffs = np.polyfit(time, data, polyorder)
        trend = np.polyval(coeffs, time)
        return data / trend - 1
    
    elif method == 'spline':
        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(time, data, k=3)
        return data / spline(time) - 1
    
    elif method == 'median':
        window_length = len(data) // 10  # 10% от длины данных
        if window_length % 2 == 0:
            window_length += 1
        trend = signal.medfilt(data, kernel_size=window_length)
        return data / trend - 1
    
    else:
        raise ValueError(f"Unknown detrending method: {method}")

import os
import logging
from typing import Tuple, Optional, List, Dict
import numpy as np
import json
import hashlib
import pickle
from pathlib import Path
from lightkurve import search_lightcurvefile, search_targetpixelfile
from astropy.time import Time

class LightCurveCache:
    """Класс для кэширования кривых блеска"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Инициализация системы кэширования
        
        Parameters
        ----------
        cache_dir : str
            Путь к директории для кэширования
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Загружает метаданные кэша"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Сохраняет метаданные кэша"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
    
    def _get_cache_key(self, params: Dict) -> str:
        """Генерирует ключ кэша на основе параметров"""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def get(self, params: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Получает данные из кэша
        
        Parameters
        ----------
        params : Dict
            Параметры запроса
            
        Returns
        -------
        Optional[Tuple[np.ndarray, np.ndarray]]
            Кортеж (time, flux) или None, если кэш не найден
        """
        cache_key = self._get_cache_key(params)
        if cache_key not in self.metadata:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"Error loading cache: {str(e)}")
            return None
    
    def put(self, params: Dict, data: Tuple[np.ndarray, np.ndarray]):
        """
        Сохраняет данные в кэш
        
        Parameters
        ----------
        params : Dict
            Параметры запроса
        data : Tuple[np.ndarray, np.ndarray]
            Данные для сохранения (time, flux)
        """
        cache_key = self._get_cache_key(params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.metadata[cache_key] = {
                'params': params,
                'created': Time.now().iso,
                'shape': (data[0].shape, data[1].shape)
            }
            self._save_metadata()
        except Exception as e:
            logging.error(f"Error saving to cache: {str(e)}")
    
    def clear(self):
        """Очищает кэш"""
        for file in self.cache_dir.glob("*.pkl"):
            file.unlink()
        self.metadata = {}
        self._save_metadata()

# Создаем глобальный экземпляр кэша
_cache = LightCurveCache()

class DataSource:
    """Источник данных для загрузки кривых блеска"""
    KEPLER = "Kepler"
    K2 = "K2"
    TESS = "TESS"

class DataQuality:
    """Флаги качества данных"""
    GOOD = 0
    DEATTACHED = 1
    SAFE_MODE = 2
    EARTH_POINT = 4
    REACTION_WHEEL = 8
    COARSE_POINT = 16
    ECLIPSE = 32

class DataValidationError(Exception):
    """Исключение для ошибок валидации данных"""
    pass

def validate_lightcurve(times: np.ndarray, flux: np.ndarray) -> dict:
    """
    Проверяет качество и целостность данных кривой блеска
    
    Parameters
    ----------
    times : np.ndarray
        Временные точки
    flux : np.ndarray
        Значения потока
        
    Returns
    -------
    dict
        Словарь с метриками качества данных
        
    Raises
    ------
    DataValidationError
        Если данные не проходят критические проверки
    """
    validation = {}
    
    # Проверка базовой целостности
    if len(times) != len(flux):
        raise DataValidationError("Размерности времени и потока не совпадают")
    
    if len(times) == 0:
        raise DataValidationError("Пустой набор данных")
    
    # Проверка типов данных и наличия nan/inf
    validation['has_nans'] = np.any(np.isnan(flux))
    validation['has_infs'] = np.any(np.isinf(flux))
    
    if validation['has_nans'] or validation['has_infs']:
        raise DataValidationError("Данные содержат недопустимые значения (nan/inf)")
    
    # Проверка монотонности времени
    time_diffs = np.diff(times)
    validation['is_monotonic'] = np.all(time_diffs > 0)
    
    if not validation['is_monotonic']:
        raise DataValidationError("Временные точки не монотонны")
    
    # Анализ пропусков в данных
    median_cadence = np.median(time_diffs)
    large_gaps = time_diffs > (3 * median_cadence)
    validation['gap_indices'] = np.where(large_gaps)[0]
    validation['gap_sizes'] = time_diffs[large_gaps]
    validation['n_gaps'] = len(validation['gap_indices'])
    
    # Анализ выбросов
    flux_median = np.median(flux)
    flux_mad = np.median(np.abs(flux - flux_median))
    outliers = np.abs(flux - flux_median) > (5 * 1.4826 * flux_mad)
    validation['outlier_indices'] = np.where(outliers)[0]
    validation['n_outliers'] = np.sum(outliers)
    
    # Расчет статистик
    validation['time_span'] = times[-1] - times[0]
    validation['median_cadence'] = median_cadence
    validation['flux_mean'] = np.mean(flux)
    validation['flux_std'] = np.std(flux)
    validation['flux_min'] = np.min(flux)
    validation['flux_max'] = np.max(flux)
    
    # Проверка достаточности данных
    if len(times) < 100:
        raise DataValidationError("Недостаточно точек данных (минимум 100)")
    
    return validation

def check_data_quality(validation_results: dict, strict: bool = False) -> bool:
    """
    Проверяет результаты валидации на соответствие критериям качества
    
    Parameters
    ----------
    validation_results : dict
        Результаты валидации от функции validate_lightcurve
    strict : bool
        Использовать ли строгие критерии качества
        
    Returns
    -------
    bool
        True если данные удовлетворяют критериям качества
    """
    # Базовые критерии
    if validation_results['has_nans'] or validation_results['has_infs']:
        return False
    
    if not validation_results['is_monotonic']:
        return False
    
    # Проверка пропусков
    max_gaps = 3 if strict else 10
    if validation_results['n_gaps'] > max_gaps:
        return False
    
    # Проверка выбросов
    max_outlier_fraction = 0.01 if strict else 0.05
    outlier_fraction = validation_results['n_outliers'] / len(validation_results['outlier_indices'])
    if outlier_fraction > max_outlier_fraction:
        return False
    
    # Проверка длины временного ряда
    min_time_span = 10 if strict else 5  # дней
    if validation_results['time_span'] < min_time_span:
        return False
    
    # Проверка частоты измерений
    max_cadence = 0.02 if strict else 0.05  # дней
    if validation_results['median_cadence'] > max_cadence:
        return False
    
    return True
    
def load_lightcurve(
    target_name: str,
    mission: str = DataSource.KEPLER,
    cadence: str = 'long',
    quarters: Optional[List[int]] = None,
    quality_bitmask: str = 'default',
    flux_column: str = 'pdcsap_flux',
    use_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает кривую блеска из различных источников данных
    
    Parameters
    ----------
    target_name : str
        Имя цели (например, 'Kepler-10' или 'TIC 25155310')
    mission : str
        Миссия ('Kepler', 'K2', или 'TESS')
    cadence : str
        Тип каденции ('long' или 'short')
    quarters : List[int], optional
        Список конкретных кварталов для загрузки (только для Kepler/K2)
    quality_bitmask : str
        Маска качества данных ('default', 'none', 'hard', или custom)
    flux_column : str
        Тип потока ('sap_flux' или 'pdcsap_flux')
        
    Returns
    -------
    times : np.ndarray
        Временные точки (в днях)
    flux : np.ndarray
        Значения потока
        
    Raises
    ------
    ValueError
        Если данные не найдены или параметры некорректны
    """
    # Формируем параметры для кэша
    cache_params = {
        "target_name": target_name,
        "mission": mission,
        "cadence": cadence,
        "quarters": quarters,
        "quality_bitmask": quality_bitmask,
        "flux_column": flux_column
    }
    
    # Проверяем кэш
    if use_cache:
        cached_data = _cache.get(cache_params)
        if cached_data is not None:
            logging.info(f"Using cached data for {target_name}")
            return cached_data
    
    logging.info(f"Loading data for {target_name} from {mission}")
    
    # Проверка параметров
    if mission not in [DataSource.KEPLER, DataSource.K2, DataSource.TESS]:
        raise ValueError(f"Unsupported mission: {mission}")
    
    if cadence not in ['long', 'short']:
        raise ValueError(f"Invalid cadence: {cadence}")
    
    # Поиск данных
    search_params = {
        "target_name": target_name,
        "mission": mission,
        "cadence": cadence,
    }
    
    if quarters is not None and mission in [DataSource.KEPLER, DataSource.K2]:
        search_params["quarter"] = quarters
    
    try:
        # Пробуем сначала найти файл кривой блеска
        lc_collection = search_lightcurvefile(**search_params).download_all()
        
        if not lc_collection:
            # Если файл кривой блеска не найден, пробуем файл пиксельных данных
            logging.info("Light curve file not found, trying target pixel file")
            tpf = search_targetpixelfile(**search_params).download_all()
            if not tpf:
                raise ValueError(f"No data found for {target_name} in {mission}")
            lc_collection = [tp.to_lightcurve() for tp in tpf]
            
    except Exception as e:
        raise ValueError(f"Error downloading data: {str(e)}")
    
    # Объединяем все секвенции в одну кривую
    lc = getattr(lc_collection[0], flux_column)
    for lcf in lc_collection[1:]:
        lc = lc.append(getattr(lcf, flux_column))
    
    # Применяем маску качества
    lc = lc.remove_nans().remove_outliers()
    if quality_bitmask != 'none':
        lc = lc.normalize().flatten()
    
    # Конвертируем время в дни от начала наблюдений
    t0 = Time(lc.time.min().value, format='jd')
    times = (lc.time.value - t0.jd)
    flux = lc.flux.value
    
    # Сохраняем в кэш
    if use_cache:
        _cache.put(cache_params, (times, flux))
    
    return times, flux

def load_kepler_realtime(target_name: str, mission='Kepler', cadence='long'):
    """
    Загружает кривую блеска с NASA/TESS для указанного объекта.
    
    target_name: имя звезды или TIC/Kepler ID, например 'Kepler-10' или 'TIC 25155310'
    mission: 'Kepler', 'K2' или 'TESS'
    cadence: 'long' или 'short'
    """
    lc_collection = search_lightcurvefile(target_name, mission=mission).download_all()
    if not lc_collection:
        raise ValueError(f"Не удалось найти данные для {target_name} в {mission}")

    # Объединяем все секвенции в одну кривую
    lc = lc_collection[0].PDCSAP_FLUX  # используем PDCSAP_FLUX
    for lcf in lc_collection[1:]:
        lc = lc.append(lcf.PDCSAP_FLUX)
    
    # Возвращаем numpy массив времени и нормализованный поток
    t = lc.time.value
    flux = lc.flux.value
    return t, flux

def generate_synthetic_transit(
    time_points: np.ndarray,
    period: float = 10,
    duration: float = 0.5,
    depth: float = 0.01,
    noise_level: float = 0.001,
    transit_shape: str = 'quadratic',
    add_stellar_var: bool = True,
    stellar_var_amplitude: float = 0.002
) -> np.ndarray:
    """
    Генерирует синтетическую кривую блеска с транзитами
    
    Parameters
    ----------
    time_points : np.ndarray
        Временные точки для генерации данных
    period : float
        Период обращения планеты в днях
    duration : float
        Длительность транзита в днях
    depth : float
        Глубина транзита (относительное уменьшение потока)
    noise_level : float
        Уровень шума
    transit_shape : str
        Форма транзита ('box', 'quadratic', или 'limb_darkened')
    add_stellar_var : bool
        Добавлять ли звездную переменность
    stellar_var_amplitude : float
        Амплитуда звездной переменности
    
    Returns
    -------
    np.ndarray
        Сгенерированная кривая блеска
    """
    flux = np.ones_like(time_points)
    
    # Добавляем транзиты с реалистичной формой
    for t_mid in np.arange(min(time_points), max(time_points), period):
        # Создаем более реалистичную U-образную форму транзита
        for i, t in enumerate(time_points):
            dt = abs(t - t_mid)
            if dt < duration/2:
                # Используем квадратичную форму для более плавного перехода
                transit_shape = 1 - depth * (1 - (2*dt/duration)**2)
                flux[i] *= transit_shape
    
    # Добавляем звездную вариабельность (долговременный тренд)
    time_span = max(time_points) - min(time_points)
    trend = 0.002 * np.sin(2*np.pi*time_points/time_span)
    flux += trend
    
    # Добавляем красный шум (коррелированный)
    red_noise = np.random.normal(0, noise_level, size=len(flux))
    red_noise = np.convolve(red_noise, np.ones(10)/10, mode='same')
    
    # Добавляем белый шум
    white_noise = np.random.normal(0, noise_level/2, size=len(flux))
    
    flux += red_noise + white_noise
    
    return flux

def generate_training_dataset(num_samples=1000, sequence_length=2000, transit_probability=0.5):
    """
    Генерирует набор данных для обучения
    
    num_samples: количество последовательностей
    sequence_length: длина каждой последовательности
    transit_probability: вероятность наличия транзита в последовательности
    """
    X = []
    y = []
    
    time = np.linspace(0, 20, sequence_length)
    
    for _ in range(num_samples):
        if np.random.random() < transit_probability:
            # Генерируем последовательность с транзитом
            period = np.random.uniform(8, 12)
            duration = np.random.uniform(0.3, 0.7)
            depth = np.random.uniform(0.005, 0.015)
            flux = generate_synthetic_transit(time, period, duration, depth)
            y.append(1)
        else:
            # Генерируем последовательность без транзита
            flux = np.ones_like(time)
            flux += np.random.normal(0, 0.001, size=len(flux))
            y.append(0)
            
        X.append(flux)
    
    return np.array(X), np.array(y)

def main():
    """Main execution block for demonstration and testing."""
    # Генерация синтетических данных для обучения
    print("Generating synthetic training data...")
    X_train, y_train = generate_training_dataset()

    # Обучение модели (CNN или LSTM на выбор)
    print("Training model...")
    # Assuming detect.py has train_on_windows
    model, history = detect.train_on_windows(X_train, y_train, model_type='cnn')

    # Визуализация процесса обучения
    print("Plotting training history...")
    visualize.plot_training_history(history)

    # Сохранение модели
    print("Saving model...")
    detect.save_model(model, 'models/exoplanet_detector.pth')

    # Загрузка реальных данных
    print("Loading real data for Kepler-10...")
    times, flux = load_kepler_realtime("Kepler-10")

    # Предобработка
    print("Preprocessing real data...")
    flux = remove_outliers(flux)
    flux = smooth_lightcurve(flux)
    flux = normalize_array(flux)

    # Предсказание
    print("Making predictions...")
    probs = detect.sliding_prediction_full(model, flux)
    candidates = detect.extract_candidates(times, probs)

    # Визуализация результатов
    print("Visualizing results...")
    visualize.plot_lightcurve(times, flux, probs, candidates)

    # Детальный просмотр кандидата
    if candidates:
        print("Plotting candidate details...")
        visualize.plot_candidate_details(times, flux, candidates[0])
    else:
        print("No candidates found.")

if __name__ == '__main__':
    main()
