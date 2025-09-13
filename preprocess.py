"""
Модуль предобработки данных TESS/MAST.

Этот модуль содержит функции для загрузки, нормализации и подготовки данных
кривых блеска TESS для дальнейшего анализа экзопланет.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
import warnings

# Астрономические библиотеки
import lightkurve as lk
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.mast import Catalogs

# Настройка логирования
logger = logging.getLogger(__name__)

# Подавление предупреждений
warnings.filterwarnings('ignore', category=UserWarning, module='lightkurve')


class TESSDataProcessor:
    """Класс для загрузки и предобработки данных TESS."""
    
    def __init__(self, cache_dir: str = "data/tess_cache"):
        """
        Инициализация процессора данных TESS.
        
        Args:
            cache_dir: Директория для кэширования данных.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Параметры обработки
        self.min_cadence_length = 1000
        self.max_outlier_fraction = 0.1
        
    def load_lightcurve(self, tic_id: Union[int, str], 
                       sectors: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Загружает кривую блеска по TIC ID.
        
        Args:
            tic_id: TIC ID звезды.
            sectors: Список секторов TESS для загрузки.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Временные метки и потоки.
            
        Raises:
            ValueError: Если данные не найдены или повреждены.
        """
        try:
            logger.info(f"Загрузка данных для TIC {tic_id}")
            
            # Поиск данных в MAST
            search_result = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS')
            
            if len(search_result) == 0:
                raise ValueError(f"Данные для TIC {tic_id} не найдены")
            
            logger.info(f"Найдено {len(search_result)} наборов данных")
            
            # Фильтрация по секторам
            if sectors is not None:
                search_result = search_result[search_result.sector.isin(sectors)]
                logger.info(f"Фильтрация по секторам {sectors}: {len(search_result)} наборов")
            
            # Загрузка и объединение данных
            lc_collection = search_result.download_all()
            
            if len(lc_collection) == 0:
                raise ValueError("Нет данных после фильтрации")
            
            # Объединение кривых блеска
            lc = lc_collection.stitch()
            
            # Предобработка
            times, fluxes = self._preprocess_lightcurve(lc)
            
            logger.info(f"Успешно загружено {len(times)} точек данных")
            return times, fluxes
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке TIC {tic_id}: {str(e)}")
            raise
    
    def _preprocess_lightcurve(self, lc) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предобработка кривой блеска.
        
        Args:
            lc: Объект LightCurve из Lightkurve.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Обработанные временные метки и потоки.
        """
        logger.info("Начинаем предобработку кривой блеска")
        
        # Удаление NaN значений
        lc = lc.remove_nans()
        logger.info(f"После удаления NaN: {len(lc)} точек")
        
        # Удаление выбросов
        lc = lc.remove_outliers(sigma=5.0)
        logger.info(f"После удаления выбросов: {len(lc)} точек")
        
        # Проверка минимальной длины
        if len(lc) < self.min_cadence_length:
            raise ValueError(f"Кривая блеска слишком короткая: {len(lc)} < {self.min_cadence_length}")
        
        # Сглаживание кривой блеска
        try:
            lc = lc.flatten(window_length=101)
            logger.info("Применено сглаживание кривой блеска")
        except Exception as e:
            logger.warning(f"Не удалось применить сглаживание: {e}")
        
        # Нормализация потока
        flux_normalized = lc.flux.value
        flux_normalized = (flux_normalized - np.median(flux_normalized)) / np.std(flux_normalized)
        
        # Получение временных меток
        times = lc.time.value
        
        logger.info("Предобработка завершена")
        return times, flux_normalized
    
    def normalize_data(self, fluxes: np.ndarray, 
                      remove_trends: bool = True,
                      filter_noise: bool = True) -> np.ndarray:
        """
        Нормализация данных кривой блеска.
        
        Args:
            fluxes: Массив потоков.
            remove_trends: Удалять ли долгопериодические тренды.
            filter_noise: Фильтровать ли высокочастотный шум.
            
        Returns:
            np.ndarray: Нормализованные потоки.
        """
        logger.info("Нормализация данных кривой блеска")
        
        normalized_fluxes = fluxes.copy()
        
        # Удаление трендов
        if remove_trends:
            normalized_fluxes = self._remove_trends(normalized_fluxes)
            logger.info("Удалены долгопериодические тренды")
        
        # Фильтрация шума
        if filter_noise:
            normalized_fluxes = self._filter_noise(normalized_fluxes)
            logger.info("Отфильтрован высокочастотный шум")
        
        # Стандартная нормализация
        normalized_fluxes = (normalized_fluxes - np.mean(normalized_fluxes)) / np.std(normalized_fluxes)
        
        logger.info("Нормализация завершена")
        return normalized_fluxes
    
    def _remove_trends(self, fluxes: np.ndarray) -> np.ndarray:
        """
        Удаление долгопериодических трендов.
        
        Args:
            fluxes: Массив потоков.
            
        Returns:
            np.ndarray: Потоки без трендов.
        """
        # Используем скользящее среднее для удаления трендов
        window_size = min(101, len(fluxes) // 10)
        if window_size % 2 == 0:
            window_size += 1
        
        trend = np.convolve(fluxes, np.ones(window_size)/window_size, mode='same')
        return fluxes - trend
    
    def _filter_noise(self, fluxes: np.ndarray) -> np.ndarray:
        """
        Фильтрация высокочастотного шума.
        
        Args:
            fluxes: Массив потоков.
            
        Returns:
            np.ndarray: Отфильтрованные потоки.
        """
        from scipy import signal
        
        # Простой фильтр низких частот
        nyquist = 0.5
        cutoff = 0.1  # Отсекаем частоты выше 10% от Найквиста
        
        b, a = signal.butter(4, cutoff / nyquist, btype='low')
        filtered_fluxes = signal.filtfilt(b, a, fluxes)
        
        return filtered_fluxes
    
    def prepare_training_data(self, lightcurves: List[Tuple[np.ndarray, np.ndarray]], 
                            window_size: int = 2000,
                            stride: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка данных для обучения модели.
        
        Args:
            lightcurves: Список кривых блеска (times, fluxes).
            window_size: Размер окна для сегментации.
            stride: Шаг между окнами.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Окна данных и метки.
        """
        logger.info("Подготовка данных для обучения")
        
        windows = []
        labels = []
        
        for times, fluxes in lightcurves:
            # Нормализация
            normalized_fluxes = self.normalize_data(fluxes)
            
            # Создание окон
            for start in range(0, len(normalized_fluxes) - window_size + 1, stride):
                window = normalized_fluxes[start:start + window_size]
                windows.append(window)
                
                # Простая эвристика для меток (в реальности нужны истинные метки)
                # Если есть значительные падения - возможный транзит
                if np.any(window < np.mean(window) - 3 * np.std(window)):
                    labels.append(1)  # Транзит
                else:
                    labels.append(0)  # Нет транзита
        
        windows_array = np.array(windows)
        labels_array = np.array(labels)
        
        logger.info(f"Подготовлено {len(windows_array)} окон данных")
        logger.info(f"Распределение классов: {np.bincount(labels_array)}")
        
        return windows_array, labels_array
    
    def get_star_info(self, tic_id: Union[int, str]) -> Dict:
        """
        Получает информацию о звезде из каталога TESS.
        
        Args:
            tic_id: TIC ID звезды.
            
        Returns:
            Dict: Информация о звезде.
        """
        try:
            catalog_data = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC")
            
            if len(catalog_data) == 0:
                raise ValueError(f"Звезда TIC {tic_id} не найдена в каталоге")
            
            star_info = {
                'tic_id': tic_id,
                'ra': float(catalog_data['ra'][0]),
                'dec': float(catalog_data['dec'][0]),
                'tmag': float(catalog_data['Tmag'][0]) if 'Tmag' in catalog_data.colnames else None,
                'teff': float(catalog_data['Teff'][0]) if 'Teff' in catalog_data.colnames else None,
                'logg': float(catalog_data['logg'][0]) if 'logg' in catalog_data.colnames else None,
                'radius': float(catalog_data['rad'][0]) if 'rad' in catalog_data.colnames else None,
                'mass': float(catalog_data['mass'][0]) if 'mass' in catalog_data.colnames else None,
            }
            
            return star_info
            
        except Exception as e:
            logger.error(f"Ошибка получения информации о звезде TIC {tic_id}: {e}")
            return {'tic_id': tic_id, 'error': str(e)}


def load_multiple_stars(tic_ids: List[Union[int, str]], 
                       sectors: Optional[List[int]] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Загружает данные для нескольких звезд.
    
    Args:
        tic_ids: Список TIC ID.
        sectors: Список секторов TESS.
        
    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: Список кривых блеска.
    """
    processor = TESSDataProcessor()
    lightcurves = []
    
    for tic_id in tic_ids:
        try:
            times, fluxes = processor.load_lightcurve(tic_id, sectors)
            lightcurves.append((times, fluxes))
        except Exception as e:
            logger.error(f"Не удалось загрузить TIC {tic_id}: {e}")
            continue
    
    logger.info(f"Успешно загружено {len(lightcurves)}/{len(tic_ids)} кривых блеска")
    return lightcurves


def create_synthetic_data(num_samples: int = 100, 
                        length: int = 2000,
                        transit_fraction: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Создает синтетические данные для тестирования.
    
    Args:
        num_samples: Количество образцов.
        length: Длина кривых блеска.
        transit_fraction: Доля образцов с транзитами.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Данные и метки.
    """
    logger.info(f"Создание {num_samples} синтетических образцов")
    
    data = []
    labels = []
    
    for i in range(num_samples):
        # Базовый поток с шумом
        times = np.linspace(0, 30, length)
        flux = np.ones_like(times) + 0.01 * np.random.randn(length)
        
        # Добавление транзита
        if np.random.rand() < transit_fraction:
            period = np.random.uniform(3, 20)
            depth = np.random.uniform(0.005, 0.03)
            duration = np.random.uniform(0.1, 0.5)
            
            for j, t in enumerate(times):
                phase = (t % period) / period
                if 0.45 <= phase <= 0.55:  # Транзит
                    flux[j] -= depth
            
            labels.append(1)
        else:
            labels.append(0)
        
        data.append(flux)
    
    return np.array(data), np.array(labels)
