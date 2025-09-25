"""
Модуль для загрузки и предобработки данных TESS/MAST
Использует Lightkurve и MAST для получения кривых блеска
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional, Union
import warnings
from pathlib import Path

# Астрономические библиотеки
import lightkurve as lk
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.mast import Catalogs

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Подавление предупреждений от Lightkurve
warnings.filterwarnings('ignore', category=UserWarning, module='lightkurve')

class TESSDataLoader:
    """
    Класс для загрузки и предобработки данных TESS
    Поддерживает загрузку по TIC ID или координатам
    """
    
    def __init__(self, cache_dir: str = "data/tess_cache"):
        """
        Инициализация загрузчика данных TESS
        
        Args:
            cache_dir: Директория для кэширования данных
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройки для обработки данных
        self.min_cadence_length = 1000  # Минимальная длина кривой блеска
        self.max_outlier_fraction = 0.1  # Максимальная доля выбросов
        
    def load_by_tic_id(self, tic_id: Union[int, str], sectors: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Загружает кривую блеска по TIC ID
        
        Args:
            tic_id: TIC ID звезды
            sectors: Список секторов TESS для загрузки (None = все доступные)
            
        Returns:
            Tuple[times, fluxes]: Временные метки и потоки
        """
        try:
            logger.info(f"Загрузка данных для TIC {tic_id}")
            
            # Поиск данных в MAST
            search_result = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS')
            
            if len(search_result) == 0:
                raise ValueError(f"Данные для TIC {tic_id} не найдены")
            
            logger.info(f"Найдено {len(search_result)} наборов данных")
            
            # Фильтрация по секторам если указаны
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
    
    def load_by_coordinates(self, ra: float, dec: float, radius: float = 0.1, 
                          sectors: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Загружает кривую блеска по координатам
        
        Args:
            ra: Прямое восхождение (градусы)
            dec: Склонение (градусы)
            radius: Радиус поиска (градусы)
            sectors: Список секторов TESS
            
        Returns:
            Tuple[times, fluxes]: Временные метки и потоки
        """
        try:
            logger.info(f"Загрузка данных для координат RA={ra}, Dec={dec}")
            
            # Создание координат
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
            
            # Поиск данных
            search_result = lk.search_lightcurve(coord, mission='TESS', radius=radius*u.deg)
            
            if len(search_result) == 0:
                raise ValueError(f"Данные для координат {ra}, {dec} не найдены")
            
            logger.info(f"Найдено {len(search_result)} наборов данных")
            
            # Фильтрация по секторам
            if sectors is not None:
                search_result = search_result[search_result.sector.isin(sectors)]
            
            # Выбор ближайшей звезды
            if len(search_result) > 1:
                logger.info("Найдено несколько звезд, выбираем ближайшую")
                # Сортируем по расстоянию от центра поиска
                distances = []
                for idx in range(len(search_result)):
                    star_coord = SkyCoord(ra=search_result[idx].ra, dec=search_result[idx].dec, unit=(u.deg, u.deg))
                    distances.append(coord.separation(star_coord).arcsec)
                
                closest_idx = np.argmin(distances)
                search_result = search_result[closest_idx:closest_idx+1]
                logger.info(f"Выбрана звезда на расстоянии {distances[closest_idx]:.2f} arcsec")
            
            # Загрузка данных
            lc_collection = search_result.download_all()
            lc = lc_collection.stitch()
            
            # Предобработка
            times, fluxes = self._preprocess_lightcurve(lc)
            
            logger.info(f"Успешно загружено {len(times)} точек данных")
            return times, fluxes
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке координат {ra}, {dec}: {str(e)}")
            raise
    
    def _preprocess_lightcurve(self, lc) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предобработка кривой блеска
        
        Args:
            lc: Объект LightCurve из Lightkurve
            
        Returns:
            Tuple[times, fluxes]: Обработанные временные метки и потоки
        """
        logger.info("Начинаем предобработку кривой блеска")
        
        # Удаление NaN значений
        lc = lc.remove_nans()
        logger.info(f"После удаления NaN: {len(lc)} точек")
        
        # Удаление выбросов (значения > 5 сигм)
        lc = lc.remove_outliers(sigma=5.0)
        logger.info(f"После удаления выбросов: {len(lc)} точек")
        
        # Проверка минимальной длины
        if len(lc) < self.min_cadence_length:
            raise ValueError(f"Кривая блеска слишком короткая: {len(lc)} < {self.min_cadence_length}")
        
        # Проверка доли выбросов
        outlier_fraction = 1 - len(lc) / len(lc.remove_nans())
        if outlier_fraction > self.max_outlier_fraction:
            logger.warning(f"Высокая доля выбросов: {outlier_fraction:.2%}")
        
        # Сглаживание кривой блеска (удаление долгопериодических вариаций)
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
    
    def batch_load_tic_ids(self, tic_ids: List[Union[int, str]], 
                          sectors: Optional[List[int]] = None,
                          max_workers: int = 4) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Пакетная загрузка данных для списка TIC ID
        
        Args:
            tic_ids: Список TIC ID
            sectors: Список секторов TESS
            max_workers: Максимальное количество потоков
            
        Returns:
            List[Tuple[times, fluxes]]: Список загруженных кривых блеска
        """
        logger.info(f"Пакетная загрузка {len(tic_ids)} звезд")
        
        results = []
        failed_ids = []
        
        for i, tic_id in enumerate(tic_ids):
            try:
                logger.info(f"Загрузка {i+1}/{len(tic_ids)}: TIC {tic_id}")
                times, fluxes = self.load_by_tic_id(tic_id, sectors)
                results.append((times, fluxes))
                
            except Exception as e:
                logger.error(f"Ошибка загрузки TIC {tic_id}: {e}")
                failed_ids.append(tic_id)
                results.append(None)
        
        logger.info(f"Успешно загружено: {len(results) - len(failed_ids)}/{len(tic_ids)}")
        if failed_ids:
            logger.warning(f"Не удалось загрузить: {failed_ids}")
        
        return results
    
    def get_star_info(self, tic_id: Union[int, str]) -> dict:
        """
        Получает информацию о звезде из каталога TESS
        
        Args:
            tic_id: TIC ID звезды
            
        Returns:
            dict: Информация о звезде
        """
        try:
            # Поиск в каталоге TESS
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
    
    def save_lightcurve(self, times: np.ndarray, fluxes: np.ndarray, 
                       filename: str, metadata: Optional[dict] = None):
        """
        Сохраняет кривую блеска в файл
        
        Args:
            times: Временные метки
            fluxes: Потоки
            filename: Имя файла
            metadata: Дополнительные метаданные
        """
        filepath = self.cache_dir / filename
        
        # Создание DataFrame
        df = pd.DataFrame({
            'time': times,
            'flux': fluxes
        })
        
        # Добавление метаданных
        if metadata:
            for key, value in metadata.items():
                df[key] = value
        
        # Сохранение
        df.to_csv(filepath, index=False)
        logger.info(f"Кривая блеска сохранена: {filepath}")
    
    def load_lightcurve(self, filename: str) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Загружает кривую блеска из файла
        
        Args:
            filename: Имя файла
            
        Returns:
            Tuple[times, fluxes, metadata]: Данные и метаданные
        """
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Файл {filepath} не найден")
        
        df = pd.read_csv(filepath)
        
        times = df['time'].values
        fluxes = df['flux'].values
        
        # Извлечение метаданных
        metadata = {}
        for col in df.columns:
            if col not in ['time', 'flux']:
                metadata[col] = df[col].iloc[0] if len(df) > 0 else None
        
        logger.info(f"Кривая блеска загружена: {filepath}")
        return times, fluxes, metadata


def create_tess_dataset(tic_ids: List[Union[int, str]], 
                       output_file: str = "data/tess_dataset.csv",
                       sectors: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Создает датасет кривых блеска TESS для обучения
    
    Args:
        tic_ids: Список TIC ID для загрузки
        output_file: Путь к выходному файлу
        sectors: Список секторов TESS
        
    Returns:
        pd.DataFrame: Датасет с кривыми блеска
    """
    logger.info(f"Создание датасета TESS из {len(tic_ids)} звезд")
    
    loader = TESSDataLoader()
    dataset_rows = []
    
    for i, tic_id in enumerate(tic_ids):
        try:
            logger.info(f"Обработка {i+1}/{len(tic_ids)}: TIC {tic_id}")
            
            # Загрузка данных
            times, fluxes = loader.load_by_tic_id(tic_id, sectors)
            
            # Получение информации о звезде
            star_info = loader.get_star_info(tic_id)
            
            # Создание записи
            row = {
                'tic_id': tic_id,
                'times': times.tolist(),  # Сохраняем как список
                'fluxes': fluxes.tolist(),
                'length': len(times),
                'ra': star_info.get('ra'),
                'dec': star_info.get('dec'),
                'tmag': star_info.get('tmag'),
                'teff': star_info.get('teff'),
                'logg': star_info.get('logg'),
                'radius': star_info.get('radius'),
                'mass': star_info.get('mass'),
            }
            
            dataset_rows.append(row)
            
        except Exception as e:
            logger.error(f"Ошибка обработки TIC {tic_id}: {e}")
            continue
    
    # Создание DataFrame
    dataset = pd.DataFrame(dataset_rows)
    
    # Сохранение
    dataset.to_csv(output_file, index=False)
    logger.info(f"Датасет сохранен: {output_file}")
    logger.info(f"Успешно обработано: {len(dataset)}/{len(tic_ids)} звезд")
    
    return dataset


if __name__ == "__main__":
    # Пример использования
    logger.info("Тестирование загрузчика данных TESS")
    
    # Создание загрузчика
    loader = TESSDataLoader()
    
    # Тестовые TIC ID (известные звезды с экзопланетами)
    test_tic_ids = [
        261136679,  # TOI-700
        38846515,   # TOI-715
        142802581,  # TOI-715
    ]
    
    try:
        # Загрузка данных для первой звезды
        tic_id = test_tic_ids[0]
        times, fluxes = loader.load_by_tic_id(tic_id)
        
        # Получение информации о звезде
        star_info = loader.get_star_info(tic_id)
        
        # Сохранение данных
        metadata = {'tic_id': tic_id, **star_info}
        loader.save_lightcurve(times, fluxes, f"tic_{tic_id}_lightcurve.csv", metadata)
        
        logger.info("Тест успешно завершен")
        
    except Exception as e:
        logger.error(f"Ошибка в тесте: {e}")
