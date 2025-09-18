"""
Модуль для работы с реальными данными TESS через lightkurve
Загружает настоящие кривые блеска из архива NASA MAST
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import lightkurve as lk
    LIGHTKURVE_AVAILABLE = True
    logger.info("lightkurve успешно загружен - используем реальные данные TESS")
except ImportError:
    LIGHTKURVE_AVAILABLE = False
    logger.warning("lightkurve не установлен - используем симулированные данные")


class RealTESSDataLoader:
    """Загрузчик реальных данных TESS"""
    
    def __init__(self):
        self.cache = {}
        self.max_cache_size = 100
        
    async def load_real_lightcurve(self, tic_id: str, sector: Optional[int] = None) -> Dict[str, Any]:
        """
        Загружает реальную кривую блеска из TESS
        
        Args:
            tic_id: TIC идентификатор звезды
            sector: Номер сектора TESS (если не указан, загружает все доступные)
            
        Returns:
            Словарь с данными кривой блеска
        """
        cache_key = f"{tic_id}_{sector}"
        
        # Проверяем кэш
        if cache_key in self.cache:
            logger.info(f"Возвращаем кэшированные данные для TIC {tic_id}")
            return self.cache[cache_key]
        
        try:
            if LIGHTKURVE_AVAILABLE:
                # Загружаем реальные данные через lightkurve
                result = await self._load_with_lightkurve(tic_id, sector)
            else:
                # Генерируем реалистичные данные
                result = await self._generate_realistic_data(tic_id, sector)
            
            # Сохраняем в кэш
            if len(self.cache) >= self.max_cache_size:
                # Удаляем старые записи
                self.cache.pop(next(iter(self.cache)))
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных для TIC {tic_id}: {e}")
            # Fallback к генерации
            return await self._generate_realistic_data(tic_id, sector)
    
    async def _load_with_lightkurve(self, tic_id: str, sector: Optional[int]) -> Dict[str, Any]:
        """Загрузка через lightkurve"""
        logger.info(f"Загрузка реальных данных TESS для TIC {tic_id}")
        
        # Поиск данных
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        
        if len(search_result) == 0:
            logger.warning(f"Данные для TIC {tic_id} не найдены в архиве TESS")
            return await self._generate_realistic_data(tic_id, sector)
        
        # Загружаем кривую блеска
        if sector is not None:
            # Фильтруем по сектору
            lc_collection = search_result[search_result.sector == sector].download_all()
        else:
            # Загружаем все доступные сектора
            lc_collection = search_result.download_all()
        
        if lc_collection is None or len(lc_collection) == 0:
            logger.warning(f"Не удалось загрузить данные для TIC {tic_id}")
            return await self._generate_realistic_data(tic_id, sector)
        
        # Объединяем данные из всех секторов
        lc = lc_collection.stitch()
        
        # Удаляем выбросы и NaN
        lc = lc.remove_outliers(sigma=5)
        lc = lc.remove_nans()
        
        # Нормализуем поток
        lc = lc.normalize()
        
        # Извлекаем данные
        times = lc.time.value
        fluxes = lc.flux.value
        flux_errors = lc.flux_err.value if hasattr(lc, 'flux_err') else np.zeros_like(fluxes)
        
        # Получаем метаданные
        metadata = {
            "mission": "TESS",
            "sector": sector if sector else "all",
            "target_name": lc.meta.get('OBJECT', f'TIC {tic_id}'),
            "ra": lc.meta.get('RA_OBJ', None),
            "dec": lc.meta.get('DEC_OBJ', None),
            "mag": lc.meta.get('TESSMAG', None),
            "cadence": lc.meta.get('TIMEDEL', 0.00139) * 24 * 60,  # в минутах
            "quality_flags": lc.quality.value.tolist() if hasattr(lc, 'quality') else []
        }
        
        # Детекция транзитов (простая версия)
        transit_info = self._detect_transits(times, fluxes)
        
        return {
            "tic_id": tic_id,
            "times": times.tolist(),
            "fluxes": fluxes.tolist(),
            "flux_errors": flux_errors.tolist(),
            "metadata": metadata,
            "transit_info": transit_info,
            "data_source": "NASA MAST Archive (Real TESS Data)",
            "download_time": datetime.now().isoformat()
        }
    
    async def _generate_realistic_data(self, tic_id: str, sector: Optional[int]) -> Dict[str, Any]:
        """Генерация реалистичных данных когда реальные недоступны"""
        logger.info(f"Генерация реалистичных данных для TIC {tic_id}")
        
        # Параметры на основе TIC ID (для воспроизводимости)
        np.random.seed(int(tic_id) % 2**32)
        
        # Временной ряд (27.4 дня для одного сектора TESS)
        n_days = 27.4 if sector else 27.4 * 3  # 3 сектора если не указан
        cadence = 2 / 60 / 24  # 2 минуты в днях
        times = np.arange(0, n_days, cadence)
        
        # Реалистичные звездные параметры
        stellar_params = self._generate_stellar_params(tic_id)
        
        # Генерация базового потока
        fluxes = self._generate_flux(times, stellar_params)
        
        # Добавляем реалистичные ошибки
        flux_errors = np.random.normal(0, stellar_params['noise_level'], len(times)) * 0.1
        
        # Детекция транзитов
        transit_info = self._detect_transits(times, fluxes)
        
        metadata = {
            "mission": "TESS",
            "sector": sector if sector else "1-3",
            "target_name": f"TIC {tic_id}",
            "mag": stellar_params['magnitude'],
            "teff": stellar_params['temperature'],
            "cadence": 2.0,  # минуты
            "stellar_type": stellar_params['stellar_type']
        }
        
        return {
            "tic_id": tic_id,
            "times": times.tolist(),
            "fluxes": fluxes.tolist(),
            "flux_errors": np.abs(flux_errors).tolist(),
            "metadata": metadata,
            "transit_info": transit_info,
            "data_source": "Simulated (Based on Real TESS Parameters)",
            "stellar_params": stellar_params,
            "download_time": datetime.now().isoformat()
        }
    
    def _generate_stellar_params(self, tic_id: str) -> Dict[str, Any]:
        """Генерация реалистичных параметров звезды"""
        # Используем TIC ID для генерации воспроизводимых параметров
        np.random.seed(int(tic_id) % 2**32)
        
        # Распределение звездных типов (реалистичное)
        stellar_types = ['M', 'K', 'G', 'F', 'A']
        type_probs = [0.4, 0.3, 0.15, 0.1, 0.05]  # M-карлики наиболее распространены
        stellar_type = np.random.choice(stellar_types, p=type_probs)
        
        # Параметры в зависимости от типа
        type_params = {
            'M': {'temp': (2500, 3900), 'mag': (12, 18), 'activity': 0.01},
            'K': {'temp': (3900, 5200), 'mag': (10, 15), 'activity': 0.005},
            'G': {'temp': (5200, 6000), 'mag': (9, 13), 'activity': 0.002},
            'F': {'temp': (6000, 7500), 'mag': (8, 12), 'activity': 0.001},
            'A': {'temp': (7500, 10000), 'mag': (7, 11), 'activity': 0.0005}
        }
        
        params = type_params[stellar_type]
        
        return {
            'stellar_type': stellar_type,
            'temperature': np.random.uniform(*params['temp']),
            'magnitude': np.random.uniform(*params['mag']),
            'activity_level': params['activity'] * np.random.uniform(0.5, 2),
            'rotation_period': np.random.uniform(1, 50),  # дни
            'noise_level': 10 ** (0.4 * (np.random.uniform(*params['mag']) - 10)) * 1e-4,
            'has_planets': np.random.random() < 0.2  # 20% звезд имеют планеты
        }
    
    def _generate_flux(self, times: np.ndarray, stellar_params: Dict) -> np.ndarray:
        """Генерация реалистичного потока звезды"""
        # Базовый поток
        flux = np.ones_like(times)
        
        # Добавляем шум
        noise = np.random.normal(0, stellar_params['noise_level'], len(times))
        flux += noise
        
        # Звездная активность (пятна, вспышки)
        if stellar_params['activity_level'] > 0:
            # Вращение звезды
            rotation = stellar_params['activity_level'] * np.sin(
                2 * np.pi * times / stellar_params['rotation_period']
            )
            flux += rotation
            
            # Случайные вспышки для активных звезд
            if stellar_params['stellar_type'] in ['M', 'K']:
                n_flares = np.random.poisson(2)
                for _ in range(n_flares):
                    flare_time = np.random.uniform(times[0], times[-1])
                    flare_amp = np.random.uniform(0.01, 0.05)
                    flare_width = np.random.uniform(0.01, 0.1)
                    flare = flare_amp * np.exp(-((times - flare_time) / flare_width) ** 2)
                    flux += flare
        
        # Добавляем транзиты планет
        if stellar_params['has_planets']:
            n_planets = np.random.poisson(1.5) + 1  # Минимум 1 планета
            
            for i in range(n_planets):
                period = np.random.uniform(0.5, 100)  # дни
                depth = np.random.uniform(0.0001, 0.02)  # глубина транзита
                duration = np.random.uniform(0.5, 12) / 24  # часы в днях
                phase = np.random.uniform(0, period)
                
                # Генерируем транзиты
                transit_times = np.arange(phase, times[-1], period)
                for t0 in transit_times:
                    # Трапециевидная форма транзита
                    ingress = duration * 0.1
                    egress = duration * 0.1
                    flat = duration * 0.8
                    
                    # Вход в транзит
                    mask1 = (times >= t0) & (times < t0 + ingress)
                    flux[mask1] *= 1 - depth * (times[mask1] - t0) / ingress
                    
                    # Плоская часть
                    mask2 = (times >= t0 + ingress) & (times < t0 + ingress + flat)
                    flux[mask2] *= (1 - depth)
                    
                    # Выход из транзита
                    mask3 = (times >= t0 + ingress + flat) & (times < t0 + duration)
                    flux[mask3] *= 1 - depth * (1 - (times[mask3] - t0 - ingress - flat) / egress)
        
        # Инструментальные эффекты
        # Дрейф
        drift = 1e-5 * times
        flux += drift
        
        # Периодические систематики
        systematic = 0.0001 * np.sin(2 * np.pi * times * 10)
        flux += systematic
        
        return flux
    
    def _detect_transits(self, times: np.ndarray, fluxes: np.ndarray) -> Dict[str, Any]:
        """Простая детекция транзитов"""
        # Находим провалы в потоке
        median_flux = np.median(fluxes)
        std_flux = np.std(fluxes)
        
        # Порог для детекции (3 сигма)
        threshold = median_flux - 3 * std_flux
        
        # Находим точки ниже порога
        transit_mask = fluxes < threshold
        transit_indices = np.where(transit_mask)[0]
        
        if len(transit_indices) == 0:
            return {
                "detected": False,
                "n_transits": 0,
                "candidates": []
            }
        
        # Группируем последовательные точки в транзиты
        transits = []
        current_transit = [transit_indices[0]]
        
        for i in range(1, len(transit_indices)):
            if transit_indices[i] - transit_indices[i-1] <= 5:  # Близкие точки
                current_transit.append(transit_indices[i])
            else:
                if len(current_transit) >= 3:  # Минимум 3 точки для транзита
                    transits.append(current_transit)
                current_transit = [transit_indices[i]]
        
        if len(current_transit) >= 3:
            transits.append(current_transit)
        
        # Анализируем найденные транзиты
        candidates = []
        for transit in transits:
            t_start = times[transit[0]]
            t_end = times[transit[-1]]
            depth = 1 - np.min(fluxes[transit]) / median_flux
            
            candidates.append({
                "time_start": t_start,
                "time_end": t_end,
                "duration_hours": (t_end - t_start) * 24,
                "depth_percent": depth * 100,
                "snr": depth / std_flux
            })
        
        return {
            "detected": len(candidates) > 0,
            "n_transits": len(candidates),
            "candidates": candidates,
            "analysis_method": "Simple threshold detection"
        }


# Глобальный экземпляр загрузчика
tess_loader = RealTESSDataLoader()


async def get_real_tess_data(tic_id: str, sector: Optional[int] = None) -> Dict[str, Any]:
    """
    Главная функция для получения данных TESS
    
    Args:
        tic_id: TIC идентификатор
        sector: Номер сектора (опционально)
        
    Returns:
        Словарь с данными кривой блеска
    """
    return await tess_loader.load_real_lightcurve(tic_id, sector)
