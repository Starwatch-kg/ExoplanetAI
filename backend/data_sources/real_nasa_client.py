"""
Real NASA Data Client - Production-ready module for accessing NASA/MAST/ExoFOP data
Модуль для работы с реальными данными NASA без синтетики
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import lightkurve as lk
    import astroquery
    from astroquery.mast import Observations
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astropy.timeseries import BoxLeastSquares
    ASTRO_AVAILABLE = True
except ImportError as e:
    ASTRO_AVAILABLE = False
    logging.warning(f"Astronomy packages not available: {e}. Install: pip install lightkurve astroquery astropy")

from core.logging import get_logger

logger = get_logger(__name__)


class RealNASAClient:
    """
    Production-ready client for NASA/MAST/ExoFOP data access
    Клиент для доступа к реальным данным NASA
    """
    
    def __init__(self, cache_dir: str = "data/nasa_cache", timeout: int = 30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        
        if not ASTRO_AVAILABLE:
            raise ImportError(
                "Required astronomy packages not installed. "
                "Run: pip install lightkurve astroquery astropy"
            )
    
    async def search_exoplanets(self, 
                               target_name: Optional[str] = None,
                               ra: Optional[float] = None, 
                               dec: Optional[float] = None,
                               radius: float = 0.1,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """
        Поиск экзопланет в NASA Exoplanet Archive
        
        Args:
            target_name: Название цели (TOI-715, Kepler-452b, etc.)
            ra: Прямое восхождение (градусы)
            dec: Склонение (градусы) 
            radius: Радиус поиска (градусы)
            limit: Максимальное количество результатов
            
        Returns:
            Список словарей с данными экзопланет
        """
        logger.info(f"Searching exoplanets: target={target_name}, ra={ra}, dec={dec}")
        
        try:
            if target_name:
                # Поиск по имени
                query = NasaExoplanetArchive.query_criteria(
                    table="pscomppars",
                    where=f"pl_name like '%{target_name}%'",
                    select="pl_name,hostname,pl_orbper,pl_rade,pl_masse,pl_eqt,ra,dec,sy_dist,discoverymethod,disc_year"
                )
            elif ra is not None and dec is not None:
                # Поиск по координатам
                coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
                query = NasaExoplanetArchive.query_region(
                    coord, radius=radius*u.deg,
                    table="pscomppars"
                )
            else:
                # Получение популярных планет
                query = NasaExoplanetArchive.query_criteria(
                    table="pscomppars",
                    where="pl_name in ('TOI-715 b','Kepler-452 b','TRAPPIST-1 e','Proxima Cen b','K2-18 b')",
                    select="pl_name,hostname,pl_orbper,pl_rade,pl_masse,pl_eqt,ra,dec,sy_dist,discoverymethod,disc_year"
                )
            
            if query is None or len(query) == 0:
                logger.warning("No exoplanets found in NASA archive")
                return []
            
            # Конвертируем в стандартный формат
            results = []
            for row in query[:limit]:
                planet_data = {
                    'name': str(row.get('pl_name', 'Unknown')),
                    'host_star': str(row.get('hostname', 'Unknown')),
                    'orbital_period_days': float(row.get('pl_orbper', 0)) if row.get('pl_orbper') else None,
                    'radius_earth_radii': float(row.get('pl_rade', 0)) if row.get('pl_rade') else None,
                    'mass_earth_masses': float(row.get('pl_masse', 0)) if row.get('pl_masse') else None,
                    'equilibrium_temperature_k': float(row.get('pl_eqt', 0)) if row.get('pl_eqt') else None,
                    'ra': float(row.get('ra', 0)) if row.get('ra') else None,
                    'dec': float(row.get('dec', 0)) if row.get('dec') else None,
                    'distance_parsecs': float(row.get('sy_dist', 0)) if row.get('sy_dist') else None,
                    'discovery_method': str(row.get('discoverymethod', 'Unknown')),
                    'discovery_year': int(row.get('disc_year', 0)) if row.get('disc_year') else None,
                    'status': 'Confirmed',
                    'source': 'NASA Exoplanet Archive'
                }
                results.append(planet_data)
            
            logger.info(f"Found {len(results)} exoplanets in NASA archive")
            return results
            
        except Exception as e:
            logger.error(f"Error searching NASA Exoplanet Archive: {e}")
            return []
    
    async def get_lightcurve(self, 
                           target_name: str,
                           mission: str = "TESS",
                           sector: Optional[int] = None,
                           cadence: str = "short") -> Optional[Dict[str, Any]]:
        """
        Загрузка реальной кривой блеска из MAST
        
        Args:
            target_name: Название цели
            mission: Миссия (TESS, Kepler, K2)
            sector: Сектор наблюдений (для TESS)
            cadence: Каденция ("short", "long")
            
        Returns:
            Словарь с данными кривой блеска или None
        """
        logger.info(f"Downloading lightcurve: {target_name} from {mission}")
        
        try:
            # Поиск данных
            if mission.upper() == "TESS":
                search_result = lk.search_lightcurve(
                    target_name, 
                    mission="TESS",
                    cadence=cadence
                )
                if sector:
                    search_result = search_result[search_result.sector == sector]
            elif mission.upper() in ["KEPLER", "K2"]:
                search_result = lk.search_lightcurve(
                    target_name,
                    mission=mission.upper(),
                    cadence=cadence
                )
            else:
                logger.error(f"Unsupported mission: {mission}")
                return None
            
            if len(search_result) == 0:
                logger.warning(f"No lightcurve data found for {target_name}")
                return None
            
            # Загрузка первого доступного файла
            lightcurve = search_result[0].download()
            
            if lightcurve is None:
                logger.warning(f"Failed to download lightcurve for {target_name}")
                return None
            
            # Очистка и нормализация
            lc_clean = lightcurve.remove_nans().remove_outliers(sigma=3)
            lc_normalized = lc_clean.normalize()
            
            # Конвертация в стандартный формат
            result = {
                'target_name': target_name,
                'mission': mission.upper(),
                'sector': getattr(lightcurve, 'sector', None),
                'cadence': cadence,
                'time': lc_normalized.time.value.tolist(),
                'flux': lc_normalized.flux.value.tolist(),
                'flux_err': lc_normalized.flux_err.value.tolist() if lc_normalized.flux_err is not None else None,
                'data_points': len(lc_normalized.time),
                'time_format': 'BJD - 2457000',
                'quality_flags': lc_normalized.quality.tolist() if hasattr(lc_normalized, 'quality') else None,
                'source': f"MAST/{mission.upper()}",
                'download_time': time.time()
            }
            
            logger.info(f"Successfully downloaded {len(result['time'])} data points for {target_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error downloading lightcurve for {target_name}: {e}")
            return None
    
    async def analyze_transit_bls(self, 
                                time: np.ndarray,
                                flux: np.ndarray,
                                period_min: float = 1.0,
                                period_max: float = 50.0,
                                duration_min: float = 0.01,
                                duration_max: float = 0.5) -> Dict[str, Any]:
        """
        Box Least Squares анализ транзитов на реальных данных
        
        Args:
            time: Массив времени
            flux: Массив потока
            period_min: Минимальный период поиска (дни)
            period_max: Максимальный период поиска (дни)
            duration_min: Минимальная длительность транзита (доли периода)
            duration_max: Максимальная длительность транзита (доли периода)
            
        Returns:
            Результаты BLS анализа
        """
        logger.info("Running BLS transit analysis on real data")
        
        try:
            # Подготовка данных
            time_clean = time[~np.isnan(flux)]
            flux_clean = flux[~np.isnan(flux)]
            
            if len(time_clean) < 100:
                logger.warning("Insufficient data points for BLS analysis")
                return {'error': 'Insufficient data points'}
            
            # BLS анализ
            bls = BoxLeastSquares(time_clean, flux_clean)
            
            # Определение сетки периодов
            periods = np.linspace(period_min, period_max, 10000)
            
            # Определение длительностей как функции периода
            durations = periods * np.linspace(duration_min, duration_max, 10)
            
            # Запуск BLS
            bls_result = bls.power(periods, durations)
            
            # Находим лучший период
            best_period = periods[np.argmax(bls_result.power)]
            best_power = np.max(bls_result.power)
            
            # Получаем параметры лучшего транзита
            best_params = bls.compute_stats(best_period, durations)
            
            # Вычисляем дополнительные метрики
            snr = best_power / np.std(bls_result.power)
            significance = (best_power - np.median(bls_result.power)) / np.std(bls_result.power)
            
            # Оценка глубины транзита
            transit_mask = best_params['transit_mask']
            if np.any(transit_mask):
                in_transit_flux = flux_clean[transit_mask]
                out_transit_flux = flux_clean[~transit_mask]
                transit_depth = 1.0 - np.median(in_transit_flux) / np.median(out_transit_flux)
            else:
                transit_depth = 0.0
            
            result = {
                'best_period': float(best_period),
                'best_power': float(best_power),
                'snr': float(snr),
                'significance': float(significance),
                'transit_depth': float(abs(transit_depth)),
                'transit_duration_hours': float(best_params.get('duration', 0) * 24),
                'transit_epoch': float(best_params.get('transit_time', 0)),
                'depth_ppm': float(abs(transit_depth) * 1e6),
                'periods_tested': len(periods),
                'data_points_used': len(time_clean),
                'method': 'Box Least Squares',
                'analysis_quality': 'real_data'
            }
            
            logger.info(f"BLS analysis complete: period={best_period:.3f}d, power={best_power:.3f}, SNR={snr:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in BLS analysis: {e}")
            return {'error': str(e)}
    
    async def get_stellar_parameters(self, target_name: str) -> Optional[Dict[str, Any]]:
        """
        Получение параметров звезды из каталогов
        
        Args:
            target_name: Название цели
            
        Returns:
            Параметры звезды или None
        """
        logger.info(f"Fetching stellar parameters for {target_name}")
        
        try:
            # Поиск в NASA Exoplanet Archive
            query = NasaExoplanetArchive.query_criteria(
                table="pscomppars",
                where=f"pl_name like '%{target_name}%' OR hostname like '%{target_name}%'",
                select="hostname,ra,dec,st_teff,st_rad,st_mass,st_logg,sy_vmag,sy_dist"
            )
            
            if query is None or len(query) == 0:
                logger.warning(f"No stellar parameters found for {target_name}")
                return None
            
            row = query[0]  # Берем первый результат
            
            stellar_params = {
                'star_name': str(row.get('hostname', 'Unknown')),
                'ra': float(row.get('ra', 0)) if row.get('ra') else None,
                'dec': float(row.get('dec', 0)) if row.get('dec') else None,
                'effective_temperature': float(row.get('st_teff', 0)) if row.get('st_teff') else None,
                'stellar_radius': float(row.get('st_rad', 0)) if row.get('st_rad') else None,
                'stellar_mass': float(row.get('st_mass', 0)) if row.get('st_mass') else None,
                'surface_gravity': float(row.get('st_logg', 0)) if row.get('st_logg') else None,
                'v_magnitude': float(row.get('sy_vmag', 0)) if row.get('sy_vmag') else None,
                'distance_pc': float(row.get('sy_dist', 0)) if row.get('sy_dist') else None,
                'source': 'NASA Exoplanet Archive'
            }
            
            logger.info(f"Retrieved stellar parameters for {stellar_params['star_name']}")
            return stellar_params
            
        except Exception as e:
            logger.error(f"Error fetching stellar parameters: {e}")
            return None
    
    async def validate_target(self, target_name: str) -> Dict[str, Any]:
        """
        Валидация цели в различных каталогах
        
        Args:
            target_name: Название цели
            
        Returns:
            Результаты валидации
        """
        logger.info(f"Validating target: {target_name}")
        
        validation_result = {
            'target_name': target_name,
            'is_valid': False,
            'found_in_catalogs': [],
            'available_missions': [],
            'recommended_mission': None,
            'planet_data': None,
            'stellar_data': None
        }
        
        try:
            # Проверка в NASA Exoplanet Archive
            exoplanet_data = await self.search_exoplanets(target_name=target_name)
            if exoplanet_data:
                validation_result['found_in_catalogs'].append('NASA Exoplanet Archive')
                validation_result['planet_data'] = exoplanet_data[0]
                validation_result['is_valid'] = True
            
            # Проверка доступности данных TESS
            try:
                tess_search = lk.search_lightcurve(target_name, mission="TESS")
                if len(tess_search) > 0:
                    validation_result['available_missions'].append('TESS')
                    validation_result['found_in_catalogs'].append('MAST/TESS')
            except:
                pass
            
            # Проверка доступности данных Kepler
            try:
                kepler_search = lk.search_lightcurve(target_name, mission="Kepler")
                if len(kepler_search) > 0:
                    validation_result['available_missions'].append('Kepler')
                    validation_result['found_in_catalogs'].append('MAST/Kepler')
            except:
                pass
            
            # Получение параметров звезды
            stellar_data = await self.get_stellar_parameters(target_name)
            if stellar_data:
                validation_result['stellar_data'] = stellar_data
            
            # Рекомендация миссии
            if 'TESS' in validation_result['available_missions']:
                validation_result['recommended_mission'] = 'TESS'
            elif 'Kepler' in validation_result['available_missions']:
                validation_result['recommended_mission'] = 'Kepler'
            
            validation_result['validation_time'] = time.time()
            
            logger.info(f"Target validation complete: {validation_result['is_valid']}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating target {target_name}: {e}")
            validation_result['error'] = str(e)
            return validation_result


# Глобальный экземпляр клиента
_nasa_client = None

def get_nasa_client() -> RealNASAClient:
    """Получить глобальный экземпляр NASA клиента"""
    global _nasa_client
    if _nasa_client is None:
        _nasa_client = RealNASAClient()
    return _nasa_client


async def test_nasa_client():
    """Тестирование NASA клиента"""
    client = get_nasa_client()
    
    # Тест поиска экзопланет
    print("Testing exoplanet search...")
    planets = await client.search_exoplanets(target_name="TOI-715")
    print(f"Found {len(planets)} planets")
    
    # Тест валидации цели
    print("Testing target validation...")
    validation = await client.validate_target("TOI-715")
    print(f"Validation result: {validation['is_valid']}")
    
    # Тест загрузки кривой блеска
    if validation['is_valid'] and validation['available_missions']:
        print("Testing lightcurve download...")
        mission = validation['recommended_mission']
        lc_data = await client.get_lightcurve("TOI-715", mission=mission)
        if lc_data:
            print(f"Downloaded {lc_data['data_points']} data points")
            
            # Тест BLS анализа
            print("Testing BLS analysis...")
            time_array = np.array(lc_data['time'])
            flux_array = np.array(lc_data['flux'])
            bls_result = await client.analyze_transit_bls(time_array, flux_array)
            print(f"BLS result: period={bls_result.get('best_period', 0):.3f}d")


if __name__ == "__main__":
    asyncio.run(test_nasa_client())
