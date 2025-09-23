import asyncio
import logging
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from astroquery.mast import Catalogs, Observations
from astroquery.exceptions import ResolverError
import lightkurve as lk
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
import os
import pickle
from pathlib import Path

from models.search_models import LightCurveData, TargetInfo

logger = logging.getLogger(__name__)

class DataService:
    """Сервис для работы с астрономическими данными"""
    
    def __init__(self):
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.catalogs_cache = {}
        
    async def initialize(self):
        """Инициализация сервиса"""
        logger.info("Инициализация DataService")
        # Предварительная загрузка часто используемых каталогов
        await self._preload_catalogs()
        
    async def _preload_catalogs(self):
        """Предварительная загрузка каталогов"""
        try:
            # Можно добавить предварительную загрузку популярных целей
            logger.info("Каталоги готовы к использованию")
        except Exception as e:
            logger.warning(f"Не удалось предзагрузить каталоги: {e}")
    
    async def get_lightcurve(
        self, 
        target_name: str, 
        catalog: str = "TIC", 
        mission: str = "TESS"
    ) -> Optional[LightCurveData]:
        """
        Получение кривой блеска для указанной цели
        """
        try:
            # Проверяем кэш
            cache_key = f"{target_name}_{catalog}_{mission}"
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                logger.info(f"Данные для {target_name} загружены из кэша")
                return cached_data
            
            logger.info(f"Загрузка данных для {target_name} из {mission}")
            
            # Поиск цели в каталоге
            target_info = await self._resolve_target(target_name, catalog)
            if not target_info:
                logger.error(f"Цель {target_name} не найдена в каталоге {catalog}")
                return None
            
            # Загрузка кривой блеска
            lightcurve = await self._download_lightcurve(target_info, mission)
            if not lightcurve:
                logger.error(f"Не удалось загрузить кривую блеска для {target_name}")
                return None
            
            # Обработка и очистка данных
            processed_lc = await self._process_lightcurve(lightcurve, target_name, mission)
            
            # Сохранение в кэш
            await self._save_to_cache(cache_key, processed_lc)
            
            return processed_lc
            
        except Exception as e:
            logger.error(f"Ошибка при получении кривой блеска для {target_name}: {e}")
            return None
    
    async def _resolve_target(self, target_name: str, catalog: str) -> Optional[Dict[str, Any]]:
        """Поиск цели в каталоге"""
        try:
            if catalog == "TIC":
                # Поиск в TIC каталоге
                if target_name.startswith("TIC"):
                    tic_id = target_name.replace("TIC", "").strip()
                else:
                    tic_id = target_name
                
                catalog_data = Catalogs.query_criteria(
                    catalog="Tic",
                    ID=int(tic_id)
                )
                
            elif catalog == "KIC":
                # Поиск в Kepler Input Catalog
                if target_name.startswith("KIC"):
                    kic_id = target_name.replace("KIC", "").strip()
                else:
                    kic_id = target_name
                    
                catalog_data = Catalogs.query_criteria(
                    catalog="Kic",
                    kic_kepler_id=int(kic_id)
                )
                
            elif catalog == "EPIC":
                # Поиск в K2 каталоге
                if target_name.startswith("EPIC"):
                    epic_id = target_name.replace("EPIC", "").strip()
                else:
                    epic_id = target_name
                    
                catalog_data = Catalogs.query_criteria(
                    catalog="K2targets",
                    k2_id=int(epic_id)
                )
            else:
                logger.error(f"Неподдерживаемый каталог: {catalog}")
                return None
            
            if len(catalog_data) == 0:
                logger.warning(f"Цель {target_name} не найдена в каталоге {catalog}")
                return None
            
            target_row = catalog_data[0]
            
            return {
                "target_name": target_name,
                "ra": float(target_row.get("ra", 0)),
                "dec": float(target_row.get("dec", 0)),
                "magnitude": float(target_row.get("Tmag", target_row.get("kepmag", 0))),
                "catalog_data": target_row
            }
            
        except Exception as e:
            logger.error(f"Ошибка при поиске цели {target_name}: {e}")
            return None
    
    async def _download_lightcurve(self, target_info: Dict[str, Any], mission: str):
        """Загрузка кривой блеска с MAST"""
        try:
            target_name = target_info["target_name"]
            
            if mission.upper() == "TESS":
                # Загрузка данных TESS
                search_result = lk.search_lightcurve(
                    target_name, 
                    mission="TESS"
                )
                
            elif mission.upper() == "KEPLER":
                # Загрузка данных Kepler
                search_result = lk.search_lightcurve(
                    target_name,
                    mission="Kepler"
                )
                
            elif mission.upper() == "K2":
                # Загрузка данных K2
                search_result = lk.search_lightcurve(
                    target_name,
                    mission="K2"
                )
            else:
                logger.error(f"Неподдерживаемая миссия: {mission}")
                return None
            
            if len(search_result) == 0:
                logger.warning(f"Данные для {target_name} в миссии {mission} не найдены")
                return None
            
            # Загружаем первый доступный файл
            lightcurve = search_result[0].download()
            
            return lightcurve
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке кривой блеска: {e}")
            return None
    
    async def _process_lightcurve(self, lightcurve, target_name: str, mission: str) -> LightCurveData:
        """Обработка и очистка кривой блеска"""
        try:
            # Удаление выбросов и нормализация
            lc_clean = lightcurve.remove_outliers(sigma=5)
            lc_clean = lc_clean.normalize()
            
            # Удаление NaN значений
            mask = np.isfinite(lc_clean.flux.value)
            
            time = lc_clean.time.value[mask]
            flux = lc_clean.flux.value[mask]
            flux_err = lc_clean.flux_err.value[mask] if hasattr(lc_clean, 'flux_err') else None
            quality = lc_clean.quality.value[mask] if hasattr(lc_clean, 'quality') else None
            
            # Вычисление статистики
            duration_days = float(np.max(time) - np.min(time))
            cadence = float(np.median(np.diff(time)))
            data_points = len(time)
            
            # Определение сектора/квартала/кампании
            sector = getattr(lightcurve, 'sector', None)
            quarter = getattr(lightcurve, 'quarter', None) 
            campaign = getattr(lightcurve, 'campaign', None)
            
            return LightCurveData(
                target_name=target_name,
                time=time.tolist(),
                flux=flux.tolist(),
                flux_err=flux_err.tolist() if flux_err is not None else None,
                quality=quality.tolist() if quality is not None else None,
                mission=mission,
                sector=sector,
                quarter=quarter,
                campaign=campaign,
                duration_days=duration_days,
                cadence=cadence,
                data_points=data_points
            )
            
        except Exception as e:
            logger.error(f"Ошибка при обработке кривой блеска: {e}")
            raise
    
    async def search_targets(self, query: str, catalog: str = "TIC", limit: int = 10) -> List[TargetInfo]:
        """Поиск целей по запросу"""
        try:
            targets = []
            
            if catalog == "TIC":
                # Поиск в TIC каталоге
                catalog_data = Catalogs.query_criteria(
                    catalog="Tic",
                    objectname=query
                )[:limit]
                
                for row in catalog_data:
                    target_info = TargetInfo(
                        target_name=f"TIC {row['ID']}",
                        catalog_id=str(row['ID']),
                        coordinates={"ra": float(row['ra']), "dec": float(row['dec'])},
                        magnitude=float(row.get('Tmag', 0)),
                        available_data=["TESS"]
                    )
                    targets.append(target_info)
            
            return targets
            
        except Exception as e:
            logger.error(f"Ошибка при поиске целей: {e}")
            return []
    
    async def _get_from_cache(self, cache_key: str) -> Optional[LightCurveData]:
        """Получение данных из кэша"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Ошибка при чтении кэша: {e}")
        return None
    
    async def _save_to_cache(self, cache_key: str, data: LightCurveData):
        """Сохранение данных в кэш"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Ошибка при сохранении в кэш: {e}")
    
    async def get_known_exoplanets(self, target_name: str) -> List[Dict[str, Any]]:
        """Получение информации об известных экзопланетах для цели"""
        try:
            # Интеграция с NASA Data Browser для получения реальных данных
            try:
                from nasa_data_browser import nasa_browser
                confirmed_planets = await nasa_browser.get_confirmed_planets(target_name)
                return confirmed_planets
            except ImportError:
                logger.warning("NASA Data Browser недоступен")
                
            # Fallback к локальной базе известных планет
            from known_exoplanets import get_target_info
            target_info = get_target_info(target_name)
            
            if target_info.get('has_planets') and target_info.get('planets'):
                return target_info['planets']
            
            return []
        except Exception as e:
            logger.error(f"Ошибка при получении известных экзопланет: {e}")
            return []
