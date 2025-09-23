"""
NASA Data Browser
Интеграция с реальными данными NASA для просмотра кривых блеска
"""

import asyncio
import aiohttp
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import json
import time
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class NASADataBrowser:
    """
    Браузер данных NASA для получения реальных кривых блеска
    """
    
    def __init__(self):
        # NASA API endpoints
        self.MAST_BASE_URL = "https://mast.stsci.edu/api/v0.1"
        self.EXOPLANET_ARCHIVE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        
        # Кэш для данных
        self.data_cache = {}
        self.cache_timeout = 3600  # 1 час
        
        # Лимиты запросов
        self.request_delay = 1.0  # секунды между запросами
        self.last_request_time = 0
        
    async def search_target(self, target_name: str, catalog: str = "TIC") -> Dict:
        """
        Поиск цели в каталогах NASA
        """
        logger.info(f"🔍 Поиск цели {target_name} в каталоге {catalog}")
        
        try:
            # Проверяем кэш
            cache_key = f"target_{catalog}_{target_name}"
            if self._check_cache(cache_key):
                return self.data_cache[cache_key]['data']
            
            # Подготавливаем запрос в зависимости от каталога
            if catalog == "TIC":
                result = await self._search_tic_target(target_name)
            elif catalog == "KIC":
                result = await self._search_kic_target(target_name)
            elif catalog == "EPIC":
                result = await self._search_epic_target(target_name)
            else:
                raise ValueError(f"Неподдерживаемый каталог: {catalog}")
            
            # Сохраняем в кэш
            self._save_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка поиска цели: {e}")
            return self._get_fallback_target_info(target_name, catalog)
    
    async def get_lightcurve_data(self, target_name: str, mission: str = "TESS",
                                sector: Optional[int] = None) -> Dict:
        """
        Получение данных кривой блеска из NASA
        """
        logger.info(f"📊 Загрузка кривой блеска для {target_name} ({mission})")
        
        try:
            # Проверяем кэш
            cache_key = f"lightcurve_{mission}_{target_name}_{sector}"
            if self._check_cache(cache_key):
                return self.data_cache[cache_key]['data']
            
            # Получаем данные в зависимости от миссии
            if mission == "TESS":
                result = await self._get_tess_lightcurve(target_name, sector)
            elif mission == "Kepler":
                result = await self._get_kepler_lightcurve(target_name)
            elif mission == "K2":
                result = await self._get_k2_lightcurve(target_name)
            else:
                raise ValueError(f"Неподдерживаемая миссия: {mission}")
            
            # Сохраняем в кэш
            self._save_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка загрузки кривой блеска: {e}")
            return self._get_fallback_lightcurve(target_name, mission)
    
    async def get_confirmed_planets(self, target_name: str) -> List[Dict]:
        """
        Получение информации о подтвержденных планетах
        """
        logger.info(f"🪐 Поиск подтвержденных планет для {target_name}")
        
        try:
            # Запрос к NASA Exoplanet Archive
            query = f"""
            SELECT pl_name, pl_orbper, pl_tranmid, pl_trandur, pl_trandep, 
                   pl_rade, pl_masse, st_rad, st_mass, st_teff
            FROM ps 
            WHERE hostname LIKE '%{target_name}%' OR pl_hostname LIKE '%{target_name}%'
            """
            
            params = {
                'query': query,
                'format': 'json'
            }
            
            await self._rate_limit()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.EXOPLANET_ARCHIVE_URL, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_planet_data(data)
                    else:
                        logger.warning(f"Exoplanet Archive запрос неуспешен: {response.status}")
                        return []
            
        except Exception as e:
            logger.error(f"Ошибка получения данных о планетах: {e}")
            return []
    
    async def _search_tic_target(self, target_name: str) -> Dict:
        """Поиск в TIC каталоге"""
        
        # Формируем запрос к MAST
        query = {
            "service": "Mast.Catalogs.Filtered.Tic",
            "format": "json",
            "params": {
                "columns": "ID,ra,dec,pmRA,pmDEC,plx,GAIAmag,Tmag,Teff,logg,MH,rad,mass,rho,lumclass,d,ebv,numcont,contratio,priority",
                "filters": [
                    {
                        "paramName": "ID",
                        "values": [{"min": int(target_name), "max": int(target_name)}]
                    }
                ]
            }
        }
        
        result = await self._make_mast_request(query)
        
        if result and result.get('data'):
            tic_data = result['data'][0]
            return self._format_tic_data(tic_data)
        else:
            raise ValueError(f"TIC {target_name} не найден")
    
    async def _search_kic_target(self, target_name: str) -> Dict:
        """Поиск в KIC каталоге"""
        
        query = {
            "service": "Mast.Catalogs.Filtered.Kic",
            "format": "json",
            "params": {
                "columns": "kepid,ra,dec,pmra,pmdec,umag,gmag,rmag,imag,zmag,gredmag,d51mag,jmag,hmag,kmag,kepmag,kp_teff,kp_logg,kp_feh,kp_ebminusv,kp_radius,kp_mass",
                "filters": [
                    {
                        "paramName": "kepid",
                        "values": [{"min": int(target_name), "max": int(target_name)}]
                    }
                ]
            }
        }
        
        result = await self._make_mast_request(query)
        
        if result and result.get('data'):
            kic_data = result['data'][0]
            return self._format_kic_data(kic_data)
        else:
            raise ValueError(f"KIC {target_name} не найден")
    
    async def _search_epic_target(self, target_name: str) -> Dict:
        """Поиск в EPIC каталоге"""
        
        query = {
            "service": "Mast.Catalogs.Filtered.K2targets",
            "format": "json",
            "params": {
                "columns": "epic_number,ra,dec,kepmag,kp_teff,kp_logg,kp_feh,kp_radius,kp_mass",
                "filters": [
                    {
                        "paramName": "epic_number",
                        "values": [{"min": int(target_name), "max": int(target_name)}]
                    }
                ]
            }
        }
        
        result = await self._make_mast_request(query)
        
        if result and result.get('data'):
            epic_data = result['data'][0]
            return self._format_epic_data(epic_data)
        else:
            raise ValueError(f"EPIC {target_name} не найден")
    
    async def _get_tess_lightcurve(self, target_name: str, sector: Optional[int] = None) -> Dict:
        """Получение TESS кривой блеска"""
        
        # Поиск доступных данных TESS
        query = {
            "service": "Mast.Caom.Filtered",
            "format": "json",
            "params": {
                "columns": "obsID,obs_id,target_name,t_min,t_max,em_min,em_max,dataproduct_type,calib_level,obs_collection,instrument_name,project,filters,productFilename",
                "filters": [
                    {
                        "paramName": "obs_collection",
                        "values": ["TESS"]
                    },
                    {
                        "paramName": "target_name",
                        "values": [f"TIC {target_name}"]
                    },
                    {
                        "paramName": "dataproduct_type",
                        "values": ["timeseries"]
                    }
                ]
            }
        }
        
        if sector:
            query["params"]["filters"].append({
                "paramName": "sequence_number",
                "values": [sector]
            })
        
        result = await self._make_mast_request(query)
        
        if result and result.get('data'):
            # Берем первый доступный файл
            observation = result['data'][0]
            return await self._download_tess_data(observation)
        else:
            raise ValueError(f"TESS данные для TIC {target_name} не найдены")
    
    async def _get_kepler_lightcurve(self, target_name: str) -> Dict:
        """Получение Kepler кривой блеска"""
        
        query = {
            "service": "Mast.Caom.Filtered",
            "format": "json",
            "params": {
                "columns": "obsID,obs_id,target_name,t_min,t_max,dataproduct_type,calib_level,obs_collection,instrument_name,productFilename",
                "filters": [
                    {
                        "paramName": "obs_collection",
                        "values": ["Kepler"]
                    },
                    {
                        "paramName": "target_name",
                        "values": [f"KIC {target_name}"]
                    },
                    {
                        "paramName": "dataproduct_type",
                        "values": ["timeseries"]
                    }
                ]
            }
        }
        
        result = await self._make_mast_request(query)
        
        if result and result.get('data'):
            observation = result['data'][0]
            return await self._download_kepler_data(observation)
        else:
            raise ValueError(f"Kepler данные для KIC {target_name} не найдены")
    
    async def _get_k2_lightcurve(self, target_name: str) -> Dict:
        """Получение K2 кривой блеска"""
        
        query = {
            "service": "Mast.Caom.Filtered",
            "format": "json",
            "params": {
                "columns": "obsID,obs_id,target_name,t_min,t_max,dataproduct_type,calib_level,obs_collection,instrument_name,productFilename",
                "filters": [
                    {
                        "paramName": "obs_collection",
                        "values": ["K2"]
                    },
                    {
                        "paramName": "target_name",
                        "values": [f"EPIC {target_name}"]
                    },
                    {
                        "paramName": "dataproduct_type",
                        "values": ["timeseries"]
                    }
                ]
            }
        }
        
        result = await self._make_mast_request(query)
        
        if result and result.get('data'):
            observation = result['data'][0]
            return await self._download_k2_data(observation)
        else:
            raise ValueError(f"K2 данные для EPIC {target_name} не найдены")
    
    async def _make_mast_request(self, query: Dict) -> Dict:
        """Выполнение запроса к MAST API"""
        
        await self._rate_limit()
        
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'ExoplanetAI/1.0'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.MAST_BASE_URL}/invoke",
                json=query,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"MAST API ошибка: {response.status}")
                    raise aiohttp.ClientError(f"MAST API returned {response.status}")
    
    async def _download_tess_data(self, observation: Dict) -> Dict:
        """Загрузка TESS данных"""
        
        # В реальной реализации здесь был бы код для загрузки FITS файлов
        # Пока возвращаем симулированные данные на основе метаданных
        
        logger.info(f"Симуляция загрузки TESS данных: {observation.get('productFilename', 'unknown')}")
        
        # Генерируем реалистичные данные на основе временного диапазона
        t_min = observation.get('t_min', 2458000)  # TESS BJD
        t_max = observation.get('t_max', t_min + 27)  # ~27 дней сектор
        
        n_points = 13000  # ~2-минутная каденция
        time = np.linspace(t_min, t_max, n_points)
        
        # Базовый поток с реалистичным шумом
        flux = np.ones(n_points)
        flux += np.random.normal(0, 100e-6, n_points)  # 100 ppm шум
        
        # Добавляем звездную активность
        rotation_period = np.random.uniform(10, 30)
        flux += 0.001 * np.sin(2 * np.pi * time / rotation_period)
        
        flux_err = np.full(n_points, 100e-6)
        
        return {
            "time": time.tolist(),
            "flux": flux.tolist(),
            "flux_err": flux_err.tolist(),
            "mission": "TESS",
            "target_name": observation.get('target_name', 'Unknown'),
            "sector": observation.get('sequence_number', 1),
            "cadence_minutes": 2,
            "data_source": "NASA MAST (simulated)",
            "observation_id": observation.get('obsID', 'unknown')
        }
    
    async def _download_kepler_data(self, observation: Dict) -> Dict:
        """Загрузка Kepler данных"""
        
        logger.info(f"Симуляция загрузки Kepler данных: {observation.get('productFilename', 'unknown')}")
        
        t_min = observation.get('t_min', 120)  # Kepler BJD - 2454833
        t_max = observation.get('t_max', t_min + 90)  # ~90 дней квартал
        
        n_points = 4320  # 30-минутная каденция
        time = np.linspace(t_min, t_max, n_points)
        
        flux = np.ones(n_points)
        flux += np.random.normal(0, 50e-6, n_points)  # 50 ppm шум
        
        # Звездная активность
        rotation_period = np.random.uniform(15, 35)
        flux += 0.0005 * np.sin(2 * np.pi * time / rotation_period)
        
        flux_err = np.full(n_points, 50e-6)
        
        return {
            "time": time.tolist(),
            "flux": flux.tolist(),
            "flux_err": flux_err.tolist(),
            "mission": "Kepler",
            "target_name": observation.get('target_name', 'Unknown'),
            "quarter": observation.get('sequence_number', 1),
            "cadence_minutes": 30,
            "data_source": "NASA MAST (simulated)",
            "observation_id": observation.get('obsID', 'unknown')
        }
    
    async def _download_k2_data(self, observation: Dict) -> Dict:
        """Загрузка K2 данных"""
        
        logger.info(f"Симуляция загрузки K2 данных: {observation.get('productFilename', 'unknown')}")
        
        t_min = observation.get('t_min', 2000)  # K2 BJD - 2454833
        t_max = observation.get('t_max', t_min + 80)  # ~80 дней кампания
        
        n_points = 3840  # 30-минутная каденция
        time = np.linspace(t_min, t_max, n_points)
        
        flux = np.ones(n_points)
        flux += np.random.normal(0, 80e-6, n_points)  # 80 ppm шум
        
        # K2 имеет больше систематических эффектов
        flux += 0.002 * np.sin(2 * np.pi * time / 6.0)  # Тепловые вариации
        
        flux_err = np.full(n_points, 80e-6)
        
        return {
            "time": time.tolist(),
            "flux": flux.tolist(),
            "flux_err": flux_err.tolist(),
            "mission": "K2",
            "target_name": observation.get('target_name', 'Unknown'),
            "campaign": observation.get('sequence_number', 1),
            "cadence_minutes": 30,
            "data_source": "NASA MAST (simulated)",
            "observation_id": observation.get('obsID', 'unknown')
        }
    
    def _format_tic_data(self, data: Dict) -> Dict:
        """Форматирование TIC данных"""
        return {
            "target_id": str(data.get('ID', 'unknown')),
            "catalog": "TIC",
            "ra": float(data.get('ra', 0)),
            "dec": float(data.get('dec', 0)),
            "magnitude": float(data.get('Tmag', 0)),
            "temperature": float(data.get('Teff', 5778)),
            "radius": float(data.get('rad', 1.0)),
            "mass": float(data.get('mass', 1.0)),
            "distance": float(data.get('d', 100)),
            "data_source": "NASA MAST TIC"
        }
    
    def _format_kic_data(self, data: Dict) -> Dict:
        """Форматирование KIC данных"""
        return {
            "target_id": str(data.get('kepid', 'unknown')),
            "catalog": "KIC",
            "ra": float(data.get('ra', 0)),
            "dec": float(data.get('dec', 0)),
            "magnitude": float(data.get('kepmag', 0)),
            "temperature": float(data.get('kp_teff', 5778)),
            "radius": float(data.get('kp_radius', 1.0)),
            "mass": float(data.get('kp_mass', 1.0)),
            "metallicity": float(data.get('kp_feh', 0.0)),
            "data_source": "NASA MAST KIC"
        }
    
    def _format_epic_data(self, data: Dict) -> Dict:
        """Форматирование EPIC данных"""
        return {
            "target_id": str(data.get('epic_number', 'unknown')),
            "catalog": "EPIC",
            "ra": float(data.get('ra', 0)),
            "dec": float(data.get('dec', 0)),
            "magnitude": float(data.get('kepmag', 0)),
            "temperature": float(data.get('kp_teff', 5778)),
            "radius": float(data.get('kp_radius', 1.0)),
            "mass": float(data.get('kp_mass', 1.0)),
            "metallicity": float(data.get('kp_feh', 0.0)),
            "data_source": "NASA MAST EPIC"
        }
    
    def _process_planet_data(self, data: List[Dict]) -> List[Dict]:
        """Обработка данных о планетах"""
        planets = []
        
        for planet in data:
            if planet.get('pl_name'):
                planets.append({
                    "name": planet['pl_name'],
                    "period": float(planet.get('pl_orbper', 0)) if planet.get('pl_orbper') else None,
                    "epoch": float(planet.get('pl_tranmid', 0)) if planet.get('pl_tranmid') else None,
                    "duration": float(planet.get('pl_trandur', 0)) if planet.get('pl_trandur') else None,
                    "depth": float(planet.get('pl_trandep', 0)) if planet.get('pl_trandep') else None,
                    "radius": float(planet.get('pl_rade', 0)) if planet.get('pl_rade') else None,
                    "mass": float(planet.get('pl_masse', 0)) if planet.get('pl_masse') else None,
                    "confirmed": True,
                    "data_source": "NASA Exoplanet Archive"
                })
        
        return planets
    
    async def _rate_limit(self):
        """Ограничение частоты запросов"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def _check_cache(self, key: str) -> bool:
        """Проверка кэша"""
        if key in self.data_cache:
            cache_time = self.data_cache[key]['timestamp']
            if time.time() - cache_time < self.cache_timeout:
                return True
            else:
                del self.data_cache[key]
        return False
    
    def _save_to_cache(self, key: str, data: Any):
        """Сохранение в кэш"""
        self.data_cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def _get_fallback_target_info(self, target_name: str, catalog: str) -> Dict:
        """Fallback информация о цели"""
        return {
            "target_id": target_name,
            "catalog": catalog,
            "ra": 0.0,
            "dec": 0.0,
            "magnitude": 12.0,
            "temperature": 5778.0,
            "radius": 1.0,
            "mass": 1.0,
            "data_source": "Fallback (NASA API unavailable)",
            "note": "Real NASA data temporarily unavailable"
        }
    
    def _get_fallback_lightcurve(self, target_name: str, mission: str) -> Dict:
        """Fallback кривая блеска"""
        
        # Параметры по умолчанию для разных миссий
        mission_params = {
            "TESS": {"duration": 27, "cadence": 2, "n_points": 13000, "noise": 100e-6},
            "Kepler": {"duration": 90, "cadence": 30, "n_points": 4320, "noise": 50e-6},
            "K2": {"duration": 80, "cadence": 30, "n_points": 3840, "noise": 80e-6}
        }
        
        params = mission_params.get(mission, mission_params["TESS"])
        
        time = np.linspace(0, params["duration"], params["n_points"])
        flux = np.ones(params["n_points"]) + np.random.normal(0, params["noise"], params["n_points"])
        flux_err = np.full(params["n_points"], params["noise"])
        
        return {
            "time": time.tolist(),
            "flux": flux.tolist(),
            "flux_err": flux_err.tolist(),
            "mission": mission,
            "target_name": target_name,
            "cadence_minutes": params["cadence"],
            "data_source": "Fallback (NASA API unavailable)",
            "note": "Simulated data - real NASA data temporarily unavailable"
        }

# Глобальный экземпляр
nasa_browser = NASADataBrowser()
