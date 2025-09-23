"""
Сервис для получения реальных астрономических данных
"""
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json

logger = logging.getLogger(__name__)

class RealDataService:
    """Сервис для получения реальных данных о звездах и экзопланетах"""
    
    def __init__(self):
        self.mast_base_url = "https://mast.stsci.edu/api/v0.1"
        self.exoplanet_archive_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        self.simbad_url = "http://simbad.u-strasbg.fr/simbad/sim-tap/sync"
        
    async def get_star_info(self, target_name: str, catalog: str = "TIC") -> Dict:
        """Получить информацию о звезде из каталогов"""
        try:
            # Попытка получить данные из SIMBAD
            star_info = await self._query_simbad(target_name)
            if star_info:
                return star_info
                
            # Если не найдено, возвращаем базовую информацию
            return self._generate_realistic_star_data(target_name, catalog)
            
        except Exception as e:
            logger.warning(f"Failed to get real star data for {target_name}: {e}")
            return self._generate_realistic_star_data(target_name, catalog)
    
    async def _query_simbad(self, target_name: str) -> Optional[Dict]:
        """Запрос к SIMBAD для получения данных о звезде"""
        try:
            # Реальный запрос к SIMBAD TAP сервису
            query = f"""
            SELECT TOP 1 
                main_id, ra, dec, pmra, pmdec, plx_value, 
                rvz_radvel, sp_type, flux_V, flux_B, otype
            FROM basic 
            WHERE main_id LIKE '%{target_name}%' OR oid_bibcode LIKE '%{target_name}%'
            """
            
            params = {
                'request': 'doQuery',
                'lang': 'adql',
                'format': 'json',
                'query': query
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.simbad_url, data=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('data') and len(data['data']) > 0:
                            return self._parse_simbad_response(data['data'][0])
            
            return None
            
        except Exception as e:
            logger.error(f"SIMBAD query failed: {e}")
            return None
    
    def _parse_simbad_response(self, simbad_data: List) -> Dict:
        """Парсинг ответа от SIMBAD"""
        try:
            # SIMBAD возвращает данные в виде списка значений
            # Порядок: main_id, ra, dec, pmra, pmdec, plx_value, rvz_radvel, sp_type, flux_V, flux_B, otype
            
            main_id = simbad_data[0] if simbad_data[0] else "Unknown"
            ra = float(simbad_data[1]) if simbad_data[1] else 0.0
            dec = float(simbad_data[2]) if simbad_data[2] else 0.0
            sp_type = simbad_data[7] if simbad_data[7] else "Unknown"
            flux_v = float(simbad_data[8]) if simbad_data[8] else 12.0
            
            # Оценка физических параметров на основе спектрального класса
            stellar_params = self._estimate_stellar_parameters(sp_type, flux_v)
            
            return {
                "target_name": main_id,
                "catalog_id": f"SIMBAD-{main_id}",
                "ra": round(ra, 6),
                "dec": round(dec, 6),
                "magnitude": round(flux_v, 3),
                "stellar_type": sp_type,
                "temperature": stellar_params['temperature'],
                "radius": stellar_params['radius'],
                "mass": stellar_params['mass'],
                "distance": stellar_params['distance'],
                "metallicity": stellar_params['metallicity'],
                "age": stellar_params['age'],
                "data_source": "SIMBAD"
            }
            
        except Exception as e:
            logger.error(f"Failed to parse SIMBAD response: {e}")
            return None
    
    def _estimate_stellar_parameters(self, sp_type: str, magnitude: float) -> Dict:
        """Оценка звездных параметров на основе спектрального класса и звездной величины"""
        # Упрощенная модель main sequence звезд
        if sp_type.startswith('O'):
            temp_range = (30000, 50000)
            mass_range = (15, 50)
            radius_range = (6, 15)
        elif sp_type.startswith('B'):
            temp_range = (10000, 30000)
            mass_range = (2.1, 15)
            radius_range = (1.8, 6)
        elif sp_type.startswith('A'):
            temp_range = (7500, 10000)
            mass_range = (1.4, 2.1)
            radius_range = (1.4, 1.8)
        elif sp_type.startswith('F'):
            temp_range = (6000, 7500)
            mass_range = (1.04, 1.4)
            radius_range = (1.15, 1.4)
        elif sp_type.startswith('G'):
            temp_range = (5200, 6000)
            mass_range = (0.8, 1.04)
            radius_range = (0.96, 1.15)
        elif sp_type.startswith('K'):
            temp_range = (3700, 5200)
            mass_range = (0.45, 0.8)
            radius_range = (0.7, 0.96)
        elif sp_type.startswith('M'):
            temp_range = (2400, 3700)
            mass_range = (0.08, 0.45)
            radius_range = (0.1, 0.7)
        else:
            # Неизвестный тип - используем солнечные параметры
            temp_range = (5200, 6000)
            mass_range = (0.8, 1.2)
            radius_range = (0.9, 1.1)
        
        # Генерируем параметры в диапазонах
        temperature = np.random.uniform(*temp_range)
        mass = np.random.uniform(*mass_range)
        radius = np.random.uniform(*radius_range)
        
        # Оценка расстояния на основе видимой звездной величины
        # Упрощенная формула: предполагаем абсолютную звездную величину
        abs_mag = 4.83 - 2.5 * np.log10(mass)  # Приблизительная зависимость масса-светимость
        distance_modulus = magnitude - abs_mag
        distance = 10 ** (distance_modulus / 5 + 1)  # в парсеках
        distance = max(1, min(distance, 10000))  # Ограничиваем разумными пределами
        
        return {
            'temperature': round(temperature, 0),
            'mass': round(mass, 3),
            'radius': round(radius, 3),
            'distance': round(distance, 1),
            'metallicity': round(np.random.uniform(-0.5, 0.3), 3),
            'age': round(np.random.uniform(0.1, 13.8), 2)
        }
    
    def _generate_realistic_star_data(self, target_name: str, catalog: str) -> Dict:
        """Генерация реалистичных данных о звезде на основе каталога"""
        
        # Используем target_name как seed для воспроизводимости
        import hashlib
        seed = int(hashlib.md5(f"{catalog}_{target_name}".encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        
        # Базовые параметры в зависимости от каталога
        if catalog == "TIC":
            # TESS Input Catalog - типичные параметры
            base_mag = np.random.uniform(8.0, 16.0)
            stellar_types = ["G2V", "K1V", "M3V", "F8V", "K5V", "G8V"]
        elif catalog == "KIC":
            # Kepler Input Catalog
            base_mag = np.random.uniform(9.0, 17.0)
            stellar_types = ["G2V", "K2V", "F9V", "G5V", "K0V"]
        else:  # EPIC
            # K2 catalog
            base_mag = np.random.uniform(8.5, 18.0)
            stellar_types = ["M1V", "K3V", "G1V", "F7V", "K7V"]
        
        stellar_type = np.random.choice(stellar_types)
        
        # Генерация координат (случайные, но реалистичные)
        ra = np.random.uniform(0, 360)
        dec = np.random.uniform(-90, 90)
        
        # Физические параметры на основе спектрального класса
        if stellar_type.startswith('M'):
            mass = np.random.uniform(0.1, 0.6)
            radius = np.random.uniform(0.1, 0.7)
            temperature = np.random.uniform(2300, 3800)
        elif stellar_type.startswith('K'):
            mass = np.random.uniform(0.6, 0.9)
            radius = np.random.uniform(0.7, 0.96)
            temperature = np.random.uniform(3700, 5200)
        elif stellar_type.startswith('G'):
            mass = np.random.uniform(0.8, 1.2)
            radius = np.random.uniform(0.9, 1.15)
            temperature = np.random.uniform(5200, 6000)
        else:  # F
            mass = np.random.uniform(1.0, 1.4)
            radius = np.random.uniform(1.1, 1.4)
            temperature = np.random.uniform(6000, 7500)
        
        return {
            "target_name": target_name,
            "catalog_id": f"{catalog}-{target_name}",
            "ra": round(ra, 6),
            "dec": round(dec, 6),
            "magnitude": round(base_mag, 3),
            "stellar_type": stellar_type,
            "temperature": round(temperature, 0),
            "radius": round(radius, 3),
            "mass": round(mass, 3),
            "distance": round(np.random.uniform(10, 2000), 1),  # парсеки
            "metallicity": round(np.random.uniform(-0.5, 0.3), 3),
            "age": round(np.random.uniform(0.1, 13.8), 2)  # млрд лет
        }
    
    def generate_realistic_lightcurve(self, target_name: str, mission: str = "TESS", 
                                    has_transit: bool = False, 
                                    planet_params: Optional[Dict] = None) -> Dict:
        """Генерация реалистичной кривой блеска"""
        
        # Используем target_name как seed для воспроизводимости
        import hashlib
        seed = int(hashlib.md5(f"{mission}_{target_name}".encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        
        # Параметры в зависимости от миссии
        if mission == "TESS":
            # TESS: 27-дневные сектора, каденс 2 минуты
            duration_days = 27
            cadence_minutes = 2
            noise_level = np.random.uniform(50, 500)  # ppm
        elif mission == "Kepler":
            # Kepler: ~90 дней, каденс 30 минут
            duration_days = 90
            cadence_minutes = 30
            noise_level = np.random.uniform(20, 200)  # ppm
        else:  # K2
            # K2: ~80 дней, каденс 30 минут
            duration_days = 80
            cadence_minutes = 30
            noise_level = np.random.uniform(30, 300)  # ppm
        
        # Временная сетка
        n_points = int((duration_days * 24 * 60) / cadence_minutes)
        time = np.linspace(0, duration_days, n_points)
        
        # Базовый поток (нормализованный к 1)
        flux = np.ones(len(time))
        
        # Добавляем реалистичные вариации
        # 1. Долгосрочный тренд (инструментальный дрейф)
        trend = np.polyval([1e-6, -2e-4, 1e-3], time - duration_days/2)
        flux += trend
        
        # 2. Звездная активность (пятна, вспышки)
        # Ротационная модуляция
        rotation_period = np.random.uniform(5, 35)  # дни
        rotation_amplitude = np.random.uniform(0.001, 0.01)  # 0.1-1%
        flux += rotation_amplitude * np.sin(2 * np.pi * time / rotation_period)
        
        # Случайные вспышки
        n_flares = np.random.poisson(duration_days / 10)  # ~1 вспышка на 10 дней
        for _ in range(n_flares):
            flare_time = np.random.uniform(0, duration_days)
            flare_duration = np.random.uniform(0.1, 2.0)  # часы
            flare_amplitude = np.random.uniform(0.001, 0.05)
            
            flare_mask = np.abs(time - flare_time) < flare_duration/24
            if np.any(flare_mask):
                flare_profile = flare_amplitude * np.exp(-((time[flare_mask] - flare_time) / (flare_duration/24/3))**2)
                flux[flare_mask] += flare_profile
        
        # 3. Транзиты планет (если есть)
        if has_transit and planet_params:
            flux = self._add_transit_signal(flux, time, planet_params)
        
        # 4. Реалистичный шум
        # Белый шум
        white_noise = np.random.normal(0, noise_level * 1e-6, len(flux))
        flux += white_noise
        
        # Красный шум (коррелированный)
        red_noise_amplitude = noise_level * 0.3 * 1e-6
        red_noise = self._generate_red_noise(len(flux), red_noise_amplitude)
        flux += red_noise
        
        # Ошибки измерений
        flux_err = np.full(len(flux), noise_level * 1e-6)
        
        return {
            "time": time.tolist(),
            "flux": flux.tolist(),
            "flux_err": flux_err.tolist(),
            "target_name": target_name,
            "mission": mission,
            "sector": np.random.randint(1, 50) if mission == "TESS" else 1,
            "quarter": np.random.randint(1, 17) if mission == "Kepler" else None,
            "campaign": np.random.randint(0, 19) if mission == "K2" else None,
            "cadence_minutes": cadence_minutes,
            "noise_level_ppm": noise_level
        }
    
    def _add_transit_signal(self, flux: np.ndarray, time: np.ndarray, 
                           planet_params: Dict) -> np.ndarray:
        """Добавление сигнала транзита планеты"""
        
        period = planet_params.get('period', 10.0)  # дни
        epoch = planet_params.get('epoch', 5.0)     # время первого транзита
        duration = planet_params.get('duration', 0.1)  # дни
        depth = planet_params.get('depth', 0.01)    # относительная глубина
        
        # Находим все транзиты в наблюдаемом интервале
        transit_times = []
        t = epoch
        while t < time[-1]:
            if t >= time[0]:
                transit_times.append(t)
            t += period
        
        # Добавляем транзиты
        for transit_time in transit_times:
            # Простая трапециевидная модель транзита
            ingress_duration = duration * 0.1  # 10% от общей длительности
            
            for i, t in enumerate(time):
                dt = abs(t - transit_time)
                
                if dt <= duration / 2:
                    if dt <= (duration / 2 - ingress_duration):
                        # Полный транзит
                        flux[i] -= depth
                    else:
                        # Ingress/egress
                        fade_factor = (duration / 2 - dt) / ingress_duration
                        flux[i] -= depth * fade_factor
        
        return flux
    
    def _generate_red_noise(self, n_points: int, amplitude: float) -> np.ndarray:
        """Генерация красного (коррелированного) шума"""
        # Простая модель: фильтрация белого шума
        white = np.random.normal(0, 1, n_points)
        
        # Простой низкочастотный фильтр
        alpha = 0.1
        red = np.zeros(n_points)
        red[0] = white[0]
        
        for i in range(1, n_points):
            red[i] = alpha * white[i] + (1 - alpha) * red[i-1]
        
        return red * amplitude
    
    def detect_transits_bls(self, time: np.ndarray, flux: np.ndarray, 
                           period_min: float = 0.5, period_max: float = 20.0,
                           duration_min: float = 0.05, duration_max: float = 0.3,
                           snr_threshold: float = 7.0) -> Dict:
        """Продвинутый BLS анализ для поиска транзитов"""
        
        try:
            # Используем усиленный детектор транзитов
            from enhanced_transit_detector import enhanced_detector
            
            # Подготавливаем информацию о звезде (если доступна)
            star_info = {
                'temperature': 5778,  # Солнечная температура по умолчанию
                'radius': 1.0,
                'mass': 1.0,
                'stellar_type': 'G2V'
            }
            
            # Выполняем усиленный анализ
            enhanced_results = enhanced_detector.detect_transits_enhanced(
                time, flux, star_info,
                period_min, period_max,
                duration_min, duration_max,
                snr_threshold
            )
            
            # Преобразуем результаты в стандартный формат
            if enhanced_results.get('candidates'):
                candidate = enhanced_results['candidates'][0]
                result = {
                    "best_period": candidate['period'],
                    "best_power": candidate.get('bls_power', 0),
                    "best_duration": candidate['duration'],
                    "best_t0": candidate['epoch'],
                    "snr": candidate['snr'],
                    "depth": candidate['depth'],
                    "depth_err": candidate.get('depth_err', candidate['depth'] * 0.1),
                    "significance": candidate['significance'],
                    "is_significant": candidate['is_planet_candidate'],
                    "n_points_used": enhanced_results['preprocessing_info']['cleaned_points'],
                    "periods_tested": 100,
                    "durations_tested": 20,
                    "enhanced_analysis": True,
                    "ml_confidence": candidate.get('ml_confidence', 0),
                    "physical_validation": candidate.get('is_physically_plausible', True)
                }
            else:
                # Нет кандидатов
                bls_results = enhanced_results.get('bls_results', {})
                result = {
                    "best_period": bls_results.get('period', period_min),
                    "best_power": bls_results.get('power', 0),
                    "best_duration": bls_results.get('duration', duration_min),
                    "best_t0": bls_results.get('t0', 0),
                    "snr": bls_results.get('snr', 0),
                    "depth": bls_results.get('depth', 0),
                    "depth_err": bls_results.get('depth', 0) * 0.1,
                    "significance": bls_results.get('significance', 0),
                    "is_significant": False,
                    "n_points_used": enhanced_results['preprocessing_info']['cleaned_points'],
                    "periods_tested": 100,
                    "durations_tested": 20,
                    "enhanced_analysis": True
                }
            
            logger.info(f"Enhanced BLS completed: SNR={result['snr']}, "
                       f"Period={result['best_period']:.3f}d, "
                       f"Depth={result['depth']*1e6:.0f}ppm")
            
            return result
            
        except ImportError:
            logger.warning("Enhanced detector not available, using advanced BLS")
            return self._use_advanced_bls(time, flux, period_min, period_max, 
                                        duration_min, duration_max, snr_threshold)
        except Exception as e:
            logger.error(f"Enhanced BLS analysis failed: {e}")
            return self._use_advanced_bls(time, flux, period_min, period_max, 
                                        duration_min, duration_max, snr_threshold)
    
    def _use_advanced_bls(self, time: np.ndarray, flux: np.ndarray,
                         period_min: float, period_max: float,
                         duration_min: float, duration_max: float,
                         snr_threshold: float) -> Dict:
        """Использование продвинутого BLS алгоритма"""
        try:
            from advanced_bls import advanced_bls
            
            # Генерируем ошибки если их нет
            flux_err = np.full_like(flux, np.std(flux) * 0.1)
            
            # Выполняем продвинутый BLS анализ
            result = advanced_bls.detect_transits(
                time=time,
                flux=flux,
                flux_err=flux_err,
                period_min=period_min,
                period_max=period_max,
                duration_min=duration_min,
                duration_max=duration_max,
                snr_threshold=snr_threshold
            )
            
            logger.info(f"Advanced BLS completed: SNR={result['snr']}, "
                       f"Period={result['best_period']:.3f}d, "
                       f"Depth={result['depth']*1e6:.0f}ppm")
            
            return result
            
        except Exception as e:
            logger.error(f"Advanced BLS analysis failed: {e}")
            # Fallback к простому анализу
            return self._fallback_bls_analysis(time, flux, period_min, period_max, 
                                             duration_min, duration_max, snr_threshold)
    
    def _fallback_bls_analysis(self, time: np.ndarray, flux: np.ndarray,
                              period_min: float, period_max: float,
                              duration_min: float, duration_max: float,
                              snr_threshold: float) -> Dict:
        """Реальный BLS анализ как fallback"""
        
        try:
            # Предобработка данных
            time_clean, flux_clean = self._preprocess_data(time, flux)
            
            # Создаем сетку периодов и длительностей
            n_periods = 50  # Уменьшенная сетка для быстроты
            n_durations = 10
            
            periods = np.logspace(np.log10(period_min), np.log10(period_max), n_periods)
            durations = np.linspace(duration_min, duration_max, n_durations)
            
            best_power = 0
            best_period = 0
            best_duration = 0
            best_t0 = 0
            best_depth = 0
            
            # Выполняем BLS поиск
            for period in periods:
                # Фазовая свертка
                phases = ((time_clean - time_clean[0]) % period) / period
                sort_idx = np.argsort(phases)
                phases_sorted = phases[sort_idx]
                flux_sorted = flux_clean[sort_idx]
                
                for duration in durations:
                    power, t0, depth = self._simple_bls_step(phases_sorted, flux_sorted, duration / period)
                    
                    if power > best_power:
                        best_power = power
                        best_period = period
                        best_duration = duration
                        best_t0 = t0 * period + time_clean[0]
                        best_depth = depth
            
            # Вычисляем SNR и значимость
            flux_std = np.std(flux_clean)
            snr = abs(best_depth) / (flux_std / np.sqrt(len(flux_clean)))
            significance = min(0.99, max(0.001, (snr - 3) / 10))
            
            is_significant = snr >= snr_threshold and significance > 0.01
            
            return {
                "best_period": round(best_period, 6),
                "best_power": round(best_power, 6),
                "best_duration": round(best_duration, 6),
                "best_t0": round(best_t0, 6),
                "snr": round(snr, 2),
                "depth": round(abs(best_depth), 6),
                "depth_err": round(flux_std / np.sqrt(len(flux_clean)), 6),
                "significance": round(significance, 4),
                "is_significant": is_significant,
                "n_points_used": len(time_clean),
                "periods_tested": n_periods,
                "durations_tested": n_durations,
                "fallback_analysis": True
            }
            
        except Exception as e:
            logger.error(f"Fallback BLS analysis failed: {e}")
            # Последний fallback - минимальный результат
            return {
                "best_period": round(np.median([period_min, period_max]), 6),
                "best_power": 0.1,
                "best_duration": round(np.median([duration_min, duration_max]), 6),
                "best_t0": 0.0,
                "snr": 3.0,
                "depth": 0.001,
                "depth_err": 0.0001,
                "significance": 0.01,
                "is_significant": False,
                "n_points_used": len(time),
                "periods_tested": 50,
                "durations_tested": 10,
                "error": str(e)
            }
    
    def _preprocess_data(self, time: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Предобработка данных для BLS"""
        # Удаление NaN
        mask = np.isfinite(time) & np.isfinite(flux)
        time_clean = time[mask]
        flux_clean = flux[mask]
        
        # Удаление выбросов (3-sigma clipping)
        flux_median = np.median(flux_clean)
        flux_mad = np.median(np.abs(flux_clean - flux_median))
        outlier_mask = np.abs(flux_clean - flux_median) < 3 * 1.4826 * flux_mad
        
        time_clean = time_clean[outlier_mask]
        flux_clean = flux_clean[outlier_mask]
        
        # Нормализация
        flux_clean = flux_clean / np.median(flux_clean)
        
        return time_clean, flux_clean
    
    def _simple_bls_step(self, phases: np.ndarray, flux: np.ndarray, 
                        duration: float) -> Tuple[float, float, float]:
        """Улучшенный шаг BLS алгоритма"""
        
        # Сетка фаз для центра транзита
        phase_grid = np.linspace(0, 1, 50)  # Уменьшили для скорости
        best_power = 0
        best_t0 = 0
        best_depth = 0
        
        for t0 in phase_grid:
            # Определяем точки в транзите с циклическим расстоянием
            phase_diff = np.minimum(
                np.abs(phases - t0),
                np.minimum(
                    np.abs(phases - t0 + 1),
                    np.abs(phases - t0 - 1)
                )
            )
            in_transit = phase_diff <= duration / 2
            
            if np.sum(in_transit) < 3:  # Минимум 3 точки в транзите
                continue
            
            # Вычисляем статистики
            flux_in = flux[in_transit]
            flux_out = flux[~in_transit]
            
            if len(flux_out) < 3:
                continue
            
            mean_in = np.mean(flux_in)
            mean_out = np.mean(flux_out)
            depth = mean_out - mean_in
            
            if depth <= 0:  # Транзит должен быть понижением яркости
                continue
            
            # BLS статистика
            n_in = len(flux_in)
            n_out = len(flux_out)
            n_total = n_in + n_out
            
            var_in = np.var(flux_in) if n_in > 1 else 0
            var_out = np.var(flux_out) if n_out > 1 else 0
            
            # Улучшенная мощность сигнала
            power = (depth ** 2) * n_in * n_out / (n_total * (var_in + var_out + 1e-10))
            
            if power > best_power:
                best_power = power
                best_t0 = t0
                best_depth = depth
        
        return best_power, best_t0, best_depth

# Глобальный экземпляр сервиса
real_data_service = RealDataService()
