#!/usr/bin/env python3
"""
Advanced GPI Service - Optimized Gravitational Phase Interferometry Service
Enhanced with caching, async optimization, and performance monitoring

Provides high-performance GPI analysis functionality with:
- Async/await optimization
- Result caching
- Connection pooling
- Performance metrics
- Error handling with retry logic
"""

import hashlib
import logging
import threading
import weakref
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Dict, List, Optional, Any

import numpy as np

# GPI AI Model не доступен - используем базовый анализ
ML_AVAILABLE = False
GPIAIModel = None

# Базовые параметры GPI
class GPIParameters:
    def __init__(self):
        self.sensitivity = 1.0
        self.frequency_range = (0.1, 10.0)

class GPIEngine:
    def __init__(self):
        pass
    
    def analyze(self, time, flux):
        # Базовый GPI анализ
        return {
            'period': np.median(np.diff(time)) * len(time),
            'depth': np.std(flux),
            'significance': 0.5
        }

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Метрики производительности"""

    total_analyses: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0

    def update_success(self, processing_time: float):
        """Обновить метрики успешного анализа"""
        self.total_analyses += 1
        self.successful_analyses += 1
        self.total_processing_time += processing_time
        self.average_processing_time = self.total_processing_time / self.total_analyses

    def update_failure(self):
        """Обновить метрики неудачного анализа"""
        self.total_analyses += 1
        self.failed_analyses += 1

    def cache_hit(self):
        """Зарегистрировать попадание в кэш"""
        self.cache_hits += 1

    def cache_miss(self):
        """Зарегистрировать промах кэша"""
        self.cache_misses += 1

    @property
    def success_rate(self) -> float:
        """Коэффициент успешности"""
        return (
            self.successful_analyses / self.total_analyses
            if self.total_analyses > 0
            else 0.0
        )

    @property
    def cache_hit_rate(self) -> float:
        """Коэффициент попаданий в кэш"""
        total_cache_requests = self.cache_hits + self.cache_misses
        return (
            self.cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0
        )


class AsyncCache:
    """Асинхронный кэш для результатов GPI"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._access_times = {}
        self._lock = threading.Lock()

    def _generate_key(
        self, target_data: Dict, custom_params: Optional[Dict] = None
    ) -> str:
        """Генерация ключа кэша"""
        key_data = {
            "target_name": target_data.get("target_name", ""),
            "data_length": len(target_data.get("time", [])),
            "data_hash": hashlib.sha256(
                str(target_data.get("time", [])[:10]).encode()
            ).hexdigest()[:16],
            "params": custom_params or {},
        }
        return hashlib.sha256(str(sorted(key_data.items())).encode()).hexdigest()

    async def get(
        self, target_data: Dict, custom_params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Получить результат из кэша"""
        key = self._generate_key(target_data, custom_params)

        with self._lock:
            if key not in self._cache:
                return None

            # Проверяем TTL
            if time.time() - self._access_times[key] > self.ttl_seconds:
                del self._cache[key]
                del self._access_times[key]
                return None

            self._access_times[key] = time.time()
            return self._cache[key].copy()

    async def set(
        self, target_data: Dict, result: Dict, custom_params: Optional[Dict] = None
    ):
        """Сохранить результат в кэш"""
        key = self._generate_key(target_data, custom_params)

        with self._lock:
            # Очищаем кэш если переполнен
            if len(self._cache) >= self.max_size:
                oldest_key = min(
                    self._access_times.keys(), key=lambda k: self._access_times[k]
                )
                del self._cache[oldest_key]
                del self._access_times[oldest_key]

            self._cache[key] = result.copy()
            self._access_times[key] = time.time()

    def clear(self):
        """Очистить кэш"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику кэша"""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "oldest_entry_age": (
                    time.time() - min(self._access_times.values())
                    if self._access_times
                    else 0
                ),
            }


def async_retry(max_retries: int = 3, delay: float = 1.0):
    """Декоратор для повторных попыток асинхронных функций"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay * (2**attempt))  # Exponential backoff
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}"
                        )

            raise last_exception

        return wrapper

    return decorator


class GPIService:
    """
    Advanced Service for Gravitational Phase Interferometry analysis.

    Enhanced with:
    - Async/await optimization
    - Result caching with TTL
    - Connection pooling
    - Performance monitoring
    - Retry logic with exponential backoff
    - Resource management
    """

    def __init__(
        self,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        thread_pool_size: int = 4,
        process_pool_size: int = 2,
    ):
        """Initialize advanced GPI service."""

        # Core components
        self.gpi_engine = GPIEngine()
        self.ai_model = None
        self.is_initialized = False

        # Performance optimization components
        self.cache = AsyncCache(max_size=cache_size, ttl_seconds=cache_ttl)
        self.metrics = PerformanceMetrics()
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        self.process_pool = ProcessPoolExecutor(max_workers=process_pool_size)

        # Async locks for thread safety
        self._analysis_lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()

        # Active tasks tracking
        self._active_tasks = weakref.WeakSet()

        # Initialize AI model if available
        if ML_AVAILABLE:
            try:
                self.ai_model = GPIAIModel()
                # Try to load pre-trained model
                model_path = (
                    Path(__file__).parent.parent / "models" / "gpi_ai_model.pkl"
                )
                if model_path.exists():
                    self.ai_model.load_model(str(model_path))
                    logger.info("Pre-trained GPI AI model loaded")
            except Exception as e:
                logger.warning(f"Could not initialize GPI AI model: {e}")
                self.ai_model = None

        self.is_initialized = True
        logger.info(
            f"Advanced GPI Service initialized: cache_size={cache_size}, "
            f"thread_pool={thread_pool_size}, process_pool={process_pool_size}"
        )

    @async_retry(max_retries=2, delay=0.5)
    async def analyze_target(
        self,
        target_data: Dict,
        use_ai: bool = True,
        custom_params: Optional[Dict] = None,
        use_cache: bool = True,
    ) -> Dict:
        """
        Advanced analyze target with caching, retry logic and performance monitoring.

        Args:
            target_data: Dictionary containing target information and lightcurve data
            use_ai: Whether to use AI enhancement
            custom_params: Custom GPI parameters
            use_cache: Whether to use result caching

        Returns:
            GPI analysis result with performance metrics
        """
        start_time = time.time()
        target_name = target_data.get("target_name", "Unknown")

        try:
            # Check cache first
            if use_cache:
                cached_result = await self.cache.get(target_data, custom_params)
                if cached_result:
                    self.metrics.cache_hit()
                    logger.debug(f"Cache hit for GPI analysis of {target_name}")

                    # Update cached result timestamp
                    cached_result["service_info"]["cache_hit"] = True
                    cached_result["service_info"][
                        "analysis_time"
                    ] = datetime.now().isoformat()
                    return cached_result
                else:
                    self.metrics.cache_miss()

            logger.info(f"Starting advanced GPI analysis for {target_name}")

            # Extract and validate lightcurve data
            time_array = np.array(target_data.get("time", []))
            flux_array = np.array(target_data.get("flux", []))
            flux_err_array = np.array(target_data.get("flux_err", []))

            if len(time_array) == 0 or len(flux_array) == 0:
                raise ValueError("No lightcurve data provided")

            # Parallel validation and preprocessing
            validation_task = asyncio.create_task(
                self._validate_lightcurve_data_async(
                    time_array, flux_array, flux_err_array
                )
            )

            # Run GPI analysis with resource pooling
            async with self._analysis_lock:
                gpi_result = await self._run_gpi_analysis_optimized(
                    time_array, flux_array, flux_err_array, target_name, custom_params
                )

            # Wait for validation to complete
            validation_result = await validation_task
            if not validation_result["is_valid"]:
                logger.warning(
                    f"Data validation issues for {target_name}: {validation_result['issues']}"
                )
                gpi_result["validation_warnings"] = validation_result["issues"]

            # Enhance with AI if available and requested
            if use_ai and self.ai_model and self.ai_model.is_trained:
                ai_result = await self._run_ai_analysis_optimized(target_data)
                gpi_result = self._combine_gpi_ai_results(gpi_result, ai_result)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # Add enhanced service metadata
            gpi_result["service_info"] = {
                "service": "Advanced GPI Service",
                "version": "2.0",
                "ai_enhanced": use_ai and self.ai_model is not None,
                "analysis_time": datetime.now().isoformat(),
                "processing_time_ms": processing_time,
                "cache_hit": False,
                "data_points": len(time_array),
                "validation_passed": validation_result["is_valid"],
            }

            # Update metrics
            self.metrics.update_success(processing_time)

            # Cache result if requested
            if use_cache:
                await self.cache.set(target_data, gpi_result, custom_params)

            logger.info(
                f"Advanced GPI analysis completed for {target_name} in {processing_time:.1f}ms"
            )
            return gpi_result

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.metrics.update_failure()

            logger.error(f"Advanced GPI analysis failed for {target_name}: {e}")
            return self._error_result(target_name, str(e), processing_time)

    async def _run_gpi_analysis_optimized(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray],
        target_name: str,
        custom_params: Optional[Dict],
    ) -> Dict:
        """Run optimized GPI analysis with resource pooling."""
        loop = asyncio.get_event_loop()

        # Choose appropriate executor based on data size
        data_size = len(time)

        if data_size > 10000:  # Large dataset - use process pool
            task = loop.run_in_executor(
                self.process_pool,
                self.gpi_engine.analyze_lightcurve,
                time,
                flux,
                flux_err,
                target_name,
                custom_params,
            )
        else:  # Small dataset - use thread pool
            task = loop.run_in_executor(
                self.thread_pool,
                self.gpi_engine.analyze_lightcurve,
                time,
                flux,
                flux_err,
                target_name,
                custom_params,
            )

        # Track active task
        self._active_tasks.add(task)

        try:
            result = await task
            return result
        finally:
            # Task completed, remove from tracking
            self._active_tasks.discard(task)

    async def _run_ai_analysis_optimized(self, target_data: Dict) -> Dict:
        """Run optimized AI analysis with resource pooling."""
        if not self.ai_model or not self.ai_model.is_trained:
            return {}

        loop = asyncio.get_event_loop()

        # Run AI analysis in thread pool with lock
        async with self._model_lock:
            try:
                task = loop.run_in_executor(
                    self.thread_pool, self.ai_model.predict, target_data
                )
                self._active_tasks.add(task)
                result = await task
                return result
            except Exception as e:
                logger.warning(f"Optimized AI analysis failed: {e}")
                return {}
            finally:
                self._active_tasks.discard(task)

    async def _validate_lightcurve_data_async(
        self, time: np.ndarray, flux: np.ndarray, flux_err: Optional[np.ndarray]
    ) -> Dict:
        """Asynchronous lightcurve data validation with enhanced checks."""
        loop = asyncio.get_event_loop()

        def _validate():
            issues = []

            # Enhanced validation checks
            if len(time) < 50:
                issues.append(
                    "Too few data points (minimum 50 required for reliable GPI)"
                )

            if len(time) != len(flux):
                issues.append("Time and flux arrays have different lengths")

            if flux_err is not None and len(flux_err) != len(flux):
                issues.append("Flux error array has different length than flux")

            # Check for NaN or infinite values
            if np.any(~np.isfinite(time)):
                issues.append("Time array contains NaN or infinite values")

            if np.any(~np.isfinite(flux)):
                issues.append("Flux array contains NaN or infinite values")

            # Check time ordering and gaps
            if not np.all(np.diff(time) > 0):
                issues.append("Time array is not monotonically increasing")

            # Check for large time gaps
            time_diffs = np.diff(time)
            median_cadence = np.median(time_diffs)
            large_gaps = np.sum(time_diffs > 5 * median_cadence)
            if large_gaps > len(time) * 0.1:
                issues.append(f"Too many large time gaps: {large_gaps}")

            # Check flux variability and outliers
            if np.std(flux) == 0:
                issues.append("Flux shows no variability")

            # Check for excessive outliers
            flux_median = np.median(flux)
            flux_mad = np.median(np.abs(flux - flux_median))
            outliers = np.sum(np.abs(flux - flux_median) > 5 * flux_mad)
            if outliers > len(flux) * 0.05:
                issues.append(f"Too many flux outliers: {outliers}")

            # Check observation duration for GPI effectiveness
            duration = np.max(time) - np.min(time)
            if duration < 5:  # Less than 5 days
                issues.append(
                    f"Observation duration too short for GPI: {duration:.1f} days"
                )

            return {
                "is_valid": len(issues) == 0,
                "issues": issues,
                "data_points": len(time),
                "duration_days": float(duration) if len(time) > 0 else 0.0,
                "flux_std": float(np.std(flux)) if len(flux) > 0 else 0.0,
                "cadence_minutes": (
                    float(median_cadence * 24 * 60) if len(time) > 1 else 0.0
                ),
                "outlier_fraction": outliers / len(flux) if len(flux) > 0 else 0.0,
            }

        # Run validation in thread pool for CPU-intensive operations
        return await loop.run_in_executor(self.thread_pool, _validate)

    async def _run_ai_analysis(self, target_data: Dict) -> Dict:
        """Run AI analysis in async context."""
        if not self.ai_model or not self.ai_model.is_trained:
            return {}

        loop = asyncio.get_event_loop()

        # Run AI analysis in thread pool
        try:
            result = await loop.run_in_executor(
                None, self.ai_model.predict, target_data
            )
            return result
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            return {}

    def _combine_gpi_ai_results(self, gpi_result: Dict, ai_result: Dict) -> Dict:
        """Combine GPI and AI analysis results."""
        if not ai_result:
            return gpi_result

        # Add AI predictions to the result
        gpi_result["ai_analysis"] = {
            "ai_prediction": ai_result.get("prediction", 0),
            "ai_confidence": ai_result.get("confidence", 0.0),
            "probability_with_planets": ai_result.get("probability_with_planets", 0.0),
            "ai_method": ai_result.get("ai_method", "Unknown"),
            "model_version": ai_result.get("model_version", "Unknown"),
        }

        # Combine confidence scores
        gpi_confidence = gpi_result.get("summary", {}).get("detection_confidence", 0.0)
        ai_confidence = ai_result.get("confidence", 0.0)

        # Weighted combination (GPI has higher weight as it's the primary method)
        combined_confidence = 0.7 * gpi_confidence + 0.3 * ai_confidence

        gpi_result["summary"]["combined_confidence"] = float(combined_confidence)
        gpi_result["summary"]["ai_enhanced"] = True

        return gpi_result

    def _validate_lightcurve_data(
        self, time: np.ndarray, flux: np.ndarray, flux_err: Optional[np.ndarray]
    ) -> Dict:
        """Validate lightcurve data quality."""
        issues = []

        # Check basic requirements
        if len(time) < 100:
            issues.append("Too few data points (minimum 100 required)")

        if len(time) != len(flux):
            issues.append("Time and flux arrays have different lengths")

        if flux_err is not None and len(flux_err) != len(flux):
            issues.append("Flux error array has different length than flux")

        # Check for NaN or infinite values
        if np.any(~np.isfinite(time)):
            issues.append("Time array contains NaN or infinite values")

        if np.any(~np.isfinite(flux)):
            issues.append("Flux array contains NaN or infinite values")

        # Check time ordering
        if not np.all(np.diff(time) > 0):
            issues.append("Time array is not monotonically increasing")

        # Check flux variability
        if np.std(flux) == 0:
            issues.append("Flux shows no variability")

        # Check observation duration
        duration = np.max(time) - np.min(time)
        if duration < 10:  # Less than 10 days
            issues.append(f"Observation duration too short: {duration:.1f} days")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "data_points": len(time),
            "duration_days": float(duration) if len(time) > 0 else 0.0,
            "flux_std": float(np.std(flux)) if len(flux) > 0 else 0.0,
        }

    def get_gpi_parameters(self) -> Dict:
        """Get current GPI parameters."""
        params = self.gpi_engine.params
        return {
            "phase_sensitivity": params.phase_sensitivity,
            "snr_threshold": params.snr_threshold,
            "min_period_days": params.min_period_days,
            "max_period_days": params.max_period_days,
            "min_orbital_cycles": params.min_orbital_cycles,
            "use_numba_acceleration": params.use_numba_acceleration,
            "use_parallel_processing": params.use_parallel_processing,
        }

    def update_gpi_parameters(self, new_params: Dict) -> bool:
        """Update GPI parameters."""
        try:
            for key, value in new_params.items():
                if hasattr(self.gpi_engine.params, key):
                    setattr(self.gpi_engine.params, key, value)
                    logger.info(f"Updated GPI parameter {key} = {value}")
                else:
                    logger.warning(f"Unknown GPI parameter: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to update GPI parameters: {e}")
            return False

    def get_service_status(self) -> Dict:
        """Get enhanced service status with performance metrics."""
        return {
            "service": "Advanced GPI Service",
            "version": "2.0",
            "initialized": self.is_initialized,
            "gpi_engine_available": self.gpi_engine is not None,
            "ai_model_available": self.ai_model is not None,
            "ai_model_trained": self.ai_model.is_trained if self.ai_model else False,
            "ml_libraries_available": ML_AVAILABLE,
            "parameters": self.get_gpi_parameters(),
            "performance_metrics": {
                "total_analyses": self.metrics.total_analyses,
                "successful_analyses": self.metrics.successful_analyses,
                "failed_analyses": self.metrics.failed_analyses,
                "success_rate": self.metrics.success_rate,
                "average_processing_time_ms": self.metrics.average_processing_time,
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "active_tasks": len(self._active_tasks),
            },
            "cache_stats": self.cache.get_stats(),
            "resource_pools": {
                "thread_pool_size": self.thread_pool._max_workers,
                "process_pool_size": self.process_pool._max_workers,
                "active_tasks_count": len(self._active_tasks),
            },
        }

    async def get_performance_metrics(self) -> Dict:
        """Get detailed performance metrics."""
        return {
            "analysis_metrics": {
                "total_analyses": self.metrics.total_analyses,
                "successful_analyses": self.metrics.successful_analyses,
                "failed_analyses": self.metrics.failed_analyses,
                "success_rate": self.metrics.success_rate,
                "average_processing_time_ms": self.metrics.average_processing_time,
                "total_processing_time_ms": self.metrics.total_processing_time,
            },
            "cache_metrics": {
                "cache_hits": self.metrics.cache_hits,
                "cache_misses": self.metrics.cache_misses,
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "cache_stats": self.cache.get_stats(),
            },
            "resource_metrics": {
                "active_tasks": len(self._active_tasks),
                "thread_pool_size": self.thread_pool._max_workers,
                "process_pool_size": self.process_pool._max_workers,
            },
            "timestamp": datetime.now().isoformat(),
        }

    async def clear_cache(self) -> Dict:
        """Clear the analysis cache."""
        self.cache.clear()
        return {
            "status": "success",
            "message": "GPI analysis cache cleared",
            "timestamp": datetime.now().isoformat(),
        }

    async def shutdown(self):
        """Gracefully shutdown the service."""
        logger.info("Shutting down Advanced GPI Service...")

        # Wait for active tasks to complete (with timeout)
        if self._active_tasks:
            logger.info(
                f"Waiting for {len(self._active_tasks)} active tasks to complete..."
            )
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._active_tasks, return_exceptions=True),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Some tasks did not complete within timeout")

        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

        # Clear cache
        self.cache.clear()

        logger.info("Advanced GPI Service shutdown completed")

    def test_gpi_system(self) -> Dict:
        """Test GPI system functionality."""
        try:
            # Test basic GPI engine
            gpi_test = self.gpi_engine.test_system()

            # Test AI model if available
            ai_test = False
            if self.ai_model and self.ai_model.is_trained:
                try:
                    # Create deterministic test data
                    time_array = np.linspace(0, 100, 1000)
                    # Use deterministic noise based on time array
                    deterministic_noise = (
                        0.001 * np.sin(time_array * 0.1) * np.cos(time_array * 0.05)
                    )
                    test_data = {
                        "time": time_array,
                        "flux": np.ones(1000) + deterministic_noise,
                    }
                    result = self.ai_model.predict(test_data)
                    ai_test = True
                except Exception as e:
                    logger.warning(f"AI model test failed: {e}")

            return {
                "gpi_engine_test": gpi_test,
                "ai_model_test": ai_test,
                "overall_status": gpi_test and (ai_test or not self.ai_model),
                "test_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"GPI system test failed: {e}")
            return {
                "gpi_engine_test": False,
                "ai_model_test": False,
                "overall_status": False,
                "error": str(e),
                "test_timestamp": datetime.now().isoformat(),
            }

    def _error_result(self, target_name: str, error_message: str) -> Dict:
        """Create error result."""
        return {
            "summary": {
                "target_name": target_name,
                "method": "Gravitational Phase Interferometry (GPI)",
                "exoplanet_detected": False,
                "detection_confidence": 0.0,
                "error": error_message,
                "analysis_timestamp": datetime.now().isoformat(),
            },
            "gpi_analysis": {
                "phase_shift_amplitude_rad": 0.0,
                "orbital_period_days": 0.0,
                "gravitational_signature_strength": 0.0,
                "phase_coherence": 0.0,
                "snr_estimate": 0.0,
            },
            "service_info": {
                "service": "GPI Service",
                "version": "1.0",
                "error": True,
                "analysis_time": datetime.now().isoformat(),
            },
        }


# Global service instance
_gpi_service_instance = None


def get_gpi_service() -> GPIService:
    """Get global GPI service instance."""
    global _gpi_service_instance
    if _gpi_service_instance is None:
        _gpi_service_instance = GPIService()
    return _gpi_service_instance
