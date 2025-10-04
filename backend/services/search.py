"""
UNIFIED SEARCH SERVICE v8.0
Единый поисковый сервис с переключением между BLS и ENSEMBLE режимами
Максимально мощный и гибкий поиск экзопланет
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.logging import get_logger
from services.bls_service import BLSResult, BLSService
from services.ensemble_search_service import (
    EnsembleSearchResult,
    UltimateEnsembleSearchEngine,
)

logger = get_logger(__name__)


class SearchMode(Enum):
    """Режимы поиска"""

    BLS = "bls"
    ENSEMBLE = "ensemble"
    HYBRID = "hybrid"  # Комбинированный режим


@dataclass
class UnifiedSearchRequest:
    """Запрос для единого поиска"""

    target_name: str
    time: np.ndarray
    flux: np.ndarray
    flux_err: Optional[np.ndarray] = None

    # Параметры поиска
    search_mode: SearchMode = SearchMode.BLS
    period_min: float = 0.5
    period_max: float = 50.0
    snr_threshold: float = 7.0

    # Дополнительные параметры
    stellar_params: Optional[Dict] = None
    search_config: Optional[Dict] = None

    # Настройки производительности
    use_parallel: bool = True
    max_workers: int = 4


@dataclass
class UnifiedSearchResult:
    """Результат единого поиска"""

    target_name: str
    search_mode: SearchMode

    # Основные результаты
    best_period: float
    confidence: float
    snr: float
    depth: float
    significance: float

    # Результаты по режимам
    bls_result: Optional[BLSResult] = None
    ensemble_result: Optional[EnsembleSearchResult] = None

    # Сравнительный анализ
    mode_comparison: Optional[Dict] = None

    # Метаданные
    processing_time: float
    quality_score: float
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        result = {
            "target_name": self.target_name,
            "search_mode": self.search_mode.value,
            "best_period": float(self.best_period),
            "confidence": float(self.confidence),
            "snr": float(self.snr),
            "depth": float(self.depth),
            "significance": float(self.significance),
            "processing_time": float(self.processing_time),
            "quality_score": float(self.quality_score),
            "recommendations": self.recommendations,
        }

        # Добавляем результаты BLS
        if self.bls_result:
            result["bls_result"] = self.bls_result.to_dict()

        # Добавляем результаты Ensemble
        if self.ensemble_result:
            result["ensemble_result"] = {
                "consensus_period": self.ensemble_result.consensus_period,
                "consensus_confidence": self.ensemble_result.consensus_confidence,
                "consensus_depth": self.ensemble_result.consensus_depth,
                "consensus_snr": self.ensemble_result.consensus_snr,
                "method_agreement": self.ensemble_result.method_agreement,
                "ensemble_uncertainty": self.ensemble_result.ensemble_uncertainty,
                "planet_candidates": self.ensemble_result.planet_candidates,
                "methods_used": self.ensemble_result.methods_used,
                "habitability_score": self.ensemble_result.habitability_score,
                "false_positive_probability": self.ensemble_result.false_positive_probability,
            }

        # Добавляем сравнение режимов
        if self.mode_comparison:
            result["mode_comparison"] = self.mode_comparison

        return result


class UnifiedSearchService:
    """Единый поисковый сервис с переключением режимов"""

    def __init__(self):
        self.initialized = False

        # Инициализация сервисов
        self.bls_service = BLSService()
        self.ensemble_service = UltimateEnsembleSearchEngine()

        # Статистика использования
        self.usage_stats = {
            "total_searches": 0,
            "bls_searches": 0,
            "ensemble_searches": 0,
            "hybrid_searches": 0,
            "avg_processing_time": {"bls": 0.0, "ensemble": 0.0, "hybrid": 0.0},
            "success_rates": {"bls": 0.0, "ensemble": 0.0, "hybrid": 0.0},
        }

        # Кэш результатов
        self.result_cache = {}

        logger.info("🚀 Unified Search Service initialized")

    async def initialize(self):
        """Инициализация всех компонентов"""
        try:
            # Инициализация BLS сервиса
            await self.bls_service.initialize()
            logger.info("✅ BLS Service initialized")

            # Инициализация Ensemble сервиса
            await self.ensemble_service.initialize()
            logger.info("✅ Ensemble Service initialized")

            self.initialized = True
            logger.info("🎉 Unified Search Service READY!")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Unified Search Service: {e}")
            raise

    async def search(self, request: UnifiedSearchRequest) -> UnifiedSearchResult:
        """
        ГЛАВНЫЙ МЕТОД ПОИСКА
        Выполняет поиск в выбранном режиме
        """
        if not self.initialized:
            raise RuntimeError("Service not initialized")

        start_time = time.time()

        logger.info(f"🔍 UNIFIED SEARCH STARTING: {request.target_name}")
        logger.info(f"🎯 Mode: {request.search_mode.value.upper()}")
        logger.info(f"📊 Data: {len(request.time)} points")

        try:
            # Обновляем статистику
            self.usage_stats["total_searches"] += 1

            # Выбор режима поиска
            if request.search_mode == SearchMode.BLS:
                result = await self._run_bls_search(request)
                self.usage_stats["bls_searches"] += 1

            elif request.search_mode == SearchMode.ENSEMBLE:
                result = await self._run_ensemble_search(request)
                self.usage_stats["ensemble_searches"] += 1

            elif request.search_mode == SearchMode.HYBRID:
                result = await self._run_hybrid_search(request)
                self.usage_stats["hybrid_searches"] += 1

            else:
                raise ValueError(f"Unknown search mode: {request.search_mode}")

            # Финализация результата
            processing_time = time.time() - start_time
            result.processing_time = processing_time

            # Обновляем статистику производительности
            await self._update_performance_stats(
                request.search_mode, processing_time, result
            )

            # Генерируем рекомендации
            result.recommendations = await self._generate_recommendations(
                result, request
            )

            # Кэшируем результат
            cache_key = f"{request.target_name}_{request.search_mode.value}_{hash(str(request.time.tolist()[:10]))}"
            self.result_cache[cache_key] = result

            logger.info(
                f"✅ UNIFIED SEARCH COMPLETED in {processing_time:.2f}s: "
                f"P={result.best_period:.3f}d, SNR={result.snr:.1f}, "
                f"Conf={result.confidence:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"❌ UNIFIED SEARCH FAILED: {e}")
            raise

    async def _run_bls_search(
        self, request: UnifiedSearchRequest
    ) -> UnifiedSearchResult:
        """Запуск BLS поиска"""
        logger.info("🔍 Running BLS Search...")

        # Выполняем BLS анализ
        bls_result = await self.bls_service.analyze(
            time=request.time,
            flux=request.flux,
            flux_err=request.flux_err,
            period_min=request.period_min,
            period_max=request.period_max,
            snr_threshold=request.snr_threshold,
            target_name=request.target_name,
        )

        # Создаем единый результат
        unified_result = UnifiedSearchResult(
            target_name=request.target_name,
            search_mode=SearchMode.BLS,
            best_period=bls_result.best_period,
            confidence=bls_result.significance,
            snr=bls_result.snr,
            depth=bls_result.depth,
            significance=bls_result.significance,
            bls_result=bls_result,
            processing_time=0.0,  # Будет установлено позже
            quality_score=await self._calculate_quality_score(bls_result),
            recommendations=[],
        )

        logger.info(f"✅ BLS Search completed: P={bls_result.best_period:.3f}d")
        return unified_result

    async def _run_ensemble_search(
        self, request: UnifiedSearchRequest
    ) -> UnifiedSearchResult:
        """Запуск Ensemble поиска"""
        logger.info("🔥 Running ULTIMATE ENSEMBLE Search...")

        # Выполняем ensemble анализ
        ensemble_result = await self.ensemble_service.ultimate_search(
            time=request.time,
            flux=request.flux,
            flux_err=request.flux_err,
            target_name=request.target_name,
            stellar_params=request.stellar_params,
            search_config={
                "period_min": request.period_min,
                "period_max": request.period_max,
            },
        )

        # Создаем единый результат
        unified_result = UnifiedSearchResult(
            target_name=request.target_name,
            search_mode=SearchMode.ENSEMBLE,
            best_period=ensemble_result.consensus_period,
            confidence=ensemble_result.consensus_confidence,
            snr=ensemble_result.consensus_snr,
            depth=ensemble_result.consensus_depth,
            significance=ensemble_result.detection_significance,
            ensemble_result=ensemble_result,
            processing_time=0.0,  # Будет установлено позже
            quality_score=await self._calculate_ensemble_quality_score(ensemble_result),
            recommendations=[],
        )

        logger.info(
            f"✅ ENSEMBLE Search completed: P={ensemble_result.consensus_period:.3f}d"
        )
        return unified_result

    async def _run_hybrid_search(
        self, request: UnifiedSearchRequest
    ) -> UnifiedSearchResult:
        """Запуск гибридного поиска (BLS + Ensemble)"""
        logger.info("🎪 Running HYBRID Search (BLS + ENSEMBLE)...")

        # Запускаем оба метода параллельно
        if request.use_parallel:
            bls_task = asyncio.create_task(
                self._run_bls_search(
                    UnifiedSearchRequest(
                        target_name=request.target_name,
                        time=request.time,
                        flux=request.flux,
                        flux_err=request.flux_err,
                        search_mode=SearchMode.BLS,
                        period_min=request.period_min,
                        period_max=request.period_max,
                        snr_threshold=request.snr_threshold,
                    )
                )
            )

            ensemble_task = asyncio.create_task(
                self._run_ensemble_search(
                    UnifiedSearchRequest(
                        target_name=request.target_name,
                        time=request.time,
                        flux=request.flux,
                        flux_err=request.flux_err,
                        search_mode=SearchMode.ENSEMBLE,
                        period_min=request.period_min,
                        period_max=request.period_max,
                        stellar_params=request.stellar_params,
                        search_config=request.search_config,
                    )
                )
            )

            # Ждем завершения обоих
            bls_result, ensemble_result = await asyncio.gather(bls_task, ensemble_task)
        else:
            # Последовательное выполнение
            bls_result = await self._run_bls_search(request)
            ensemble_result = await self._run_ensemble_search(request)

        # Объединяем результаты
        hybrid_result = await self._combine_hybrid_results(
            bls_result, ensemble_result, request
        )

        logger.info(f"✅ HYBRID Search completed: P={hybrid_result.best_period:.3f}d")
        return hybrid_result

    async def _combine_hybrid_results(
        self,
        bls_result: UnifiedSearchResult,
        ensemble_result: UnifiedSearchResult,
        request: UnifiedSearchRequest,
    ) -> UnifiedSearchResult:
        """Объединение результатов BLS и Ensemble"""

        # Сравнительный анализ
        mode_comparison = await self._compare_results(bls_result, ensemble_result)

        # Выбираем лучший результат на основе качества и уверенности
        if mode_comparison["ensemble_advantage"] > 0.2:
            # Ensemble значительно лучше
            best_result = ensemble_result
            chosen_mode = "ensemble"
        elif mode_comparison["bls_advantage"] > 0.2:
            # BLS значительно лучше
            best_result = bls_result
            chosen_mode = "bls"
        else:
            # Примерно равны, выбираем по SNR
            if ensemble_result.snr > bls_result.snr:
                best_result = ensemble_result
                chosen_mode = "ensemble"
            else:
                best_result = bls_result
                chosen_mode = "bls"

        # Создаем гибридный результат
        hybrid_result = UnifiedSearchResult(
            target_name=request.target_name,
            search_mode=SearchMode.HYBRID,
            best_period=best_result.best_period,
            confidence=best_result.confidence,
            snr=best_result.snr,
            depth=best_result.depth,
            significance=best_result.significance,
            bls_result=bls_result.bls_result,
            ensemble_result=ensemble_result.ensemble_result,
            mode_comparison=mode_comparison,
            processing_time=0.0,  # Будет установлено позже
            quality_score=max(bls_result.quality_score, ensemble_result.quality_score),
            recommendations=[],
        )

        # Добавляем информацию о выбранном методе
        mode_comparison["chosen_method"] = chosen_mode
        mode_comparison["choice_reason"] = self._explain_method_choice(
            mode_comparison, chosen_mode
        )

        return hybrid_result

    async def _compare_results(
        self, bls_result: UnifiedSearchResult, ensemble_result: UnifiedSearchResult
    ) -> Dict:
        """Сравнение результатов BLS и Ensemble"""

        # Сравнение основных метрик
        period_diff = abs(bls_result.best_period - ensemble_result.best_period)
        period_agreement = 1.0 - (
            period_diff / max(bls_result.best_period, ensemble_result.best_period)
        )

        snr_ratio = ensemble_result.snr / bls_result.snr if bls_result.snr > 0 else 1.0
        confidence_ratio = (
            ensemble_result.confidence / bls_result.confidence
            if bls_result.confidence > 0
            else 1.0
        )

        # Преимущества каждого метода
        ensemble_advantage = (
            (snr_ratio - 1.0) * 0.4
            + (confidence_ratio - 1.0) * 0.3
            + (ensemble_result.quality_score - bls_result.quality_score) * 0.3
        )

        bls_advantage = -ensemble_advantage  # Инвертируем

        return {
            "period_agreement": float(period_agreement),
            "snr_ratio": float(snr_ratio),
            "confidence_ratio": float(confidence_ratio),
            "ensemble_advantage": float(ensemble_advantage),
            "bls_advantage": float(bls_advantage),
            "bls_metrics": {
                "period": float(bls_result.best_period),
                "snr": float(bls_result.snr),
                "confidence": float(bls_result.confidence),
                "quality": float(bls_result.quality_score),
            },
            "ensemble_metrics": {
                "period": float(ensemble_result.best_period),
                "snr": float(ensemble_result.snr),
                "confidence": float(ensemble_result.confidence),
                "quality": float(ensemble_result.quality_score),
                "methods_used": (
                    len(ensemble_result.ensemble_result.methods_used)
                    if ensemble_result.ensemble_result
                    else 0
                ),
                "method_agreement": (
                    float(ensemble_result.ensemble_result.method_agreement)
                    if ensemble_result.ensemble_result
                    else 0.0
                ),
            },
        }

    def _explain_method_choice(self, comparison: Dict, chosen_method: str) -> str:
        """Объяснение выбора метода"""
        if chosen_method == "ensemble":
            if comparison["ensemble_advantage"] > 0.3:
                return "Ensemble показал значительно лучшие результаты по SNR и уверенности"
            elif comparison["snr_ratio"] > 1.5:
                return "Ensemble обнаружил более сильный сигнал"
            else:
                return "Ensemble показал немного лучшие результаты"
        else:
            if comparison["bls_advantage"] > 0.3:
                return "BLS показал более надежные и стабильные результаты"
            elif comparison["snr_ratio"] < 0.7:
                return "BLS обнаружил более четкий сигнал"
            else:
                return "BLS показал сопоставимые результаты с меньшими вычислительными затратами"

    async def _calculate_quality_score(self, bls_result: BLSResult) -> float:
        """Расчет качества для BLS результата"""
        quality_factors = []

        # SNR фактор
        snr_factor = min(1.0, bls_result.snr / 15.0)
        quality_factors.append(snr_factor * 0.4)

        # Significance фактор
        sig_factor = bls_result.significance
        quality_factors.append(sig_factor * 0.3)

        # False alarm probability фактор
        fap_factor = 1.0 - bls_result.false_alarm_probability
        quality_factors.append(fap_factor * 0.2)

        # Depth фактор
        depth_factor = min(1.0, bls_result.depth * 1000)  # Нормализуем глубину
        quality_factors.append(depth_factor * 0.1)

        return sum(quality_factors)

    async def _calculate_ensemble_quality_score(
        self, ensemble_result: EnsembleSearchResult
    ) -> float:
        """Расчет качества для Ensemble результата"""
        quality_factors = []

        # SNR фактор
        snr_factor = min(1.0, ensemble_result.consensus_snr / 15.0)
        quality_factors.append(snr_factor * 0.3)

        # Confidence фактор
        conf_factor = ensemble_result.consensus_confidence
        quality_factors.append(conf_factor * 0.25)

        # Method agreement фактор
        agreement_factor = ensemble_result.method_agreement
        quality_factors.append(agreement_factor * 0.2)

        # Bootstrap confidence фактор
        bootstrap_factor = ensemble_result.bootstrap_confidence
        quality_factors.append(bootstrap_factor * 0.15)

        # False positive фактор
        fp_factor = 1.0 - ensemble_result.false_positive_probability
        quality_factors.append(fp_factor * 0.1)

        return sum(quality_factors)

    async def _generate_recommendations(
        self, result: UnifiedSearchResult, request: UnifiedSearchRequest
    ) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []

        # Рекомендации по качеству сигнала
        if result.snr < 7:
            recommendations.append(
                "Низкий SNR - рекомендуется дополнительная валидация"
            )
        elif result.snr > 15:
            recommendations.append("Высокий SNR - сильный кандидат для подтверждения")

        # Рекомендации по режиму поиска
        if request.search_mode == SearchMode.BLS and result.confidence < 0.7:
            recommendations.append("Рекомендуется проверить с помощью Ensemble поиска")
        elif request.search_mode == SearchMode.ENSEMBLE and result.ensemble_result:
            if result.ensemble_result.method_agreement < 0.6:
                recommendations.append(
                    "Низкое согласие между методами - требуется осторожность"
                )

        # Рекомендации по периоду
        if result.best_period < 1.0:
            recommendations.append("Короткий период - проверьте на алиасы")
        elif result.best_period > 30.0:
            recommendations.append(
                "Длинный период - требуется больше данных для подтверждения"
            )

        # Рекомендации по глубине
        if result.depth < 0.001:
            recommendations.append("Малая глубина транзита - возможен ложный сигнал")
        elif result.depth > 0.1:
            recommendations.append(
                "Глубокий транзит - проверьте на затмевающие двойные"
            )

        return recommendations

    async def _update_performance_stats(
        self, mode: SearchMode, processing_time: float, result: UnifiedSearchResult
    ):
        """Обновление статистики производительности"""
        mode_str = mode.value

        # Обновляем среднее время обработки
        current_avg = self.usage_stats["avg_processing_time"][mode_str]
        count = self.usage_stats[f"{mode_str}_searches"]

        if count > 1:
            new_avg = ((current_avg * (count - 1)) + processing_time) / count
        else:
            new_avg = processing_time

        self.usage_stats["avg_processing_time"][mode_str] = new_avg

        # Обновляем успешность (на основе качества)
        success = 1.0 if result.quality_score > 0.5 else 0.0
        current_success = self.usage_stats["success_rates"][mode_str]

        if count > 1:
            new_success = ((current_success * (count - 1)) + success) / count
        else:
            new_success = success

        self.usage_stats["success_rates"][mode_str] = new_success

    async def get_service_status(self) -> Dict:
        """Получение статуса сервиса"""
        return {
            "initialized": self.initialized,
            "bls_service_status": await self.bls_service.get_status(),
            "ensemble_service_available": self.ensemble_service.initialized,
            "usage_statistics": self.usage_stats,
            "cache_size": len(self.result_cache),
        }

    async def cleanup(self):
        """Очистка ресурсов"""
        try:
            await self.bls_service.cleanup()
            # Ensemble service cleanup если есть метод
            if hasattr(self.ensemble_service, "cleanup"):
                await self.ensemble_service.cleanup()

            logger.info("✅ Unified Search Service cleaned up")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")


# Глобальный экземпляр
unified_search_service = UnifiedSearchService()
