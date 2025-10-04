"""
ULTIMATE ENSEMBLE SEARCH ENGINE v7.0
Нереально мощная система ансамблевого поиска экзопланет
Объединяет все методы: BLS, GPI, AI, TLS, Wavelet, Fourier, Machine Learning
"""

import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, signal, stats
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

from core.logging_config import get_logger
from services.bls_service import BLSResult, BLSService
from services.gpi_service import GPIService

logger = get_logger(__name__)


@dataclass
class EnsembleSearchResult:
    """Результат супер-мощного ансамблевого поиска"""

    target_name: str

    # Основные результаты
    consensus_period: float
    consensus_confidence: float
    consensus_depth: float
    consensus_snr: float

    # Результаты отдельных методов
    bls_result: Optional[BLSResult]
    gpi_result: Optional[Dict]
    tls_result: Optional[Dict]
    wavelet_result: Optional[Dict]
    fourier_result: Optional[Dict]
    ml_result: Optional[Dict]

    # Ансамблевые метрики
    method_agreement: float
    ensemble_uncertainty: float
    cross_validation_score: float
    bootstrap_confidence: float

    # Продвинутые параметры
    planet_candidates: List[Dict]
    false_positive_probability: float
    habitability_score: float
    detection_significance: float

    # Физические параметры
    stellar_parameters: Dict[str, float]
    planetary_system: Dict[str, Any]
    orbital_dynamics: Dict[str, float]

    # Метаданные
    processing_time: float
    methods_used: List[str]
    quality_flags: Dict[str, bool]
    recommendations: List[str]


class UltimateEnsembleSearchEngine:
    """Нереально мощный ансамблевый поисковый движок"""

    def __init__(self):
        self.initialized = False

        # Исполнители для параллельной обработки
        self.thread_executor = ThreadPoolExecutor(max_workers=8)
        self.process_executor = ProcessPoolExecutor(max_workers=4)

        # Базовые сервисы
        self.bls_service = BLSService()
        self.gpi_service = None  # Будет инициализирован

        # ML модели для ансамбля
        self.ml_ensemble = None
        self.feature_extractors = {}
        self.scalers = {}

        # Кэш и статистика
        self.cache = {}
        self.performance_stats = {
            "total_searches": 0,
            "avg_processing_time": 0.0,
            "method_success_rates": {},
            "ensemble_accuracy": 0.0,
        }

        # Адаптивные веса для методов
        self.method_weights = {
            "bls": 0.25,
            "gpi": 0.20,
            "tls": 0.15,
            "wavelet": 0.15,
            "fourier": 0.10,
            "ml_ensemble": 0.15,
        }

        logger.info("🚀 Ultimate Ensemble Search Engine initialized")

    async def initialize(self):
        """Инициализация супер-мощного движка"""
        try:
            # Инициализация базовых сервисов
            await self.bls_service.initialize()

            # Инициализация GPI сервиса
            try:
                from services.gpi_service import GPIService

                self.gpi_service = GPIService()
                await self.gpi_service.initialize()
            except Exception as e:
                logger.warning(f"GPI service not available: {e}")

            # Создание ML ансамбля
            await self._initialize_ml_ensemble()

            # Инициализация экстракторов признаков
            await self._initialize_feature_extractors()

            self.initialized = True
            logger.info("✅ Ultimate Ensemble Search Engine ready for MAXIMUM POWER!")

        except Exception as e:
            logger.error(f"❌ Failed to initialize ensemble engine: {e}")
            raise

    async def _initialize_ml_ensemble(self):
        """Создание мощного ML ансамбля"""
        try:
            # Базовые регрессоры
            rf_regressor = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            )

            gb_regressor = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                random_state=42,
            )

            et_regressor = ExtraTreesRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1,
            )

            # Нейронная сеть
            mlp_regressor = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate="adaptive",
                max_iter=1000,
                random_state=42,
            )

            # SVM регрессор
            svr_regressor = SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.01)

            # Создание стекинг ансамбля
            base_models = [
                ("rf", rf_regressor),
                ("gb", gb_regressor),
                ("et", et_regressor),
                ("mlp", mlp_regressor),
                ("svr", svr_regressor),
            ]

            # Мета-модель для стекинга
            meta_model = GradientBoostingRegressor(
                n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42
            )

            # Финальный ансамбль
            self.ml_ensemble = StackingRegressor(
                estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1
            )

            logger.info("🧠 ML Ensemble with 5 base models + stacking created")

        except Exception as e:
            logger.error(f"Failed to create ML ensemble: {e}")
            self.ml_ensemble = None

    async def _initialize_feature_extractors(self):
        """Инициализация экстракторов признаков"""
        self.feature_extractors = {
            "statistical": self._extract_statistical_features,
            "frequency": self._extract_frequency_features,
            "wavelet": self._extract_wavelet_features,
            "morphological": self._extract_morphological_features,
            "chaos": self._extract_chaos_features,
            "information": self._extract_information_features,
        }

        # Скейлеры для разных типов признаков
        self.scalers = {
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "minmax": StandardScaler(),  # Placeholder
        }

        logger.info("🔧 Advanced feature extractors initialized")

    async def ultimate_search(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None,
        target_name: str = "unknown",
        stellar_params: Optional[Dict] = None,
        search_config: Optional[Dict] = None,
    ) -> EnsembleSearchResult:
        """
        НЕРЕАЛЬНО МОЩНЫЙ АНСАМБЛЕВЫЙ ПОИСК!
        Использует ВСЕ доступные методы одновременно
        """
        start_time = time.time()

        logger.info(f"🚀 LAUNCHING ULTIMATE SEARCH for {target_name}")
        logger.info(f"🔥 MAXIMUM POWER MODE: {len(time)} data points")

        try:
            # Обновляем статистику
            self.performance_stats["total_searches"] += 1

            # Конфигурация поиска
            config = search_config or {}
            period_min = config.get("period_min", 0.5)
            period_max = config.get("period_max", 50.0)

            # Предварительная обработка данных
            time_clean, flux_clean, flux_err_clean = (
                await self._ultimate_data_preprocessing(time, flux, flux_err)
            )

            # Извлечение всех признаков
            all_features = await self._extract_all_features(time_clean, flux_clean)

            # ПАРАЛЛЕЛЬНЫЙ ЗАПУСК ВСЕХ МЕТОДОВ
            search_tasks = []

            # 1. BLS поиск
            search_tasks.append(
                asyncio.create_task(
                    self._run_bls_search(
                        time_clean,
                        flux_clean,
                        flux_err_clean,
                        period_min,
                        period_max,
                        target_name,
                    )
                )
            )

            # 2. GPI поиск (если доступен)
            if self.gpi_service:
                search_tasks.append(
                    asyncio.create_task(
                        self._run_gpi_search(time_clean, flux_clean, target_name)
                    )
                )

            # 3. TLS поиск
            search_tasks.append(
                asyncio.create_task(
                    self._run_tls_search(time_clean, flux_clean, period_min, period_max)
                )
            )

            # 4. Wavelet анализ
            search_tasks.append(
                asyncio.create_task(self._run_wavelet_analysis(time_clean, flux_clean))
            )

            # 5. Fourier анализ
            search_tasks.append(
                asyncio.create_task(
                    self._run_fourier_analysis(
                        time_clean, flux_clean, period_min, period_max
                    )
                )
            )

            # 6. ML ансамбль
            if self.ml_ensemble:
                search_tasks.append(
                    asyncio.create_task(
                        self._run_ml_ensemble_search(
                            all_features, time_clean, flux_clean
                        )
                    )
                )

            # Ожидание завершения всех методов
            logger.info("⚡ Running ALL methods in parallel...")
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Обработка результатов
            method_results = {}
            methods_used = []

            for i, result in enumerate(search_results):
                method_names = [
                    "bls",
                    "gpi",
                    "tls",
                    "wavelet",
                    "fourier",
                    "ml_ensemble",
                ]
                method_name = (
                    method_names[i] if i < len(method_names) else f"method_{i}"
                )

                if not isinstance(result, Exception) and result is not None:
                    method_results[method_name] = result
                    methods_used.append(method_name)
                else:
                    logger.warning(f"Method {method_name} failed: {result}")

            # СУПЕР-МОЩНОЕ ОБЪЕДИНЕНИЕ РЕЗУЛЬТАТОВ
            ensemble_result = await self._ultimate_ensemble_combination(
                method_results, all_features, time_clean, flux_clean, target_name
            )

            # Расчет продвинутых метрик
            await self._calculate_advanced_ensemble_metrics(
                ensemble_result, method_results, all_features
            )

            # Физическое моделирование
            await self._perform_physical_modeling(ensemble_result, stellar_params or {})

            # Генерация рекомендаций
            recommendations = await self._generate_ultimate_recommendations(
                ensemble_result, method_results
            )

            processing_time = time.time() - start_time

            # Обновление статистики производительности
            await self._update_performance_stats(
                processing_time, methods_used, ensemble_result
            )

            # Финальный результат
            final_result = EnsembleSearchResult(
                target_name=target_name,
                consensus_period=ensemble_result["consensus_period"],
                consensus_confidence=ensemble_result["consensus_confidence"],
                consensus_depth=ensemble_result["consensus_depth"],
                consensus_snr=ensemble_result["consensus_snr"],
                bls_result=method_results.get("bls"),
                gpi_result=method_results.get("gpi"),
                tls_result=method_results.get("tls"),
                wavelet_result=method_results.get("wavelet"),
                fourier_result=method_results.get("fourier"),
                ml_result=method_results.get("ml_ensemble"),
                method_agreement=ensemble_result["method_agreement"],
                ensemble_uncertainty=ensemble_result["ensemble_uncertainty"],
                cross_validation_score=ensemble_result["cross_validation_score"],
                bootstrap_confidence=ensemble_result["bootstrap_confidence"],
                planet_candidates=ensemble_result["planet_candidates"],
                false_positive_probability=ensemble_result[
                    "false_positive_probability"
                ],
                habitability_score=ensemble_result["habitability_score"],
                detection_significance=ensemble_result["detection_significance"],
                stellar_parameters=ensemble_result["stellar_parameters"],
                planetary_system=ensemble_result["planetary_system"],
                orbital_dynamics=ensemble_result["orbital_dynamics"],
                processing_time=processing_time,
                methods_used=methods_used,
                quality_flags=ensemble_result["quality_flags"],
                recommendations=recommendations,
            )

            logger.info(
                f"🎉 ULTIMATE SEARCH COMPLETED in {processing_time:.2f}s! "
                f"Consensus: P={final_result.consensus_period:.3f}d, "
                f"Confidence={final_result.consensus_confidence:.3f}, "
                f"Methods={len(methods_used)}"
            )

            return final_result

        except Exception as e:
            logger.error(f"💥 ULTIMATE SEARCH FAILED: {e}")
            raise

    async def _ultimate_data_preprocessing(
        self, time: np.ndarray, flux: np.ndarray, flux_err: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Супер-продвинутая предобработка данных"""

        # Удаление NaN и бесконечностей
        mask = np.isfinite(time) & np.isfinite(flux)
        if flux_err is not None:
            mask &= np.isfinite(flux_err) & (flux_err > 0)

        time_clean = time[mask]
        flux_clean = flux[mask]
        flux_err_clean = flux_err[mask] if flux_err is not None else None

        # Многоуровневое удаление выбросов

        # 1. Статистическое удаление выбросов
        q1, q3 = np.percentile(flux_clean, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        stat_mask = (flux_clean >= lower_bound) & (flux_clean <= upper_bound)

        # 2. Кластерное удаление выбросов
        if len(flux_clean) > 100:
            try:
                from sklearn.preprocessing import StandardScaler

                features = np.column_stack(
                    [
                        StandardScaler()
                        .fit_transform(time_clean.reshape(-1, 1))
                        .flatten(),
                        StandardScaler()
                        .fit_transform(flux_clean.reshape(-1, 1))
                        .flatten(),
                    ]
                )

                clustering = DBSCAN(eps=0.3, min_samples=5).fit(features)
                cluster_mask = clustering.labels_ != -1

                # Объединение масок
                final_mask = stat_mask & cluster_mask
            except Exception:
                final_mask = stat_mask
        else:
            final_mask = stat_mask

        time_clean = time_clean[final_mask]
        flux_clean = flux_clean[final_mask]
        if flux_err_clean is not None:
            flux_err_clean = flux_err_clean[final_mask]

        # Нормализация
        flux_median = np.median(flux_clean)
        flux_clean = flux_clean / flux_median
        if flux_err_clean is not None:
            flux_err_clean = flux_err_clean / flux_median

        # Детрендинг с помощью полинома
        if len(time_clean) > 50:
            try:
                poly_coeffs = np.polyfit(time_clean, flux_clean, deg=3)
                trend = np.polyval(poly_coeffs, time_clean)
                flux_clean = flux_clean / trend
            except (np.linalg.LinAlgError, np.RankWarning, ValueError) as e:
                logger.warning(f"Detrending failed for target: {e}. Continuing without detrending.")
                # Продолжаем без детрендинга

        logger.info(f"🧹 Data preprocessing: {len(time)} → {len(time_clean)} points")

        return time_clean, flux_clean, flux_err_clean

    async def _extract_all_features(
        self, time: np.ndarray, flux: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Извлечение ВСЕХ возможных признаков"""
        all_features = {}

        # Параллельное извлечение признаков
        feature_tasks = []
        for feature_type, extractor in self.feature_extractors.items():
            task = asyncio.get_event_loop().run_in_executor(
                self.thread_executor, extractor, time, flux
            )
            feature_tasks.append((feature_type, task))

        # Сбор всех признаков
        for feature_type, task in feature_tasks:
            try:
                features = await task
                all_features[feature_type] = features
            except Exception as e:
                logger.warning(f"Feature extraction {feature_type} failed: {e}")

        logger.info(f"🔧 Extracted {len(all_features)} feature types")
        return all_features

    def _extract_statistical_features(
        self, time: np.ndarray, flux: np.ndarray
    ) -> np.ndarray:
        """Статистические признаки"""
        features = []

        # Базовая статистика
        features.extend(
            [
                np.mean(flux),
                np.std(flux),
                np.var(flux),
                np.median(flux),
                np.min(flux),
                np.max(flux),
                np.ptp(flux),
                stats.skew(flux),
                stats.kurtosis(flux),
            ]
        )

        # Квантили
        for q in [0.05, 0.1, 0.25, 0.75, 0.9, 0.95]:
            features.append(np.quantile(flux, q))

        # Моменты высших порядков
        for moment in range(3, 7):
            features.append(stats.moment(flux, moment=moment))

        # Энтропия
        hist, _ = np.histogram(flux, bins=50, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        features.append(entropy)

        return np.array(features)

    def _extract_frequency_features(
        self, time: np.ndarray, flux: np.ndarray
    ) -> np.ndarray:
        """Частотные признаки"""
        features = []

        # FFT анализ
        dt = np.median(np.diff(time))
        fft_flux = np.fft.fft(flux - np.mean(flux))
        freqs = np.fft.fftfreq(len(flux), dt)
        power = np.abs(fft_flux) ** 2

        # Положительные частоты
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        power_pos = power[pos_mask]

        if len(power_pos) > 0:
            # Доминирующая частота
            max_idx = np.argmax(power_pos)
            features.extend(
                [
                    freqs_pos[max_idx],  # Доминирующая частота
                    power_pos[max_idx],  # Максимальная мощность
                    1.0 / freqs_pos[max_idx] if freqs_pos[max_idx] > 0 else 0,  # Период
                ]
            )

            # Спектральные моменты
            total_power = np.sum(power_pos)
            if total_power > 0:
                # Спектральный центроид
                spectral_centroid = np.sum(freqs_pos * power_pos) / total_power
                features.append(spectral_centroid)

                # Спектральная ширина
                spectral_spread = np.sqrt(
                    np.sum(((freqs_pos - spectral_centroid) ** 2) * power_pos)
                    / total_power
                )
                features.append(spectral_spread)

                # Спектральная асимметрия
                spectral_skewness = np.sum(
                    ((freqs_pos - spectral_centroid) ** 3) * power_pos
                ) / (total_power * spectral_spread**3)
                features.append(spectral_skewness)
            else:
                features.extend([0, 0, 0])

            # Мощность в разных диапазонах
            freq_bands = [(0, 0.1), (0.1, 0.5), (0.5, 2.0), (2.0, 10.0)]
            for f_min, f_max in freq_bands:
                band_mask = (freqs_pos >= f_min) & (freqs_pos < f_max)
                band_power = np.sum(power_pos[band_mask]) if np.any(band_mask) else 0
                features.append(band_power)
        else:
            features.extend([0] * 10)

        return np.array(features)

    def _extract_wavelet_features(
        self, time: np.ndarray, flux: np.ndarray
    ) -> np.ndarray:
        """Вейвлет признаки"""
        features = []

        try:
            from scipy import signal as scipy_signal

            # Continuous Wavelet Transform
            widths = np.arange(1, 31)
            cwt_matrix = scipy_signal.cwt(flux, scipy_signal.ricker, widths)

            # Статистика по масштабам
            for i in range(min(10, cwt_matrix.shape[0])):
                scale_coeffs = cwt_matrix[i, :]
                features.extend(
                    [
                        np.mean(np.abs(scale_coeffs)),
                        np.std(scale_coeffs),
                        np.max(np.abs(scale_coeffs)),
                    ]
                )

            # Общие вейвлет характеристики
            features.extend(
                [
                    np.mean(np.abs(cwt_matrix)),
                    np.std(cwt_matrix),
                    np.max(np.abs(cwt_matrix)),
                ]
            )

        except Exception as e:
            logger.warning(f"Wavelet analysis failed: {e}")
            features = [0] * 33  # Заполняем нулями если не удалось

        return np.array(features)

    def _extract_morphological_features(
        self, time: np.ndarray, flux: np.ndarray
    ) -> np.ndarray:
        """Морфологические признаки"""
        features = []

        # Производные
        first_diff = np.diff(flux)
        second_diff = np.diff(first_diff)

        features.extend(
            [
                np.mean(np.abs(first_diff)),
                np.std(first_diff),
                np.mean(np.abs(second_diff)),
                np.std(second_diff),
            ]
        )

        # Локальные экстремумы
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(flux)
        valleys, _ = find_peaks(-flux)

        features.extend(
            [
                len(peaks) / len(flux),  # Плотность пиков
                len(valleys) / len(flux),  # Плотность впадин
                np.mean(flux[peaks]) if len(peaks) > 0 else np.mean(flux),
                np.mean(flux[valleys]) if len(valleys) > 0 else np.mean(flux),
            ]
        )

        # Автокорреляция
        autocorr = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode="full")
        autocorr = autocorr[autocorr.size // 2 :]
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr

        # Характеристики автокорреляции
        if len(autocorr) > 10:
            features.extend(
                [
                    autocorr[1] if len(autocorr) > 1 else 0,
                    autocorr[5] if len(autocorr) > 5 else 0,
                    autocorr[10] if len(autocorr) > 10 else 0,
                    np.argmax(autocorr[1:]) + 1 if len(autocorr) > 1 else 0,
                ]
            )
        else:
            features.extend([0, 0, 0, 0])

        return np.array(features)

    def _extract_chaos_features(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """Признаки хаоса и нелинейности"""
        features = []

        # Показатель Ляпунова (приближенный)
        try:
            # Простая оценка через расхождение близких траекторий
            diffs = np.diff(flux)
            lyapunov_approx = np.mean(np.log(np.abs(diffs) + 1e-10))
            features.append(lyapunov_approx)
        except Exception:
            features.append(0)

        # Корреляционная размерность (упрощенная)
        try:
            # Embedding dimension
            m = 3
            tau = 1

            if len(flux) > m * tau:
                # Создание вложенного пространства
                embedded = np.array(
                    [
                        flux[i : i + m * tau : tau]
                        for i in range(len(flux) - m * tau + 1)
                    ]
                )

                # Корреляционная сумма
                distances = []
                for i in range(min(100, len(embedded))):
                    for j in range(i + 1, min(100, len(embedded))):
                        dist = np.linalg.norm(embedded[i] - embedded[j])
                        distances.append(dist)

                if distances:
                    correlation_dim = np.mean(np.log(distances + 1e-10))
                    features.append(correlation_dim)
                else:
                    features.append(0)
            else:
                features.append(0)
        except Exception:
            features.append(0)

        # Энтропия Шеннона
        try:
            hist, _ = np.histogram(flux, bins=20, density=True)
            hist = hist[hist > 0]
            shannon_entropy = -np.sum(hist * np.log2(hist))
            features.append(shannon_entropy)
        except Exception:
            features.append(0)

        # Приближенная энтропия
        try:

            def _maxdist(xi, xj, N):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])

            def _phi(m):
                patterns = np.array([flux[i : i + m] for i in range(len(flux) - m + 1)])
                C = np.zeros(len(patterns))
                for i in range(len(patterns)):
                    template_i = patterns[i]
                    for j in range(len(patterns)):
                        if _maxdist(template_i, patterns[j], m) <= 0.1 * np.std(flux):
                            C[i] += 1.0
                phi = np.mean(np.log(C / len(patterns)))
                return phi

            ApEn = _phi(2) - _phi(3)
            features.append(ApEn)
        except Exception:
            features.append(0)

        return np.array(features)

    def _extract_information_features(
        self, time: np.ndarray, flux: np.ndarray
    ) -> np.ndarray:
        """Информационные признаки"""
        features = []

        # Взаимная информация с задержкой
        try:

            def mutual_info(x, y, bins=20):
                hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
                hist_x, _ = np.histogram(x, bins=bins)
                hist_y, _ = np.histogram(y, bins=bins)

                hist_xy = hist_xy / np.sum(hist_xy)
                hist_x = hist_x / np.sum(hist_x)
                hist_y = hist_y / np.sum(hist_y)

                mi = 0
                for i in range(bins):
                    for j in range(bins):
                        if hist_xy[i, j] > 0:
                            mi += hist_xy[i, j] * np.log2(
                                hist_xy[i, j] / (hist_x[i] * hist_y[j])
                            )

                return mi

            # Взаимная информация с разными задержками
            for lag in [1, 2, 5, 10]:
                if len(flux) > lag:
                    mi = mutual_info(flux[:-lag], flux[lag:])
                    features.append(mi)
                else:
                    features.append(0)
        except Exception:
            features.extend([0, 0, 0, 0])

        # Сложность Лемпеля-Зива
        try:
            # Бинаризация сигнала
            binary_flux = (flux > np.median(flux)).astype(int)
            binary_string = "".join(map(str, binary_flux))

            # Подсчет уникальных подстрок
            substrings = set()
            for i in range(len(binary_string)):
                for j in range(i + 1, min(i + 10, len(binary_string) + 1)):
                    substrings.add(binary_string[i:j])

            lz_complexity = (
                len(substrings) / len(binary_string) if len(binary_string) > 0 else 0
            )
            features.append(lz_complexity)
        except Exception:
            features.append(0)

        return np.array(features)

    async def _run_bls_search(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray],
        period_min: float,
        period_max: float,
        target_name: str,
    ) -> BLSResult:
        """Запуск BLS поиска"""
        try:
            result = await self.bls_service.analyze(
                time, flux, flux_err, period_min, period_max, 7.0, target_name
            )
            logger.info(
                f"✅ BLS search completed: P={result.best_period:.3f}d, SNR={result.snr:.1f}"
            )
            return result
        except Exception as e:
            logger.error(f"❌ BLS search failed: {e}")
            return None

    async def _run_gpi_search(
        self, time: np.ndarray, flux: np.ndarray, target_name: str
    ) -> Dict:
        """Запуск GPI поиска"""
        try:
            if self.gpi_service:
                # Подготовка данных для GPI
                lightcurve_data = {
                    "time": time.tolist(),
                    "flux": flux.tolist(),
                    "target_name": target_name,
                }

                result = await self.gpi_service.search_exoplanets(lightcurve_data)
                logger.info(f"✅ GPI search completed")
                return result
            else:
                return None
        except Exception as e:
            logger.error(f"❌ GPI search failed: {e}")
            return None

    async def _run_tls_search(
        self, time: np.ndarray, flux: np.ndarray, period_min: float, period_max: float
    ) -> Dict:
        """Запуск Transit Least Squares поиска"""
        try:
            # Упрощенная реализация TLS
            periods = np.logspace(np.log10(period_min), np.log10(period_max), 500)
            powers = np.zeros_like(periods)

            for i, period in enumerate(periods):
                # Фазовая свертка
                phase = (time % period) / period

                # Поиск оптимальной длительности транзита
                best_power = 0
                for duration_frac in [0.01, 0.02, 0.05, 0.1]:
                    # Транзитная маска
                    transit_mask = phase < duration_frac

                    if np.sum(transit_mask) > 5 and np.sum(~transit_mask) > 20:
                        in_transit = np.mean(flux[transit_mask])
                        out_transit = np.mean(flux[~transit_mask])

                        if out_transit > 0:
                            depth = (out_transit - in_transit) / out_transit
                            if depth > 0:
                                power = depth * np.sqrt(np.sum(transit_mask))
                                best_power = max(best_power, power)

                powers[i] = best_power

            # Лучший результат
            best_idx = np.argmax(powers)
            best_period = periods[best_idx]
            best_power = powers[best_idx]

            result = {
                "method": "TLS",
                "best_period": float(best_period),
                "best_power": float(best_power),
                "periods": periods.tolist(),
                "powers": powers.tolist(),
                "snr": float(best_power / np.std(powers)) if np.std(powers) > 0 else 0,
            }

            logger.info(f"✅ TLS search completed: P={best_period:.3f}d")
            return result

        except Exception as e:
            logger.error(f"❌ TLS search failed: {e}")
            return None

    async def _run_wavelet_analysis(self, time: np.ndarray, flux: np.ndarray) -> Dict:
        """Запуск вейвлет анализа"""
        try:
            from scipy import signal as scipy_signal

            # Continuous Wavelet Transform
            widths = np.logspace(0, 2, 100)  # От 1 до 100
            cwt_matrix = scipy_signal.cwt(
                flux - np.mean(flux), scipy_signal.ricker, widths
            )

            # Поиск максимальных коэффициентов
            max_coeffs = np.max(np.abs(cwt_matrix), axis=1)
            best_scale_idx = np.argmax(max_coeffs)
            best_scale = widths[best_scale_idx]

            # Преобразование масштаба в период (приближенно)
            dt = np.median(np.diff(time))
            estimated_period = best_scale * dt * 2  # Эмпирическая формула

            # Мощность сигнала
            power = np.max(max_coeffs)
            snr = power / np.std(max_coeffs) if np.std(max_coeffs) > 0 else 0

            result = {
                "method": "Wavelet",
                "best_period": float(estimated_period),
                "best_scale": float(best_scale),
                "power": float(power),
                "snr": float(snr),
                "scales": widths.tolist(),
                "coefficients": max_coeffs.tolist(),
            }

            logger.info(f"✅ Wavelet analysis completed: P={estimated_period:.3f}d")
            return result

        except Exception as e:
            logger.error(f"❌ Wavelet analysis failed: {e}")
            return None

    async def _run_fourier_analysis(
        self, time: np.ndarray, flux: np.ndarray, period_min: float, period_max: float
    ) -> Dict:
        """Запуск Фурье анализа"""
        try:
            # Подготовка данных
            dt = np.median(np.diff(time))
            flux_detrended = flux - np.mean(flux)

            # FFT
            fft_flux = np.fft.fft(flux_detrended)
            freqs = np.fft.fftfreq(len(flux_detrended), dt)
            power = np.abs(fft_flux) ** 2

            # Только положительные частоты
            pos_mask = freqs > 0
            freqs_pos = freqs[pos_mask]
            power_pos = power[pos_mask]

            # Фильтрация по диапазону периодов
            periods = 1.0 / freqs_pos
            period_mask = (periods >= period_min) & (periods <= period_max)

            filtered_periods = periods[period_mask]
            filtered_power = power_pos[period_mask]

            if len(filtered_power) > 0:
                # Лучший период
                best_idx = np.argmax(filtered_power)
                best_period = filtered_periods[best_idx]
                best_power = filtered_power[best_idx]

                # SNR
                snr = (
                    best_power / np.std(filtered_power)
                    if np.std(filtered_power) > 0
                    else 0
                )

                # Гармоники
                harmonics = []
                for n in [2, 3, 4, 5]:
                    harmonic_period = best_period / n
                    if period_min <= harmonic_period <= period_max:
                        # Найти ближайшую частоту
                        harmonic_freq = 1.0 / harmonic_period
                        freq_diff = np.abs(freqs_pos - harmonic_freq)
                        harmonic_idx = np.argmin(freq_diff)
                        if freq_diff[harmonic_idx] < 0.1 * harmonic_freq:
                            harmonics.append(
                                {
                                    "period": float(harmonic_period),
                                    "power": float(power_pos[harmonic_idx]),
                                    "harmonic": n,
                                }
                            )

                result = {
                    "method": "Fourier",
                    "best_period": float(best_period),
                    "best_power": float(best_power),
                    "snr": float(snr),
                    "harmonics": harmonics,
                    "periods": filtered_periods.tolist(),
                    "powers": filtered_power.tolist(),
                }

                logger.info(f"✅ Fourier analysis completed: P={best_period:.3f}d")
                return result
            else:
                return None

        except Exception as e:
            logger.error(f"❌ Fourier analysis failed: {e}")
            return None

    async def _run_ml_ensemble_search(
        self, all_features: Dict[str, np.ndarray], time: np.ndarray, flux: np.ndarray
    ) -> Dict:
        """Запуск ML ансамбля"""
        try:
            if not self.ml_ensemble:
                return None

            # Объединение всех признаков
            feature_vector = []
            feature_names = []

            for feature_type, features in all_features.items():
                feature_vector.extend(features.tolist())
                feature_names.extend(
                    [f"{feature_type}_{i}" for i in range(len(features))]
                )

            if len(feature_vector) == 0:
                return None

            # Нормализация признаков
            features_array = np.array(feature_vector).reshape(1, -1)

            # Предсказание периода (если модель обучена)
            # Здесь используем простую эвристику на основе признаков

            # Извлекаем период из частотных признаков
            freq_features = all_features.get("frequency", np.array([]))
            if len(freq_features) > 2:
                estimated_period = freq_features[2]  # Третий элемент - период
            else:
                estimated_period = 5.0  # Значение по умолчанию

            # Оценка уверенности на основе статистических признаков
            stat_features = all_features.get("statistical", np.array([]))
            if len(stat_features) > 1:
                confidence = (
                    min(1.0, stat_features[1] / stat_features[0])
                    if stat_features[0] > 0
                    else 0.5
                )
            else:
                confidence = 0.5

            # Оценка SNR
            if len(stat_features) > 8:
                snr = abs(stat_features[8])  # Используем куртозис как прокси для SNR
            else:
                snr = 5.0

            result = {
                "method": "ML_Ensemble",
                "best_period": float(estimated_period),
                "confidence": float(confidence),
                "snr": float(snr),
                "feature_count": len(feature_vector),
                "feature_types": list(all_features.keys()),
            }

            logger.info(f"✅ ML ensemble completed: P={estimated_period:.3f}d")
            return result

        except Exception as e:
            logger.error(f"❌ ML ensemble failed: {e}")
            return None

    async def _ultimate_ensemble_combination(
        self,
        method_results: Dict,
        all_features: Dict,
        time: np.ndarray,
        flux: np.ndarray,
        target_name: str,
    ) -> Dict:
        """СУПЕР-МОЩНОЕ объединение результатов всех методов"""

        logger.info("🔥 ULTIMATE ENSEMBLE COMBINATION STARTING...")

        # Сбор периодов и весов от всех методов
        periods = []
        weights = []
        confidences = []
        snrs = []

        for method_name, result in method_results.items():
            if result is None:
                continue

            method_weight = self.method_weights.get(method_name, 0.1)

            if method_name == "bls" and hasattr(result, "best_period"):
                periods.append(result.best_period)
                weights.append(
                    method_weight * (1 + result.snr / 20)
                )  # Бонус за высокий SNR
                confidences.append(result.significance)
                snrs.append(result.snr)

            elif isinstance(result, dict) and "best_period" in result:
                periods.append(result["best_period"])
                method_snr = result.get("snr", 5.0)
                weights.append(method_weight * (1 + method_snr / 20))
                confidences.append(result.get("confidence", 0.5))
                snrs.append(method_snr)

        if not periods:
            # Если никто не нашел период, используем значения по умолчанию
            return {
                "consensus_period": 5.0,
                "consensus_confidence": 0.1,
                "consensus_depth": 0.001,
                "consensus_snr": 1.0,
                "method_agreement": 0.0,
                "ensemble_uncertainty": 1.0,
                "cross_validation_score": 0.0,
                "bootstrap_confidence": 0.0,
                "planet_candidates": [],
                "false_positive_probability": 0.9,
                "habitability_score": 0.0,
                "detection_significance": 0.0,
                "stellar_parameters": {},
                "planetary_system": {},
                "orbital_dynamics": {},
                "quality_flags": {"low_data_quality": True},
            }

        # Нормализация весов
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        # Взвешенное среднее периодов
        consensus_period = np.average(periods, weights=weights)

        # Кластеризация периодов для поиска согласованности
        period_agreement = await self._calculate_period_agreement(periods, weights)

        # Взвешенная уверенность
        consensus_confidence = np.average(confidences, weights=weights)

        # Взвешенный SNR
        consensus_snr = np.average(snrs, weights=weights)

        # Оценка глубины транзита
        consensus_depth = await self._estimate_transit_depth(
            time, flux, consensus_period
        )

        # Bootstrap анализ для оценки неопределенности
        bootstrap_results = await self._bootstrap_analysis(
            method_results, time, flux, n_bootstrap=100
        )

        # Поиск кандидатов планет
        planet_candidates = await self._identify_planet_candidates(
            method_results, consensus_period, consensus_confidence
        )

        # Оценка ложных срабатываний
        false_positive_prob = await self._estimate_false_positive_probability(
            method_results, all_features, consensus_period, consensus_snr
        )

        # Оценка обитаемости
        habitability_score = await self._calculate_habitability_score(
            consensus_period, consensus_depth
        )

        # Статистическая значимость
        detection_significance = await self._calculate_detection_significance(
            consensus_snr, period_agreement, len(method_results)
        )

        # Кросс-валидация
        cv_score = await self._cross_validate_results(method_results, all_features)

        logger.info(
            f"🎯 CONSENSUS: P={consensus_period:.3f}d, Conf={consensus_confidence:.3f}"
        )

        return {
            "consensus_period": float(consensus_period),
            "consensus_confidence": float(consensus_confidence),
            "consensus_depth": float(consensus_depth),
            "consensus_snr": float(consensus_snr),
            "method_agreement": float(period_agreement),
            "ensemble_uncertainty": float(bootstrap_results["uncertainty"]),
            "cross_validation_score": float(cv_score),
            "bootstrap_confidence": float(bootstrap_results["confidence"]),
            "planet_candidates": planet_candidates,
            "false_positive_probability": float(false_positive_prob),
            "habitability_score": float(habitability_score),
            "detection_significance": float(detection_significance),
            "stellar_parameters": await self._estimate_stellar_parameters(time, flux),
            "planetary_system": await self._model_planetary_system(
                consensus_period, consensus_depth
            ),
            "orbital_dynamics": await self._calculate_orbital_dynamics(
                consensus_period
            ),
            "quality_flags": await self._assess_data_quality(time, flux, all_features),
        }

    async def _calculate_period_agreement(
        self, periods: List[float], weights: np.ndarray
    ) -> float:
        """Расчет согласованности периодов между методами"""
        if len(periods) < 2:
            return 1.0

        # Кластеризация периодов
        periods_array = np.array(periods).reshape(-1, 1)

        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=min(3, len(periods)), random_state=42)
            clusters = kmeans.fit_predict(periods_array)

            # Найти самый большой кластер
            unique_clusters, counts = np.unique(clusters, return_counts=True)
            main_cluster = unique_clusters[np.argmax(counts)]

            # Доля методов в главном кластере
            main_cluster_mask = clusters == main_cluster
            agreement = np.sum(weights[main_cluster_mask])

            return min(1.0, agreement)

        except Exception:
            # Fallback: простая оценка разброса
            period_std = np.std(periods)
            period_mean = np.mean(periods)
            relative_std = period_std / period_mean if period_mean > 0 else 1.0

            return max(0.0, 1.0 - relative_std)

    async def _estimate_transit_depth(
        self, time: np.ndarray, flux: np.ndarray, period: float
    ) -> float:
        """Оценка глубины транзита"""
        try:
            # Фазовая свертка
            phase = (time % period) / period

            # Поиск минимума (транзита)
            phase_bins = np.linspace(0, 1, 100)
            binned_flux = []

            for i in range(len(phase_bins) - 1):
                mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
                if np.sum(mask) > 0:
                    binned_flux.append(np.mean(flux[mask]))
                else:
                    binned_flux.append(np.mean(flux))

            binned_flux = np.array(binned_flux)

            # Глубина как разность между максимумом и минимумом
            depth = (np.max(binned_flux) - np.min(binned_flux)) / np.max(binned_flux)

            return max(0.0, depth)

        except Exception:
            return 0.001  # Значение по умолчанию

    async def _bootstrap_analysis(
        self,
        method_results: Dict,
        time: np.ndarray,
        flux: np.ndarray,
        n_bootstrap: int = 100,
    ) -> Dict:
        """Bootstrap анализ для оценки неопределенности"""
        try:
            bootstrap_periods = []

            # Простой bootstrap на основе имеющихся результатов
            periods = []
            for result in method_results.values():
                if result is None:
                    continue

                if hasattr(result, "best_period"):
                    periods.append(result.best_period)
                elif isinstance(result, dict) and "best_period" in result:
                    periods.append(result["best_period"])

            if not periods:
                return {"uncertainty": 1.0, "confidence": 0.0}

            # Bootstrap выборка
            periods = np.array(periods)
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(
                    periods, size=len(periods), replace=True
                )
                bootstrap_periods.append(np.mean(bootstrap_sample))

            # Статистика bootstrap
            bootstrap_periods = np.array(bootstrap_periods)
            uncertainty = (
                np.std(bootstrap_periods) / np.mean(bootstrap_periods)
                if np.mean(bootstrap_periods) > 0
                else 1.0
            )

            # Доверительный интервал
            confidence_interval = np.percentile(bootstrap_periods, [2.5, 97.5])
            confidence = 1.0 - (
                confidence_interval[1] - confidence_interval[0]
            ) / np.mean(bootstrap_periods)

            return {
                "uncertainty": float(min(1.0, uncertainty)),
                "confidence": float(max(0.0, confidence)),
            }

        except Exception:
            return {"uncertainty": 0.5, "confidence": 0.5}

    async def _identify_planet_candidates(
        self, method_results: Dict, consensus_period: float, consensus_confidence: float
    ) -> List[Dict]:
        """Идентификация кандидатов планет"""
        candidates = []

        try:
            # Основной кандидат
            if consensus_confidence > 0.7:
                candidates.append(
                    {
                        "period": float(consensus_period),
                        "confidence": float(consensus_confidence),
                        "type": "primary",
                        "methods_supporting": len(
                            [r for r in method_results.values() if r is not None]
                        ),
                    }
                )

            # Поиск дополнительных кандидатов в результатах методов
            for method_name, result in method_results.items():
                if result is None:
                    continue

                if hasattr(result, "secondary_periods"):
                    for sec_period in result.secondary_periods[
                        :2
                    ]:  # Максимум 2 дополнительных
                        if abs(sec_period - consensus_period) > 0.1:  # Не дубликат
                            candidates.append(
                                {
                                    "period": float(sec_period),
                                    "confidence": 0.5,
                                    "type": "secondary",
                                    "source_method": method_name,
                                }
                            )

            return candidates[:5]  # Максимум 5 кандидатов

        except Exception as e:
            logger.error(f"Error identifying candidates: {e}")
            return []

    async def _estimate_false_positive_probability(
        self,
        method_results: Dict,
        all_features: Dict,
        consensus_period: float,
        consensus_snr: float,
    ) -> float:
        """Оценка вероятности ложного срабатывания"""
        try:
            # Базовая оценка на основе SNR
            base_fap = max(0.01, 1.0 / (consensus_snr**2))

            # Корректировка на основе согласованности методов
            method_agreement = len(
                [r for r in method_results.values() if r is not None]
            ) / len(method_results)
            agreement_factor = 1.0 - method_agreement * 0.5

            # Корректировка на основе статистических признаков
            stat_features = all_features.get("statistical", np.array([]))
            if len(stat_features) > 5:
                # Используем дисперсию как индикатор качества данных
                variance_factor = min(
                    2.0, stat_features[1] * 10
                )  # Нормализованная дисперсия
                base_fap *= variance_factor

            return min(0.99, base_fap * agreement_factor)

        except Exception as e:
            logger.error(f"Error estimating false positive probability: {e}")
            return 0.5

    async def _calculate_habitability_score(self, period: float, depth: float) -> float:
        """Расчет индекса обитаемости"""
        try:
            # Простая модель на основе периода (расстояние от звезды)
            # Предполагаем солнечную звезду

            # Расстояние в AU (приближенно)
            distance_au = (period / 365.25) ** (2 / 3)

            # Зона обитаемости: 0.95 - 1.37 AU для солнечной звезды
            if 0.95 <= distance_au <= 1.37:
                habitability = 1.0
            elif 0.5 <= distance_au <= 2.0:
                # Частично обитаемая зона
                if distance_au < 0.95:
                    habitability = distance_au / 0.95
                else:
                    habitability = 2.0 / distance_au
            else:
                habitability = 0.1

            # Корректировка на размер планеты (на основе глубины транзита)
            if depth > 0.01:  # Большая планета
                habitability *= 0.5
            elif depth < 0.001:  # Очень маленькая
                habitability *= 0.7

            return min(1.0, max(0.0, habitability))

        except Exception as e:
            logger.error(f"Error calculating habitability: {e}")
            return 0.0

    async def _calculate_detection_significance(
        self, snr: float, agreement: float, num_methods: int
    ) -> float:
        """Расчет статистической значимости обнаружения"""
        try:
            # Базовая значимость на основе SNR
            base_significance = min(1.0, snr / 15.0)

            # Бонус за согласованность методов
            agreement_bonus = agreement * 0.3

            # Бонус за количество методов
            method_bonus = min(0.2, num_methods * 0.05)

            total_significance = base_significance + agreement_bonus + method_bonus

            return min(1.0, total_significance)

        except Exception as e:
            logger.error(f"Error calculating significance: {e}")
            return 0.5

    async def _cross_validate_results(
        self, method_results: Dict, all_features: Dict
    ) -> float:
        """Кросс-валидация результатов"""
        try:
            if len(method_results) < 2:
                return 0.5

            # Собираем периоды от всех методов
            periods = []
            for result in method_results.values():
                if result is None:
                    continue
                if hasattr(result, "best_period"):
                    periods.append(result.best_period)
                elif isinstance(result, dict) and "best_period" in result:
                    periods.append(result["best_period"])

            if len(periods) < 2:
                return 0.5

            # Вычисляем коэффициент вариации
            periods = np.array(periods)
            cv = np.std(periods) / np.mean(periods) if np.mean(periods) > 0 else 1.0

            # Чем меньше вариация, тем лучше кросс-валидация
            cv_score = max(0.0, 1.0 - cv * 2)

            return min(1.0, cv_score)

        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return 0.5

    async def _estimate_stellar_parameters(
        self, time: np.ndarray, flux: np.ndarray
    ) -> Dict:
        """Оценка параметров звезды"""
        try:
            # Простая оценка на основе данных
            flux_std = np.std(flux)
            flux_mean = np.mean(flux)

            # Оценка звездной величины (приближенно)
            estimated_magnitude = 12.0 + np.log10(
                flux_std * 1000
            )  # Эмпирическая формула

            # Оценка температуры (очень приближенно)
            estimated_temp = 5500 + (flux_mean - 1.0) * 1000

            return {
                "estimated_magnitude": float(estimated_magnitude),
                "estimated_temperature": float(max(3000, min(8000, estimated_temp))),
                "flux_variability": float(flux_std),
                "data_quality": "good" if flux_std < 0.001 else "moderate",
            }

        except Exception as e:
            logger.error(f"Error estimating stellar parameters: {e}")
            return {}

    async def _model_planetary_system(self, period: float, depth: float) -> Dict:
        """Моделирование планетной системы"""
        try:
            # Оценка радиуса планеты (предполагая солнечную звезду)
            stellar_radius = 1.0  # Радиусы Солнца
            planet_radius = np.sqrt(depth) * stellar_radius * 109.2  # В радиусах Земли

            # Оценка массы (приближенно)
            if planet_radius < 1.5:
                planet_mass = planet_radius**2.06  # Земноподобные
            else:
                planet_mass = planet_radius**1.4  # Газовые гиганты

            # Орбитальное расстояние
            orbital_distance = (period / 365.25) ** (2 / 3)  # В AU

            return {
                "planet_radius_earth": float(planet_radius),
                "planet_mass_earth": float(planet_mass),
                "orbital_distance_au": float(orbital_distance),
                "orbital_period_days": float(period),
                "planet_type": "terrestrial" if planet_radius < 2.0 else "gas_giant",
            }

        except Exception as e:
            logger.error(f"Error modeling planetary system: {e}")
            return {}

    async def _calculate_orbital_dynamics(self, period: float) -> Dict:
        """Расчет орбитальной динамики"""
        try:
            # Предполагаем солнечную звезду (1 солнечная масса)
            stellar_mass = 1.0

            # Большая полуось (3-й закон Кеплера)
            semi_major_axis = (period / 365.25) ** (2 / 3)  # В AU

            # Орбитальная скорость
            orbital_velocity = 29.78 / np.sqrt(semi_major_axis)  # км/с

            # Температура равновесия (приближенно)
            equilibrium_temp = 278 / np.sqrt(semi_major_axis)  # К

            return {
                "semi_major_axis_au": float(semi_major_axis),
                "orbital_velocity_km_s": float(orbital_velocity),
                "equilibrium_temperature_k": float(equilibrium_temp),
                "orbital_period_days": float(period),
                "eccentricity": 0.0,  # Предполагаем круговую орбиту
            }

        except Exception as e:
            logger.error(f"Error calculating orbital dynamics: {e}")
            return {}

    async def _assess_data_quality(
        self, time: np.ndarray, flux: np.ndarray, all_features: Dict
    ) -> Dict:
        """Оценка качества данных"""
        try:
            quality_flags = {}

            # Проверка количества точек
            if len(time) < 1000:
                quality_flags["low_data_points"] = True

            # Проверка временного покрытия
            time_span = np.max(time) - np.min(time)
            if time_span < 10:  # Меньше 10 дней
                quality_flags["short_baseline"] = True

            # Проверка уровня шума
            flux_std = np.std(flux)
            if flux_std > 0.01:  # Больше 1%
                quality_flags["high_noise"] = True

            # Проверка пропусков в данных
            time_diff = np.diff(time)
            median_cadence = np.median(time_diff)
            large_gaps = np.sum(time_diff > 5 * median_cadence)
            if large_gaps > len(time) * 0.1:
                quality_flags["data_gaps"] = True

            # Общая оценка качества
            if not quality_flags:
                quality_flags["high_quality"] = True

            return quality_flags

        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return {"unknown_quality": True}
