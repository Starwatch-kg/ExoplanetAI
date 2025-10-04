"""
Константы для ExoplanetAI - замена магических чисел
"""

# === ФИЗИЧЕСКИЕ КОНСТАНТЫ ===
class TransitConstants:
    """Константы для транзитного анализа"""
    MIN_TRANSIT_DEPTH = 0.001  # ppm - минимальная детектируемая глубина транзита
    MAX_TRANSIT_DEPTH = 0.1    # ppm - максимальная реалистичная глубина
    MIN_PERIOD_DAYS = 0.5      # дней - минимальный орбитальный период
    MAX_PERIOD_DAYS = 50.0     # дней - максимальный период для горячих юпитеров
    MIN_DURATION_HOURS = 0.5   # часов - минимальная длительность транзита
    MAX_DURATION_HOURS = 12.0  # часов - максимальная длительность транзита


class MLConstants:
    """Константы для машинного обучения"""
    DEFAULT_CONFIDENCE_THRESHOLD = 0.8
    MIN_CONFIDENCE_THRESHOLD = 0.5
    MAX_CONFIDENCE_THRESHOLD = 0.99
    MIN_DATA_POINTS = 100      # минимум точек для анализа
    RECOMMENDED_DATA_POINTS = 1000  # рекомендуемое количество
    MAX_FEATURES = 1000        # максимум признаков для ML
    DEFAULT_CROSS_VALIDATION_FOLDS = 5


class APIConstants:
    """Константы для API"""
    DEFAULT_PAGE_SIZE = 20
    MAX_PAGE_SIZE = 100
    MIN_PAGE_SIZE = 1
    DEFAULT_TIMEOUT_SECONDS = 60
    LONG_OPERATION_TIMEOUT_SECONDS = 300
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 1


class CacheConstants:
    """Константы для кэширования"""
    PLANET_INFO_TTL_HOURS = 6      # TTL для информации о планетах
    LIGHTCURVE_TTL_HOURS = 2       # TTL для кривых блеска
    SEARCH_RESULTS_TTL_MINUTES = 30 # TTL для результатов поиска
    HEALTH_CHECK_TTL_SECONDS = 30   # TTL для health checks


class DataQualityConstants:
    """Константы для контроля качества данных"""
    MIN_FLUX_POINTS = 50           # минимум точек для анализа потока
    MAX_OUTLIER_FRACTION = 0.1     # максимальная доля выбросов (10%)
    SIGMA_CLIP_THRESHOLD = 3.0     # порог для sigma clipping
    MIN_OBSERVATION_DURATION_HOURS = 1.0  # минимальная длительность наблюдений


class ValidationConstants:
    """Константы для валидации"""
    MAX_TARGET_NAME_LENGTH = 50
    MIN_TARGET_NAME_LENGTH = 1
    MAX_SEARCH_QUERY_LENGTH = 200
    ALLOWED_MISSIONS = ['TESS', 'Kepler', 'K2', 'JWST']
    ALLOWED_CATALOGS = ['TIC', 'KIC', 'EPIC', 'Gaia DR3']


# === АСТРОНОМИЧЕСКИЕ КОНСТАНТЫ ===
class AstronomicalConstants:
    """Физические и астрономические константы"""
    EARTH_RADIUS_KM = 6371.0       # км
    JUPITER_RADIUS_KM = 69911.0    # км
    SOLAR_RADIUS_KM = 695700.0     # км
    AU_KM = 149597870.7            # км - астрономическая единица
    
    # Типичные звездные радиусы (в солнечных радиусах)
    MAIN_SEQUENCE_RADIUS_RANGE = (0.1, 20.0)
    GIANT_RADIUS_RANGE = (10.0, 100.0)
    WHITE_DWARF_RADIUS_RANGE = (0.008, 0.02)


# === КОНФИГУРАЦИЯ ЛОГИРОВАНИЯ ===
class LoggingConstants:
    """Константы для системы логирования"""
    MAX_LOG_FILE_SIZE_MB = 100
    MAX_LOG_FILES_COUNT = 5
    LOG_ROTATION_INTERVAL_HOURS = 24
    STRUCTURED_LOG_MAX_FIELD_LENGTH = 1000


# === БЕЗОПАСНОСТЬ ===
class SecurityConstants:
    """Константы безопасности"""
    JWT_EXPIRY_HOURS = 24
    REFRESH_TOKEN_EXPIRY_DAYS = 30
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15
    PASSWORD_MIN_LENGTH = 8
    API_KEY_LENGTH = 32


# === ПРОИЗВОДИТЕЛЬНОСТЬ ===
class PerformanceConstants:
    """Константы производительности"""
    MAX_CONCURRENT_REQUESTS = 100
    REQUEST_QUEUE_SIZE = 1000
    WORKER_POOL_SIZE = 4
    MAX_MEMORY_USAGE_MB = 2048
    RESPONSE_TIME_WARNING_MS = 1000
    RESPONSE_TIME_ERROR_MS = 5000
