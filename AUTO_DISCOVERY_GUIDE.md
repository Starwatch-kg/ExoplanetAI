# 🌟 ExoplanetAI Auto Discovery System v2.0

## 🚀 Overview

ExoplanetAI Auto Discovery System - это комплексная система автоматического обнаружения экзопланет, использующая реальные астрономические данные NASA, ESA и других космических агентств с применением передовых методов машинного обучения.

## ✨ Ключевые возможности

- **🔄 Автоматический инжест данных**: Непрерывный мониторинг и загрузка новых кривых блеска из MAST/NASA/ExoFOP
- **🤖 ML-детекция**: Продвинутые модели машинного обучения для классификации кандидатов в экзопланеты
- **📦 Версионирование моделей**: Автоматическое развертывание и откат моделей
- **⚡ Обработка в реальном времени**: Параллельная обработка множественных целей с настраиваемой конкурентностью
- **🔍 Контроль качества**: Комплексная валидация данных и оценка качества
- **📊 Мониторинг**: Встроенный мониторинг производительности и детекция дрейфа

## 🏗️ Архитектура

```
[MAST/NASA/ExoFOP] → Data Ingest → Preprocessing → Feature Extraction → ML Classification → Results
                                                                      ↓
                    Model Registry ← Model Training ← Performance Monitoring
```

## 🚀 Быстрый старт

### 1. Запуск полного пайплайна

```bash
./start_auto_discovery.sh
```

Это запустит:
- 📡 Backend API сервер (порт 8001)
- 🌐 Frontend веб-интерфейс (порт 5173)
- 📥 Сервис инжеста данных
- 🔍 Сервис автоматического обнаружения

### 2. Использование CLI инструмента

```bash
# Проверить статус пайплайна
python backend/cli/auto_discovery_cli.py status

# Запустить инжест данных
python backend/cli/auto_discovery_cli.py ingestion start

# Запустить сервис обнаружения
python backend/cli/auto_discovery_cli.py discovery start --confidence 0.85

# Выполнить один цикл обнаружения
python backend/cli/auto_discovery_cli.py discovery cycle

# Загрузить конкретные цели
python backend/cli/auto_discovery_cli.py ingestion targets TIC-441420236 TIC-307210830

# Список доступных моделей
python backend/cli/auto_discovery_cli.py models list

# Развернуть версию модели
python backend/cli/auto_discovery_cli.py models deploy exoplanet_classifier v1.0

# Запустить полный пайплайн
python backend/cli/auto_discovery_cli.py pipeline start
```

### 3. API эндпоинты

#### 🔍 Управление обнаружением
- `GET /api/v1/auto-discovery/status` - Статус обнаружения
- `POST /api/v1/auto-discovery/start` - Запустить сервис обнаружения
- `POST /api/v1/auto-discovery/stop` - Остановить сервис обнаружения
- `GET /api/v1/auto-discovery/candidates` - Список найденных кандидатов

#### 📥 Инжест данных
- `GET /api/v1/auto-discovery/data-ingestion/stats` - Статистика инжеста
- `POST /api/v1/auto-discovery/data-ingestion/start` - Запустить инжест
- `POST /api/v1/auto-discovery/data-ingestion/stop` - Остановить инжест

#### 🤖 Управление моделями
- `GET /api/v1/auto-discovery/models/registry` - Информация о реестре моделей
- `GET /api/v1/auto-discovery/models/{name}/versions` - Список версий модели
- `POST /api/v1/auto-discovery/models/{name}/deploy/{version}` - Развернуть модель
- `POST /api/v1/auto-discovery/models/{name}/rollback` - Откатить модель

#### 🔧 Управление пайплайном
- `GET /api/v1/auto-discovery/pipeline/status` - Статус полного пайплайна
- `POST /api/v1/auto-discovery/pipeline/start-full` - Запустить всё
- `POST /api/v1/auto-discovery/pipeline/stop-full` - Остановить всё
- `POST /api/v1/auto-discovery/pipeline/configure` - Настроить пайплайн

## ⚙️ Конфигурация

### Переменные окружения

```bash
# Настройки инжеста данных
INGESTION_INTERVAL_HOURS=6
MIN_DATA_POINTS=100
MIN_TIME_SPAN_DAYS=10.0

# Настройки обнаружения
DISCOVERY_INTERVAL_HOURS=6
CONFIDENCE_THRESHOLD=0.85
MAX_CONCURRENT_TASKS=5

# Настройки моделей
PERFORMANCE_THRESHOLD=0.85
DRIFT_THRESHOLD=0.1
MAX_VERSIONS_PER_MODEL=10

# Настройки кэша
CACHE_REDIS_URL=redis://localhost:6379
CACHE_TTL_HOURS=6
```

## 📊 Источники данных

- **🛰️ NASA Exoplanet Archive**: Подтвержденные экзопланеты и кандидаты
- **🔭 MAST**: Кривые блеска TESS и Kepler
- **🎯 ExoFOP-TESS**: TESS Objects of Interest
- **🇪🇺 ESA Archive**: Данные Европейского космического агентства

## 🧠 ML пайплайн

### 1. 🔧 Предобработка данных
- Фильтрация по quality flags
- Удаление выбросов sigma-clipping
- Сглаживание Savitzky-Golay
- Wavelet denoising (10 типов)
- Нормализация и центрирование

### 2. 📈 Извлечение признаков
- 50+ признаков временных рядов
- Транзитные параметры
- Частотный анализ
- Морфологические характеристики
- Метрики качества данных

### 3. 🤖 Модели классификации
- **LightGBM**: Gradient boosting для табличных признаков
- **Random Forest**: Устойчивый ансамбль
- **1D-CNN**: Глубокое обучение для временных рядов
- **Ensemble**: Голосующий классификатор

## 📊 Мониторинг

### Доступные метрики
- Скорость и качество инжеста данных
- Статистика кандидатов
- Метрики производительности моделей
- Использование системных ресурсов
- Время ответа API

## 🔧 Устранение неполадок

### Частые проблемы

1. **Backend не запускается**
   - Проверьте доступность порта 8001
   - Убедитесь, что Python зависимости установлены
   - Проверьте backend.log на ошибки

2. **Инжест данных не работает**
   - Проверьте интернет-соединение
   - Проверьте доступность NASA API
   - Просмотрите логи инжеста

### Логи и отладка

```bash
# Просмотр логов backend
tail -f backend.log

# Проверка статуса пайплайна
python backend/cli/auto_discovery_cli.py status

# Статистика инжеста
curl http://localhost:8001/api/v1/auto-discovery/data-ingestion/stats
```

## 🚀 Оптимизация производительности

### Рекомендуемые настройки

**Для разработки**:
```bash
CONFIDENCE_THRESHOLD=0.75
MAX_CONCURRENT_TASKS=3
CHECK_INTERVAL_HOURS=12
```

**Для production**:
```bash
CONFIDENCE_THRESHOLD=0.85
MAX_CONCURRENT_TASKS=5
CHECK_INTERVAL_HOURS=6
```

## 🔒 Безопасность

- JWT аутентификация для доступа к API
- Ролевой контроль доступа
- Rate limiting на публичных эндпоинтах
- Валидация и санитизация входных данных
- Безопасное хранение артефактов моделей

## 📝 Лицензия

MIT License - см. файл LICENSE для деталей.
