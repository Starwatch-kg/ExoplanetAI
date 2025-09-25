# API Документация

## Обзор

Exoplanet Search API предоставляет RESTful интерфейс для поиска экзопланет с использованием алгоритма BLS (Box-fitting Least Squares). API построен на FastAPI и поддерживает интеграцию с архивами данных TESS, Kepler и K2.

**Base URL**: `http://localhost:8000`

## Аутентификация

В текущей версии API не требует аутентификации. Все endpoints доступны публично.

## Endpoints

### 1. Проверка состояния

#### `GET /api/health`

Проверка состояния API и подключенных сервисов.

**Ответ:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T12:00:00.000Z",
  "services": {
    "bls": "active",
    "data": "active"
  }
}
```

### 2. Поиск экзопланет

#### `POST /api/search`

Выполняет поиск экзопланет для указанной цели с использованием BLS алгоритма.

**Параметры запроса:**
```json
{
  "target_name": "TIC 441420236",
  "catalog": "TIC",
  "mission": "TESS",
  "period_min": 0.5,
  "period_max": 50.0,
  "duration_min": 0.01,
  "duration_max": 0.5,
  "snr_threshold": 7.0
}
```

**Описание параметров:**
- `target_name` (string): Имя цели (TIC ID, KIC ID, EPIC ID)
- `catalog` (string): Каталог ("TIC", "KIC", "EPIC")
- `mission` (string): Миссия ("TESS", "Kepler", "K2")
- `period_min` (float): Минимальный период поиска (дни)
- `period_max` (float): Максимальный период поиска (дни)
- `duration_min` (float): Минимальная длительность транзита (дни)
- `duration_max` (float): Максимальная длительность транзита (дни)
- `snr_threshold` (float): Пороговое значение SNR

**Ответ:**
```json
{
  "target_name": "TIC 441420236",
  "search_timestamp": "2024-01-15T12:00:00.000Z",
  "lightcurve": {
    "target_name": "TIC 441420236",
    "time": [1325.1, 1325.2, ...],
    "flux": [0.999, 1.001, ...],
    "flux_err": [0.001, 0.001, ...],
    "mission": "TESS",
    "sector": 1,
    "duration_days": 27.4,
    "cadence": 0.00139,
    "data_points": 18720
  },
  "bls_result": {
    "periods": [0.5, 0.51, ...],
    "power": [0.001, 0.002, ...],
    "best_period": 3.52,
    "best_power": 0.85,
    "best_t0": 1325.67,
    "best_duration": 0.1,
    "best_depth": 0.00125
  },
  "candidates": [
    {
      "period": 3.52,
      "t0": 1325.67,
      "duration": 2.4,
      "depth": 1250.0,
      "snr": 12.5,
      "sde": 15.2,
      "bls_power": 0.85,
      "planet_radius": 1.2,
      "semi_major_axis": 0.045,
      "equilibrium_temp": 850,
      "false_alarm_probability": 0.001,
      "planet_probability": 0.92
    }
  ],
  "total_candidates": 1,
  "processing_time": 45.2,
  "search_parameters": { ... }
}
```

### 3. Получение кривой блеска

#### `GET /api/lightcurve/{target_name}`

Получение данных кривой блеска для указанной цели без выполнения BLS анализа.

**Параметры URL:**
- `target_name` (string): Имя цели

**Query параметры:**
- `catalog` (string, optional): Каталог (по умолчанию "TIC")
- `mission` (string, optional): Миссия (по умолчанию "TESS")

**Пример запроса:**
```
GET /api/lightcurve/TIC%20441420236?catalog=TIC&mission=TESS
```

**Ответ:**
```json
{
  "target_name": "TIC 441420236",
  "time": [1325.1, 1325.2, ...],
  "flux": [0.999, 1.001, ...],
  "flux_err": [0.001, 0.001, ...],
  "quality": [0, 0, ...],
  "mission": "TESS",
  "sector": 1,
  "duration_days": 27.4,
  "cadence": 0.00139,
  "data_points": 18720
}
```

### 4. Доступные каталоги

#### `GET /api/catalogs`

Получение списка поддерживаемых каталогов и миссий.

**Ответ:**
```json
{
  "catalogs": ["TIC", "KIC", "EPIC"],
  "missions": ["TESS", "Kepler", "K2"],
  "description": {
    "TIC": "TESS Input Catalog",
    "KIC": "Kepler Input Catalog",
    "EPIC": "K2 Ecliptic Plane Input Catalog"
  }
}
```

### 5. Поиск целей

#### `GET /api/targets/search`

Поиск целей по имени или идентификатору в каталоге.

**Query параметры:**
- `query` (string): Поисковый запрос
- `catalog` (string, optional): Каталог для поиска (по умолчанию "TIC")
- `limit` (int, optional): Максимальное количество результатов (по умолчанию 10)

**Пример запроса:**
```
GET /api/targets/search?query=441420236&catalog=TIC&limit=5
```

**Ответ:**
```json
{
  "targets": [
    {
      "target_name": "TIC 441420236",
      "catalog_id": "441420236",
      "coordinates": {
        "ra": 123.456,
        "dec": -12.345
      },
      "magnitude": 12.4,
      "available_data": ["TESS"]
    }
  ]
}
```

### 6. Экспорт результатов

#### `POST /api/export`

Экспорт результатов поиска в CSV или JSON формате.

**Параметры запроса:**
```json
{
  "target": "TIC 441420236",
  "timestamp": "2024-01-15T12:00:00.000Z",
  "candidates": [
    {
      "id": 1,
      "period_days": 3.52,
      "snr": 12.5,
      ...
    }
  ]
}
```

**Query параметры:**
- `format` (string, optional): Формат экспорта ("csv" или "json", по умолчанию "csv")

**Ответ:**
```json
{
  "data": "id,period_days,snr,...\n1,3.52,12.5,...",
  "filename": "exoplanet_search_20240115_120000.csv"
}
```

## Коды ошибок

### HTTP Status Codes

- `200 OK`: Успешный запрос
- `400 Bad Request`: Неверные параметры запроса
- `404 Not Found`: Цель или данные не найдены
- `422 Unprocessable Entity`: Ошибка валидации данных
- `500 Internal Server Error`: Внутренняя ошибка сервера

### Примеры ошибок

**404 - Цель не найдена:**
```json
{
  "detail": "Данные для цели TIC 123456789 не найдены"
}
```

**400 - Неверные параметры:**
```json
{
  "detail": "Минимальный период должен быть меньше максимального"
}
```

**500 - Внутренняя ошибка:**
```json
{
  "detail": "Ошибка при выполнении BLS анализа"
}
```

## Ограничения и лимиты

### Временные ограничения
- **Таймаут запроса**: 5 минут (300 секунд)
- **Время кэширования**: 24 часа для кривых блеска

### Ограничения на параметры
- **Период поиска**: 0.1 - 1000 дней
- **SNR порог**: 3.0 - 15.0
- **Максимальный размер сетки периодов**: 50,000 точек

### Rate Limiting
В текущей версии rate limiting не реализован, но рекомендуется:
- Не более 10 запросов в минуту на поиск
- Не более 100 запросов в час на получение данных

## Примеры использования

### Python

```python
import requests

# Поиск экзопланет
response = requests.post('http://localhost:8000/api/search', json={
    "target_name": "TIC 441420236",
    "catalog": "TIC",
    "mission": "TESS",
    "period_min": 1.0,
    "period_max": 30.0,
    "snr_threshold": 8.0
})

if response.status_code == 200:
    result = response.json()
    print(f"Найдено {result['total_candidates']} кандидатов")
else:
    print(f"Ошибка: {response.json()['detail']}")
```

### JavaScript

```javascript
// Поиск экзопланет
const searchExoplanets = async () => {
  try {
    const response = await fetch('http://localhost:8000/api/search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        target_name: 'TIC 441420236',
        catalog: 'TIC',
        mission: 'TESS',
        period_min: 1.0,
        period_max: 30.0,
        snr_threshold: 8.0
      })
    });

    if (response.ok) {
      const result = await response.json();
      console.log(`Найдено ${result.total_candidates} кандидатов`);
    } else {
      const error = await response.json();
      console.error(`Ошибка: ${error.detail}`);
    }
  } catch (error) {
    console.error('Ошибка сети:', error);
  }
};
```

### cURL

```bash
# Поиск экзопланет
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "target_name": "TIC 441420236",
    "catalog": "TIC",
    "mission": "TESS",
    "period_min": 1.0,
    "period_max": 30.0,
    "snr_threshold": 8.0
  }'

# Получение кривой блеска
curl "http://localhost:8000/api/lightcurve/TIC%20441420236?catalog=TIC&mission=TESS"

# Проверка состояния
curl "http://localhost:8000/api/health"
```

## Интерактивная документация

После запуска сервера доступна интерактивная документация:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Эти интерфейсы позволяют:
- Просматривать все доступные endpoints
- Тестировать API запросы
- Изучать схемы данных
- Скачивать OpenAPI спецификацию
