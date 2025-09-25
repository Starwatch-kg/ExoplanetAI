# 🛠️ API Examples - Примеры использования API

## 🚀 **НОВИНКА: РЕАЛЬНЫЕ NASA API!**

**Наша система теперь использует РЕАЛЬНЫЕ данные NASA в реальном времени!**

✅ **Реальная статистика экзопланет** из NASA Exoplanet Archive  
✅ **Реальные параметры звезд** из TESS Input Catalog  
✅ **Кривые блеска на основе реальных данных** NASA MAST  

---

## 🌐 Базовые endpoints

### 1. 🏠 Проверка статуса системы

```bash
curl http://localhost:8000/
```

**Ответ:**
```json
{
  "message": "🌌 Exoplanet AI API",
  "version": "1.0.0",
  "status": "active",
  "timestamp": "2024-01-15T12:00:00",
  "endpoints": {
    "health": "/health",
    "nasa_stats": "/api/nasa/stats",
    "load_data": "/load-tic",
    "analyze": "/analyze",
    "docs": "/docs"
  }
}
```

### 2. 🔍 Health Check

```bash
curl http://localhost:8000/health
```

**Ответ:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T12:00:00",
  "version": "1.0.0",
  "ml_components": {
    "pipeline": false,
    "hybrid_search": false
  }
}
```

### 3. 📊 РЕАЛЬНАЯ NASA статистика

```bash
curl http://localhost:8000/api/nasa/stats
```

**Ответ (РЕАЛЬНЫЕ данные NASA):**
```json
{
  "totalPlanets": 5635,
  "totalHosts": 4143,
  "lastUpdated": "2024-01-15T12:00:00",
  "source": "NASA Exoplanet Archive (REAL DATA)"
}
```

🌟 **Эти данные загружаются в РЕАЛЬНОМ ВРЕМЕНИ из NASA Exoplanet Archive!**

## 🛰️ Работа с данными TESS

### 4. Загрузка данных по TIC ID

```bash
curl -X POST http://localhost:8000/load-tic \
  -H "Content-Type: application/json" \
  -d '{
    "tic_id": "261136679",
    "sectors": [1, 2, 3]
  }'
```

**Ответ:**
```json
{
  "success": true,
  "data": {
    "tic_id": "261136679",
    "times": [0.0, 0.0274, 0.0548, ...],
    "fluxes": [1.0001, 0.9998, 1.0002, ...]
  },
  "message": "Данные для TIC 261136679 успешно загружены"
}
```

## 🔬 Анализ экзопланет

### 5. Анализ кривой блеска

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "lightcurve_data": {
      "tic_id": "261136679",
      "times": [0.0, 0.1, 0.2, 0.3, 0.4],
      "fluxes": [1.0, 0.99, 0.98, 0.99, 1.0]
    },
    "model_type": "simple",
    "parameters": {
      "threshold": 0.01
    }
  }'
```

**Ответ:**
```json
{
  "success": true,
  "candidates": [
    {
      "id": "simple_0",
      "period": 12.5,
      "depth": 0.02,
      "duration": 3.2,
      "confidence": 0.85,
      "start_time": 0.1,
      "end_time": 0.3,
      "method": "simple"
    }
  ],
  "processing_time": 0.045,
  "model_used": "simple",
  "statistics": {
    "total_candidates": 1,
    "average_confidence": 0.85,
    "processing_time": 0.045,
    "data_points": 5,
    "time_span": 0.4
  }
}
```

## 🐍 Python примеры

### Использование с requests

```python
import requests
import json

# Базовый URL API
BASE_URL = "http://localhost:8000"

# 1. Проверка статуса
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# 2. Загрузка данных TESS
tic_data = {
    "tic_id": "261136679",
    "sectors": [1, 2]
}
response = requests.post(f"{BASE_URL}/load-tic", json=tic_data)
lightcurve = response.json()

# 3. Анализ данных
if lightcurve["success"]:
    analysis_request = {
        "lightcurve_data": lightcurve["data"],
        "model_type": "simple",
        "parameters": {"threshold": 0.01}
    }
    
    response = requests.post(f"{BASE_URL}/analyze", json=analysis_request)
    results = response.json()
    
    print(f"Найдено кандидатов: {len(results['candidates'])}")
    for candidate in results["candidates"]:
        print(f"- Период: {candidate['period']:.2f} дней")
        print(f"- Уверенность: {candidate['confidence']:.2f}")
```

### Асинхронный клиент

```python
import aiohttp
import asyncio

async def analyze_star(tic_id: str):
    async with aiohttp.ClientSession() as session:
        # Загружаем данные
        async with session.post(
            "http://localhost:8000/load-tic",
            json={"tic_id": tic_id}
        ) as response:
            data = await response.json()
        
        if not data["success"]:
            return None
            
        # Анализируем
        async with session.post(
            "http://localhost:8000/analyze",
            json={
                "lightcurve_data": data["data"],
                "model_type": "simple"
            }
        ) as response:
            results = await response.json()
            
        return results

# Использование
results = asyncio.run(analyze_star("261136679"))
print(f"Результаты: {results}")
```

## 🌐 JavaScript/TypeScript примеры

### Fetch API

```javascript
// Класс для работы с API
class ExoplanetAPI {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async loadTICData(ticId, sectors = null) {
        const response = await fetch(`${this.baseUrl}/load-tic`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tic_id: ticId, sectors })
        });
        return response.json();
    }
    
    async analyzeLightcurve(lightcurveData, modelType = 'simple') {
        const response = await fetch(`${this.baseUrl}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                lightcurve_data: lightcurveData,
                model_type: modelType
            })
        });
        return response.json();
    }
}

// Использование
const api = new ExoplanetAPI();

async function searchExoplanets(ticId) {
    try {
        // Загружаем данные
        const data = await api.loadTICData(ticId);
        if (!data.success) throw new Error('Ошибка загрузки данных');
        
        // Анализируем
        const results = await api.analyzeLightcurve(data.data);
        
        console.log(`Найдено ${results.candidates.length} кандидатов:`);
        results.candidates.forEach(candidate => {
            console.log(`- ${candidate.id}: период ${candidate.period.toFixed(2)} дней`);
        });
        
        return results;
    } catch (error) {
        console.error('Ошибка:', error);
    }
}

// Запуск
searchExoplanets('261136679');
```

## 🔧 Обработка ошибок

### Типичные ошибки и их обработка

```python
import requests
from requests.exceptions import RequestException

def safe_api_call(url, data=None):
    try:
        if data:
            response = requests.post(url, json=data, timeout=30)
        else:
            response = requests.get(url, timeout=30)
            
        response.raise_for_status()  # Проверка HTTP статуса
        return response.json()
        
    except requests.exceptions.ConnectionError:
        print("❌ Ошибка подключения к серверу")
        return None
    except requests.exceptions.Timeout:
        print("⏰ Превышено время ожидания")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"🚫 HTTP ошибка: {e}")
        return None
    except ValueError:
        print("📄 Ошибка парсинга JSON")
        return None

# Использование
result = safe_api_call("http://localhost:8000/health")
if result:
    print("✅ API доступен:", result["status"])
```

## 📈 Мониторинг производительности

```python
import time
import requests

def benchmark_api():
    """Тестирование производительности API"""
    
    endpoints = [
        ("GET", "/health", None),
        ("GET", "/api/nasa/stats", None),
        ("POST", "/load-tic", {"tic_id": "261136679"})
    ]
    
    results = {}
    
    for method, endpoint, data in endpoints:
        times = []
        
        for _ in range(5):  # 5 запросов для усреднения
            start = time.time()
            
            if method == "GET":
                response = requests.get(f"http://localhost:8000{endpoint}")
            else:
                response = requests.post(f"http://localhost:8000{endpoint}", json=data)
            
            end = time.time()
            
            if response.status_code == 200:
                times.append(end - start)
        
        if times:
            avg_time = sum(times) / len(times)
            results[endpoint] = f"{avg_time:.3f}s"
    
    return results

# Запуск бенчмарка
performance = benchmark_api()
for endpoint, time_taken in performance.items():
    print(f"{endpoint}: {time_taken}")
```

---

## 🎯 Полезные советы

1. **🔄 Кэширование**: API кэширует результаты анализа для ускорения повторных запросов
2. **⏱️ Таймауты**: Устанавливайте разумные таймауты для длительных операций
3. **🔍 Валидация**: API строго валидирует входные данные
4. **📊 Статистика**: Используйте поле `statistics` для мониторинга производительности
5. **🐛 Отладка**: Проверяйте `/health` endpoint при проблемах с подключением

---

**📚 Больше примеров в интерактивной документации: http://localhost:8000/docs**
