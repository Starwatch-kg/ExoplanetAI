# 🔧 Исправление проблемы CORS в Exoplanet AI

## 📋 Проблема
Фронтенд получал ошибку CORS при обращении к backend API:
```
Запрос из постороннего источника заблокирован: Политика одного источника запрещает чтение удаленного ресурса на http://localhost:8000/api/v1/bls. (Причина: отсутствует заголовок CORS «Access-Control-Allow-Origin»). Код состояния: 500.
```

## ✅ Решение

### 1. Обновлены настройки CORS в `backend/main_enhanced.py`

**Изменения:**
- Расширен список разрешенных origins
- Добавлены дополнительные HTTP методы (включая OPTIONS и HEAD)
- Расширен список разрешенных заголовков
- Временно добавлен `"*"` для отладки

```python
# Расширенная конфигурация CORS
allowed_origins = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # React dev server
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://localhost:8080",  # Альтернативные порты
    "http://127.0.0.1:8080",
    "http://localhost:4173",  # Vite preview
    "http://127.0.0.1:4173",
    "*"  # Временно для отладки
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=[
        "Accept", "Accept-Language", "Content-Language", "Content-Type",
        "Authorization", "X-Requested-With", "X-Request-ID", "X-Trace-ID",
        "Cache-Control", "Pragma"
    ],
    expose_headers=["X-Request-ID", "X-Trace-ID", "X-Process-Time"]
)
```

### 2. Добавлен специальный обработчик OPTIONS запросов

```python
@app.options("/api/v1/{path:path}")
async def options_handler(request: Request):
    """Обработчик OPTIONS запросов для CORS preflight"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With, X-Request-ID, X-Trace-ID, Cache-Control, Pragma",
            "Access-Control-Max-Age": "86400"
        }
    )
```

### 3. Улучшена обработка ошибок в BLS endpoint

- Добавлена детальная обработка ошибок импорта модулей
- Улучшена диагностика проблем с `enhanced_bls` модулем
- Добавлены информативные сообщения об ошибках

### 4. Добавлен тестовый endpoint для проверки CORS

```python
@app.get("/api/v1/test-cors", tags=["health"])
async def test_cors(request: Request):
    """Простой endpoint для проверки CORS настроек"""
    return {
        "message": "CORS работает!",
        "timestamp": time.time(),
        "origin": request.headers.get("origin", "unknown"),
        "user_agent": request.headers.get("user-agent", "unknown"),
        "method": request.method,
        "url": str(request.url)
    }
```

## 🧪 Тестирование

### Вариант 1: Python скрипт
```bash
cd d:\Vs\Exoplanet_AI-1.5.0
python test_cors.py
```

### Вариант 2: HTML тестер
1. Откройте файл `test_cors.html` в браузере
2. Нажмите кнопки для тестирования различных endpoints
3. Проверьте результаты в интерфейсе

### Вариант 3: Ручное тестирование
```bash
# Тест CORS
curl -H "Origin: http://localhost:5173" \
     -H "Content-Type: application/json" \
     http://localhost:8000/api/v1/test-cors

# Тест BLS endpoint
curl -X POST \
     -H "Origin: http://localhost:5173" \
     -H "Content-Type: application/json" \
     -d '{"target_name":"TIC 123456789","catalog":"TIC","mission":"TESS","period_min":1.0,"period_max":10.0,"snr_threshold":7.0,"use_enhanced":true}' \
     http://localhost:8000/api/v1/bls
```

## 🚀 Запуск системы

1. **Запустите backend:**
```bash
cd d:\Vs\Exoplanet_AI-1.5.0\backend
python main_enhanced.py
```

2. **Запустите frontend:**
```bash
cd d:\Vs\Exoplanet_AI-1.5.0\frontend
npm run dev
```

3. **Проверьте работу:**
   - Откройте http://localhost:5173 (frontend)
   - Попробуйте выполнить поиск экзопланет
   - Проверьте консоль браузера на отсутствие ошибок CORS

## 🔍 Диагностика проблем

### Если CORS все еще не работает:

1. **Проверьте порты:**
   - Backend должен работать на порту 8000
   - Frontend должен работать на порту 5173

2. **Проверьте логи backend:**
   - Ищите сообщения об ошибках импорта
   - Проверьте успешность инициализации CORS middleware

3. **Проверьте браузер:**
   - Откройте Developer Tools (F12)
   - Во вкладке Network проверьте заголовки запросов и ответов
   - Убедитесь, что присутствуют заголовки `Access-Control-Allow-*`

4. **Временное решение:**
   - В настройках CORS уже добавлен `"*"` для всех origins
   - Это должно решить проблему для разработки
   - В продакшене замените на конкретные домены

## 📝 Примечания

- **Безопасность:** В продакшене удалите `"*"` из allowed_origins и укажите конкретные домены
- **Производительность:** CORS preflight запросы кэшируются на 24 часа (`Access-Control-Max-Age: 86400`)
- **Отладка:** Используйте тестовые файлы для быстрой диагностики проблем

## 🎯 Результат

После применения этих изменений:
- ✅ CORS ошибки должны исчезнуть
- ✅ Фронтенд сможет успешно обращаться к API
- ✅ BLS endpoint будет работать корректно
- ✅ Улучшена диагностика ошибок

Если проблемы остаются, используйте тестовые инструменты для детальной диагностики.
