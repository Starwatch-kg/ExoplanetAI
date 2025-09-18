# 🚀 Быстрый старт - Exoplanet AI

## Установка и запуск за 5 минут

### 1. Запуск Backend (FastAPI)

```bash
# В первом терминале
./start_backend.sh
```

Или вручную:
```bash
cd backend
pip install -r requirements.txt
python main.py
```

Backend будет доступен по адресу: http://localhost:8000

### 2. Запуск Frontend (React)

```bash
# Во втором терминале
./start_frontend.sh
```

Или вручную:
```bash
cd frontend
npm install
npm run dev
```

Frontend будет доступен по адресу: http://localhost:5173

### 3. Использование

1. Откройте http://localhost:5173 в браузере
2. Нажмите "Начать поиск экзопланет"
3. Введите TIC ID (например: `261136679`)
4. Выберите модель ИИ
5. Нажмите "Запустить анализ"
6. Наслаждайтесь результатами!

## 🎯 Демо TIC ID

Попробуйте эти TIC ID для демонстрации:
- `261136679` - Высокая вероятность транзитов
- `38846515` - Средняя активность
- `142802581` - Сложный случай
- `123456789` - Случайные данные

## 🔧 Настройка

### Переменные окружения

Создайте файл `frontend/.env.local`:
```
VITE_API_URL=http://localhost:8000
```

### Изменение портов

**Backend** (в `backend/main.py`):
```python
uvicorn.run("main:app", host="0.0.0.0", port=8000)
```

**Frontend** (в `frontend/vite.config.ts`):
```typescript
export default defineConfig({
  server: { port: 5173 }
})
```

## 🐛 Устранение неполадок

### Backend не запускается
```bash
# Проверьте Python версию
python3 --version  # Должна быть 3.8+

# Установите зависимости
cd backend
pip install -r requirements.txt

# Проверьте логи
python main.py
```

### Frontend не запускается
```bash
# Проверьте Node.js версию
node --version  # Должна быть 16+

# Очистите кэш
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### API недоступен
- Убедитесь, что backend запущен на порту 8000
- Проверьте CORS настройки в `backend/main.py`
- Проверьте URL в `frontend/src/api/exoplanetApi.ts`

## 📊 Структура проекта

```
Exoplanet_AI/
├── frontend/          # React приложение
│   ├── src/
│   │   ├── components/    # UI компоненты
│   │   ├── api/          # API клиент
│   │   └── App.tsx       # Главное приложение
│   └── package.json
├── backend/           # FastAPI сервер
│   ├── main.py        # API эндпоинты
│   └── requirements.txt
├── src/              # ML пайплайн
│   ├── exoplanet_pipeline.py
│   └── ...
├── start_backend.sh  # Скрипт запуска backend
├── start_frontend.sh # Скрипт запуска frontend
└── README.md
```

## 🎨 Особенности UI

- **Космическая тема** - темный фон с звездами
- **Анимации** - плавные переходы и эффекты
- **Интерактивные графики** - Plotly.js с зумом и наведением
- **Адаптивный дизайн** - работает на всех устройствах
- **Реальное время** - статус анализа в реальном времени

## 🔬 Модели ИИ

1. **Autoencoder** - Детекция аномалий (2 сек)
2. **Classifier** - Классификация транзитов (3 сек)
3. **Hybrid** - BLS + нейронные сети (5 сек)
4. **Ensemble** - Все модели вместе (8 сек)

## 📈 API Endpoints

- `GET /health` - Проверка состояния
- `GET /models` - Список моделей
- `POST /load-tic` - Загрузка данных TESS
- `POST /analyze` - Анализ кривой блеска
- `GET /results/{tic_id}` - Результаты анализа

## 🚀 Готово!

Теперь у вас есть полнофункциональная веб-платформа для поиска экзопланет с ИИ!

**Следующие шаги:**
1. Изучите код в `frontend/src/components/`
2. Добавьте новые модели в `backend/main.py`
3. Настройте интеграцию с реальными данными TESS
4. Разверните на сервере для продакшена

**Удачи в исследовании Вселенной! 🌌**