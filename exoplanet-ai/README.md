# ExoplanetAI - Продвинутая система обнаружения экзопланет

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![React 18+](https://img.shields.io/badge/react-18+-61dafb.svg)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ed.svg)](https://docker.com)
[![NASA Space Apps](https://img.shields.io/badge/NASA-Space%20Apps-orange.svg)](https://spaceappschallenge.org/)

> 🚀 **Продвинутая система обнаружения экзопланет с использованием ИИ**

## 🌟 Особенности

- **🤖 ИИ-анализ**: Современные модели машинного обучения для обнаружения транзитов
- **📡 Поддержка миссий**: Интеграция данных TESS, Kepler и K2
- **⚡ Алгоритм BLS**: Оптимизированный Box Least Squares с ансамблевыми методами
- **📊 Интерактивная визуализация**: Современный интерфейс на React с графиками Plotly
- **🔬 Профессиональные инструменты**: Расширенные инструменты анализа для астрономов
- **🌐 Обработка в реальном времени**: Анализ данных и результаты в реальном времени
- **📱 Адаптивный дизайн**: Работает на компьютерах, планшетах и мобильных устройствах
- **🔒 Готов к работе**: Развертывание с помощью Docker с мониторингом

## 🏗️ Архитектура

```
exoplanet-ai/
├── backend/            # Python FastAPI backend
│   ├── config/         # Файлы конфигурации
│   ├── services/       # Сервисы бизнес-логики
│   ├── ml/             # Машинное обучение и модели
│   ├── tests/          # Тесты
│   └── main.py         # Основной файл приложения
├── frontend/           # React/TypeScript frontend
│   ├── src/
│   │   ├── components/ # Переиспользуемые UI компоненты
│   │   ├── pages/      # Страницы приложения
│   │   ├── services/   # API сервисы
│   │   └── types/      # Определения типов TypeScript
│   └── public/         # Статические ресурсы
├── ml/                 # ML алгоритмы и модели
├── docs/               # Документация
├── config/             # Файлы конфигурации
└── scripts/            # Скрипты развертывания
```

## 🚀 Быстрый старт

### Предварительные требования

- **Docker & Docker Compose** (рекомендуется)
- **Python 3.11+** (для разработки)
- **Node.js 18+** (для разработки)
- **Git**

### Вариант 1: Docker (Рекомендуется)

```bash
# Клонирование репозитория
git clone https://github.com/your-org/exoplanet-ai.git
cd exoplanet-ai

# Запуск с помощью Docker Compose
docker-compose up -d

# Доступ к приложению
# Frontend: http://localhost
# Backend API: http://localhost/api/v1
# Проверка состояния: http://localhost/health
```

### Вариант 2: Локальная разработка

```bash
# Настройка backend
cd exoplanet-ai/backend
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Настройка frontend (в новом терминале)
cd exoplanet-ai/frontend
npm install
npm run dev

# Доступ к приложению
# Frontend: http://localhost:5173
# Backend API: http://localhost:8001/api/v1
```

## 📖 Руководство по использованию

### Базовый анализ

1. **Перейдите на страницу поиска**: Используйте меню навигации или перейдите по адресу `/search`
2. **Введите ID цели**: Введите TIC, KIC или EPIC ID (например, "TIC 123456789")
3. **Настройте параметры**: Измените диапазон периодов, длительность и порог SNR
4. **Начните анализ**: Нажмите "Начать поиск" для запуска BLS анализа
5. **Просмотрите результаты**: Изучите кривую блеска и параметры транзита

### Расширенные возможности

- **Фазовое складывание**: Просмотр данных, сложенных по обнаруженному периоду
- **Интерактивные графики**: Масштабирование, панорамирование и исследование данных кривой блеска
- **Опции экспорта**: Скачивание результатов в форматах CSV, JSON или PDF
- **История поиска**: Сохранение и управление историей анализов

### Использование API

```python
import requests

# Проверка состояния
response = requests.get('http://localhost:8001/api/v1/health')

# Поиск экзопланет
search_data = {
    "target_name": "TIC 123456789",
    "catalog": "TIC",
    "mission": "TESS",
    "period_min": 0.5,
    "period_max": 50.0,
    "snr_threshold": 7.0
}

response = requests.post('http://localhost:8001/api/v1/search', json=search_data)
results = response.json()
```

## 🔧 Конфигурация

### Переменные окружения

Создайте файл `.env` в корневой директории проекта:

```env
# Конфигурация API
API_V1_STR=/api/v1
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# База данных
DATABASE_URL=sqlite:///./exoplanet_ai.db
REDIS_URL=redis://localhost:6379/0

# Внешние API
NASA_API_KEY=your-nasa-api-key
MAST_API_TOKEN=your-mast-token

# Конфигурация ИИ
ENABLE_AI_FEATURES=true
MODEL_CACHE_DIR=./ml_models
MAX_MODEL_MEMORY_MB=2048

# Мониторинг
SENTRY_DSN=your-sentry-dsn
LOG_LEVEL=INFO
```

### Конфигурация алгоритма BLS

```python
from ml.bls_ensemble import BLSParameters

# Пользовательские параметры BLS
params = BLSParameters(
    min_period=0.5,
    max_period=100.0,
    duration=0.1,
    frequency_factor=1.0,
    snr_threshold=7.0
)
```

## 🧪 Тестирование

### Запуск тестов

```bash
# Backend тесты
cd exoplanet-ai/backend
pytest tests/ -v

# Frontend тесты
cd exoplanet-ai/frontend
npm test

# Интеграционные тесты
python test_system.py
```

### Тестирование производительности

```bash
# Нагрузочное тестирование
python test_load.py

# Профилирование памяти
python -m memory_profiler test_memory.py
```

## 📊 Мониторинг

### Проверка состояния

- **Состояние приложения**: `GET /health`
- **Состояние базы данных**: `GET /api/v1/health/database`
- **Состояние моделей ИИ**: `GET /api/v1/health/models`

### Метрики

- **Метрики Prometheus**: `GET /metrics`
- **Статистика API**: `GET /api/v1/stats`

## 🚢 Развертывание

### Производственное развертывание

```bash
# Сборка образа
docker build -t exoplanet-ai:latest .

# Запуск с производственной конфигурацией
docker run -d \
  --name exoplanet-ai \
  -p 80:80 \
  -p 8001:8001 \
  -v /path/to/data:/app/data \
  -e DATABASE_URL=postgresql://user:pass@db:5432/exoplanet_ai \
  exoplanet-ai:latest
```

### Развертывание в Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: exoplanet-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: exoplanet-ai
  template:
    metadata:
      labels:
        app: exoplanet-ai
    spec:
      containers:
      - name: exoplanet-ai
        image: exoplanet-ai:latest
        ports:
        - containerPort: 8001
        env:
        - name: DATABASE_URL
          value: "postgresql://..."
---
apiVersion: v1
kind: Service
metadata:
  name: exoplanet-ai-service
spec:
  selector:
    app: exoplanet-ai
  ports:
  - port: 80
    targetPort: 8001
  type: LoadBalancer
```

## 🔬 Наука и алгоритмы

### Алгоритм BLS Ensemble

Наша улучшенная реализация BLS включает:

- **Параллельная обработка**: Оптимизация для многоядерных процессоров
- **Ансамблевые методы**: Комбинации нескольких наборов параметров для надежности
- **Предварительная обработка сигналов**: Расширенная фильтрация и нормализация
- **Статистическая валидация**: Расчет вероятности ложного срабатывания

### Модели машинного обучения

- **Классификатор транзитов**: Модель на основе CNN
- **Оптимизация периода**: Уточнение периода на основе градиентов
- **Характеризация шума**: Автоматическое моделирование шума

## 🤝 Вклад в проект

Мы приветствуем ваш вклад! Смотрите наше [Руководство по внесению вклада](docs/CONTRIBUTING.md).

### Рабочий процесс разработки

1. **Сделайте форк репозитория**
2. **Создайте ветку для функции**: `git checkout -b feature/amazing-feature`
3. **Внесите изменения**
4. **Добавьте тесты**: Убедитесь, что покрытие тестами > 90%
5. **Запустите набор тестов**: `pytest tests/`
6. **Отправьте pull request**

### Стиль кода

- **Python**: Black, isort, mypy
- **TypeScript**: ESLint, Prettier
- **Коммиты**: Соглашение conventional commits

## 📄 Лицензия

Этот проект лицензирован по лицензии MIT - смотрите файл [LICENSE](LICENSE) для подробностей.

## 🙏 Благодарности

- **NASA**: Данные миссий TESS, Kepler и K2
- **Space Apps Challenge**: Вдохновение и сообщество
- **Астрономическое сообщество**: Научное руководство и обратная связь
- **Авторы открытого кода**: Библиотеки и инструменты

## 📞 Поддержка

- **Проблемы**: [GitHub Issues](https://github.com/your-org/exoplanet-ai/issues)
- **Обсуждения**: [GitHub Discussions](https://github.com/your-org/exoplanet-ai/discussions)
- **Документация**: [Wiki](https://github.com/your-org/exoplanet-ai/wiki)

## 🔄 Журнал изменений

Смотрите [CHANGELOG.md](docs/CHANGELOG.md) для истории версий.

---

**Сделано с ❤️ для исследования космоса и научных открытий**
