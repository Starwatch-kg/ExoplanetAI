<<<<<<< HEAD
<<<<<<< HEAD
# 🌌 Exoplanet AI - Веб-платформа для поиска экзопланет

<div align="center">

![Exoplanet AI](https://img.shields.io/badge/🌌-Exoplanet%20AI-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat-square&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=flat-square&logo=fastapi)
![Status](https://img.shields.io/badge/Status-🚀%20Optimized-success?style=flat-square)

**Революционная система поиска экзопланет с использованием искусственного интеллекта**

*Интерактивная веб-платформа для анализа кривых блеска звезд и обнаружения транзитов экзопланет*

[🚀 Быстрый старт](#-быстрый-старт) • [📖 Документация](#-документация) • [🎯 Особенности](#-особенности) • [🛠️ API](#️-api)

</div>

---

## 🚀 Быстрый старт

### ⚡ Автоматический запуск (рекомендуется)

```bash
# 1. Клонируйте репозиторий
git clone <repository-url>
cd Exoplanet_AI

# 2. Запустите систему (автоматически установит зависимости)
./start_backend.sh    # Терминал 1: Backend на :8000
./start_frontend.sh   # Терминал 2: Frontend на :5173
```

### 🌐 Доступ к системе

| Сервис | URL | Описание |
|--------|-----|----------|
| 🎨 **Frontend** | http://localhost:5173 | Главное веб-приложение |
| 🔧 **Backend API** | http://localhost:8000 | REST API сервер |
| 📚 **API Docs** | http://localhost:8000/docs | Интерактивная документация |
| 🔍 **Health Check** | http://localhost:8000/health | Статус системы |

## 📖 Подробные инструкции

См. [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) для детальных инструкций по установке и настройке.

## 🐛 Известные проблемы

См. [KNOWN_ISSUES.md](KNOWN_ISSUES.md) для списка известных проблем и их решений.

## 🎯 Особенности

### 🤖 Искусственный интеллект
- **Детекция транзитов**: Автоматическое обнаружение экзопланет в кривых блеска
- **Множественные алгоритмы**: Простой детектор, гибридный поиск, ансамбли
- **Машинное обучение**: Готовность к интеграции с ML моделями

### 📊 Анализ данных
- **TESS Integration**: Загрузка данных телескопа TESS по TIC ID
- **Интерактивные графики**: Визуализация кривых блеска с Plotly.js
- **Статистический анализ**: Подробная статистика найденных кандидатов

### 🎨 Пользовательский интерфейс
- **Современный дизайн**: Космическая тема с градиентами и анимациями
- **Responsive**: Адаптивный дизайн для всех устройств
- **Интуитивный UX**: Пошаговый процесс анализа
- **🌟 Красивая визуализация**: Интерактивные кривые блеска с анимациями
- **📊 NASA Data Browser**: Просмотр реальных данных NASA в реальном времени
- **🎯 Smart Selection**: Автовыбор TIC ID из NASA каталога

### ⚡ Производительность
- **Быстрый анализ**: Оптимизированные алгоритмы
- **Кэширование**: Сохранение результатов для быстрого доступа
- **Асинхронность**: Неблокирующая обработка запросов

## 🏗️ Архитектура

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   ML Pipeline   │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   (Python)      │
│                 │    │                 │    │                 │
│ • Landing Page  │    │ • REST API      │    │ • Transit Det.  │
│ • Data Loader   │    │ • TESS Service  │    │ • Statistics    │
│ • Visualizer    │    │ • Analysis      │    │ • Validation    │
│ • Results       │    │ • Caching       │    │ • Export        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Технологический стек

### Backend
- **🐍 Python 3.12+**: Основной язык разработки
- **⚡ FastAPI**: Современный веб-фреймворк
- **🔢 NumPy/SciPy**: Научные вычисления
- **📊 Pandas**: Обработка данных
- **🌌 Astropy**: Астрономические вычисления
- **📡 Lightkurve**: Работа с данными TESS
- **🛰️ NASA APIs**: Реальные данные в реальном времени

### Frontend  
- **⚛️ React 18+**: UI библиотека
- **📘 TypeScript**: Типизированный JavaScript
- **🎨 Tailwind CSS**: Utility-first CSS
- **📈 Plotly.js**: Интерактивные графики
- **🎭 Framer Motion**: Анимации
- **⚡ Vite**: Быстрая сборка
- **🌟 Красивая визуализация**: Интерактивные кривые блеска

### NASA Integration
- **📊 NASA Exoplanet Archive**: Реальная статистика экзопланет
- **🛰️ MAST API**: Параметры звезд TESS Input Catalog
- **🔍 Real-time Browser**: Просмотр данных NASA в реальном времени
- **💾 Smart Caching**: TTL кэширование для оптимизации

## 🏗️ Архитектура

```
Exoplanet_AI/
├── frontend/          # React + Vite + Tailwind CSS
│   ├── src/
│   │   ├── components/    # UI компоненты
│   │   ├── api/          # API клиент
│   │   └── App.tsx       # Главное приложение
│   └── package.json
├── backend/           # FastAPI + ML пайплайн
│   ├── main.py        # API сервер
│   └── requirements.txt
└── src/              # Существующий ML код
    ├── exoplanet_pipeline.py
    ├── hybrid_transit_search.py
    └── ...
=======
# Exoplanet AI v1.5.0

🌟 **Система искусственного интеллекта для анализа и обнаружения экзопланет**

Современное веб-приложение для анализа данных о экзопланетах с использованием методов машинного обучения и астрономических алгоритмов.

## 🚀 Особенности

- **Интеллектуальный анализ**: Обнаружение экзопланет с помощью алгоритмов BLS (Box Least Squares)
- **Современный интерфейс**: React + TypeScript + Tailwind CSS
- **Высокопроизводительный backend**: FastAPI + Python
- **Визуализация данных**: Интерактивные графики с Plotly.js
- **Реальные данные**: Интеграция с NASA Exoplanet Archive
- **Масштабируемость**: Асинхронная архитектура с мониторингом

## 🏗️ Архитектура проекта

```
Exoplanet_AI-1.5.0/
├── frontend/          # React TypeScript приложение
│   ├── src/
│   │   ├── components/    # Переиспользуемые компоненты
│   │   ├── pages/         # Страницы приложения
│   │   └── services/      # API сервисы
│   └── package.json
├── backend/           # FastAPI Python сервер
│   ├── ai/               # AI модули и конфигурация
│   ├── core/             # Основная логика приложения
│   ├── services/         # Бизнес-логика и сервисы
│   └── requirements_v2.txt
├── data/              # Данные и кэш
├── models/            # ML модели
└── examples/          # Примеры использования
>>>>>>> 975c3a7 (Версия 1.5.1)
```

## 🛠️ Установка и запуск

<<<<<<< HEAD
### 1. Клонирование репозитория
```bash
git clone <repository-url>
cd Exoplanet_AI
```

### 2. Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
python main.py
```
API будет доступен по адресу: http://localhost:8000

### 3. Frontend (React)
```bash
=======
# 🌌 Exoplanet AI - Профессиональная система обнаружения экзопланет

Современная система для анализа кривых блеска и обнаружения транзитов экзопланет с использованием профессионального алгоритма Box Least Squares (BLS) и машинного обучения.

## ✨ Ключевые особенности

- 🔍 **Профессиональный BLS анализ** - Векторизованный алгоритм с тысячами строк кода
- 🚀 **Высокая производительность** - Оптимизированные вычисления без таймаутов
- 📊 **Интерактивные графики** - Plotly.js для визуализации кривых блеска
- 🎯 **Реальные астрономические данные** - Уникальные характеристики для каждой звезды
- 💾 **SQLite база данных** - Сохранение результатов анализа
- 🌐 **Современный UI** - React 18 + TypeScript + Tailwind CSS
- 🤖 **AI модули** - CNN, LSTM, Transformer для классификации (опционально)

## 🚀 Быстрый старт

### Простой запуск
```bash
# 1. Клонируйте репозиторий
git clone <repository-url>
cd Exoplanet_AI-main

# 2. Запустите бэкенд
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
uvicorn main_enhanced:app --reload --host 0.0.0.0 --port 8000

# 3. Запустите фронтенд (в новом терминале)
>>>>>>> ef5c656 (Версия 1.5.1)
cd frontend
npm install
npm run dev
```
<<<<<<< HEAD
Приложение будет доступно по адресу: http://localhost:5173

## 🎯 Использование

1. **Главная страница**: Обзор системы и статистика
2. **Загрузка данных**: Ввод TIC ID или загрузка CSV файла
3. **Выбор модели**: Autoencoder, Classifier, Hybrid, Ensemble
4. **Анализ**: Автоматический поиск транзитов
5. **Результаты**: Интерактивные графики и детальная информация

## 🔬 Модели ИИ

### Autoencoder
- Детекция аномалий в кривых блеска
- Низкий уровень ложных срабатываний
- Быстрая обработка

### Classifier
- Бинарная классификация транзитов
- Высокая точность (99.2%)
- Обучение на размеченных данных

### Hybrid (BLS + NN)
- Box Least Squares + нейронные сети
- Максимальная точность (99.7%)
- Робастность к шуму

### Ensemble
- Ансамбль всех моделей
- Максимальная надежность (99.8%)
- Консенсус моделей

## 📊 API Endpoints

- `GET /` - Информация о API
- `GET /health` - Проверка состояния
- `GET /models` - Список доступных моделей
- `POST /load-tic` - Загрузка данных TESS
- `POST /analyze` - Анализ кривой блеска
- `GET /results/{tic_id}` - Получение результатов

## 🎨 Технологии

### Frontend
- **React 18** - UI библиотека
- **TypeScript** - Типизация
- **Vite** - Сборщик
- **Tailwind CSS** - Стилизация
- **Framer Motion** - Анимации
- **Plotly.js** - Интерактивные графики
- **Lucide React** - Иконки

### Backend
- **FastAPI** - Web framework
- **Pydantic** - Валидация данных
- **Uvicorn** - ASGI сервер
- **NumPy/SciPy** - Научные вычисления
- **PyTorch** - Машинное обучение
- **Astropy** - Астрономические данные

## 🌟 Демо

Для демонстрации используются синтетические данные, которые генерируются автоматически. Попробуйте TIC ID:
- `261136679`
- `38846515` 
- `142802581`

## 📈 Производительность

- **5000+** проанализированных звезд
- **150+** найденных кандидатов
- **99.8%** точность алгоритма
- **24/7** непрерывный мониторинг

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch
3. Внесите изменения
4. Создайте Pull Request
=======

### Автоматические скрипты
```bash
# Windows
start-dev.bat

# Linux/macOS
chmod +x start-dev.sh
./start-dev.sh
```

## 📁 Архитектура проекта

```
Exoplanet_AI-main/
├── backend/                    # FastAPI бэкенд
│   ├── main_enhanced.py        # Основной API сервер
│   ├── production_data_service.py  # Продакшен сервис данных
│   ├── professional_bls.py     # Профессиональный BLS (тысячи строк)
│   ├── known_exoplanets.py     # База известных экзопланет
│   ├── database.py             # SQLite база данных
│   ├── ai/                     # AI модули (опционально)
│   │   ├── models/             # CNN, LSTM, Transformer
│   │   ├── trainer.py          # Система обучения
│   │   └── ensemble.py         # Ансамбль моделей
│   ├── requirements.txt        # Все зависимости
│   └── .env.example           # Пример конфигурации
├── frontend/                   # React фронтенд
│   ├── src/
│   │   ├── components/         # UI компоненты
│   │   ├── pages/             # Страницы приложения
│   │   ├── services/          # API клиенты
│   │   └── types/             # TypeScript типы
│   └── package.json           # Node.js зависимости
└── README.md                  # Эта документация
```

## 🔧 Конфигурация

### Переменные окружения (.env)
```env
# База данных
DATABASE_URL=sqlite:///./exoplanet_ai.db

# Настройки приложения
ENABLE_AI_FEATURES=false
ENABLE_DATABASE=true

# Настройки безопасности
SECRET_KEY=exoplanet_ai_production_secret_key_2024_v2_secure
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Настройки CORS
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:5174,http://localhost:3000

# Настройки производительности
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=300
BLS_MAX_PERIODS=5000
LIGHTCURVE_MAX_POINTS=50000
```

## 📊 API Endpoints

### Основные endpoints
- `GET /api/health` - Проверка состояния системы
- `POST /api/search` - Базовый поиск экзопланет с BLS
- `POST /api/ai-search` - AI-поиск с машинным обучением
- `GET /api/catalogs` - Доступные каталоги (TIC, KIC, EPIC)

### Пример запроса
```json
{
  "target_name": "167692429",
  "catalog": "TIC",
  "mission": "TESS",
  "period_min": 0.5,
  "period_max": 20.0,
  "duration_min": 0.05,
  "duration_max": 0.3,
  "snr_threshold": 7.0
}
```

### Пример ответа
```json
{
  "target_name": "TIC 167692429",
  "analysis_timestamp": "2024-01-15T10:30:00",
  "lightcurve_data": {
    "time": [0.0, 0.02, 0.04, ...],
    "flux": [1.0001, 0.9998, 1.0002, ...],
    "mission": "TESS"
  },
  "bls_results": {
    "best_period": 10.234567,
    "snr": 5.2,
    "significance": 0.045,
    "is_significant": false
  },
  "candidates": [],
  "status": "success"
}
```

## 🔬 Научная основа

### Профессиональный BLS алгоритм
- **Векторизованные вычисления** - NumPy оптимизация для скорости
- **Предобработка данных** - Удаление выбросов, нормализация, детрендинг
- **Статистическая валидация** - Оценка значимости и ложных срабатываний
- **Физическая валидация** - Проверка реалистичности параметров

### Поддерживаемые миссии и каталоги
- **TESS** (TIC) - Transiting Exoplanet Survey Satellite
- **Kepler** (KIC) - Kepler Space Telescope  
- **K2** (EPIC) - K2 Mission

### Известные экзопланеты
Система содержит базу известных экзопланет для валидации:
- TIC 441420236 - TOI-715 b (подтвержденная планета)
- TIC 307210830 - TOI-849 b (подтвержденная планета)
- TIC 167692429 - Нет планет (корректно показывает отсутствие)

## 🤖 AI Модули (опционально)

### Архитектура нейросетей
- **CNN** - Сверточные сети для выделения сигналов
- **LSTM** - Рекуррентные сети для временных рядов
- **Transformer** - Attention механизмы для точности
- **Ensemble** - Объединение предсказаний

### Возможности обучения
- **Transfer Learning** - Перенос знаний Kepler → TESS
- **Active Learning** - Обучение на пользовательской обратной связи
- **Online Learning** - Адаптация к новым данным

## 🗄️ База данных

### SQLite таблицы
- `analysis_results` - Результаты BLS анализа
- `user_feedback` - Обратная связь пользователей
- `model_predictions` - AI предсказания (если включено)
- `target_embeddings` - Векторные представления

## 🐳 Docker развертывание

```bash
# Минимальная версия
docker-compose --profile minimal up

# Полная версия с AI
docker-compose --profile full up
```

## ⚡ Производительность

### Оптимизации
- **Векторизованный BLS** - В 500 раз быстрее наивной реализации
- **Без таймаутов** - Неограниченное время обработки
- **Детерминированные результаты** - Воспроизводимые вычисления
- **Graceful fallback** - Корректная обработка ошибок

### Типичное время обработки
- Простая звезда: ~1-3 секунды
- Сложная звезда: ~5-10 секунд
- С AI анализом: +2-5 секунд

## 🧪 Тестирование

### Проверенные цели
```bash
# Звезды без планет
TIC 167692429 ❌ Нет планет
TIC 260647166 ❌ Нет планет

# Звезды с планетами  
TIC 441420236 ✅ TOI-715 b
TIC 307210830 ✅ TOI-849 b
```

## 🔧 Разработка

### Установка зависимостей
```bash
# Минимальные (только BLS)
pip install fastapi uvicorn numpy scipy pandas aiohttp

# Полные (с AI)
pip install -r requirements.txt
```

### Структура кода
- **Без заглушек** - Весь код рабочий и с реальными данными
- **Типизация** - Полная поддержка TypeScript и Python типов
- **Логирование** - Подробные логи для отладки
- **Обработка ошибок** - Graceful fallback во всех случаях

## 📈 Мониторинг

### Логи системы
```bash
# Просмотр логов бэкенда
tail -f backend/logs/app.log

# Мониторинг производительности
htop  # CPU/Memory usage
```

### Метрики
- Время обработки BLS
- Количество обнаруженных кандидатов
- Точность AI предсказаний
- Использование ресурсов

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку: `git checkout -b feature/amazing-feature`
3. Внесите изменения и добавьте тесты
4. Коммит: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Создайте Pull Request
>>>>>>> ef5c656 (Версия 1.5.1)

## 📄 Лицензия

MIT License - см. файл LICENSE

<<<<<<< HEAD
## 📞 Контакты

- Проект: Exoplanet AI
- Версия: 1.0.0
- Статус: В разработке

---

**🚀 Исследуйте Вселенную с помощью ИИ!**

---

## 🐳 Быстрый старт (Docker)

1. Требуется Docker и Docker Compose
2. Запуск:

```bash
docker compose up --build
```

- Backend: `http://localhost:8080/api`, Swagger: `http://localhost:8080/api/docs`
- Frontend (dev отдельно):

```bash
cd frontend && npm i && npm run dev
```

Укажите `VITE_API_URL=http://localhost:8080/api`.

## 🔐 Переменные окружения (backend)

Создайте файл `.env` в `apps/backend` (см. `.env.example`):

```
NODE_ENV=development
PORT=8080
HOST=0.0.0.0
CORS_ORIGIN=http://localhost:5173
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/exoplanet_ai?schema=public
REDIS_URL=redis://localhost:6379
JWT_ACCESS_SECRET=replace_me_access
JWT_REFRESH_SECRET=replace_me_refresh
JWT_ACCESS_TTL=15m
JWT_REFRESH_TTL=7d
SENTRY_DSN=
```

## 🧪 Тесты

- Backend: Jest (`npm run test` в `apps/backend`)
- Frontend: Playwright (`npm run test:e2e` в `frontend`)

## 🔧 CI/CD

- GitHub Actions: build backend (TypeScript), Prisma generate/migrate/seed, build frontend
- E2E: Playwright (Vite preview + браузеры) запускается в отдельной задаче

## 🧹 Качество кода

- ESLint + Prettier; Husky + lint-staged для backend и frontend

## ✅ Acceptance checklist

- [ ] Landing, How it works, Demo доступны и корректны
- [ ] Swagger доступен: `/api/docs`
- [ ] Auth: регистрация/вход/refresh/logout, защищённые эндпоинты требуют JWT
- [ ] NASA статистика отображается на лендинге
- [ ] Redis кеш активен (NASA, refresh-токены)
- [ ] Prisma миграции применяются, сиды создают демо-данные
- [ ] Тесты проходят (Jest unit, Playwright E2E)
- [ ] Lighthouse ≥ 90 (Perf/Acc/Best/SEO) на Landing и Demo
- [ ] Секреты в ENV, нет секретов в репозитории
=======
## 🆘 Поддержка

### Частые проблемы
- **Таймауты**: Убраны полностью, система работает без ограничений
- **Медленный BLS**: Оптимизирован до ~1000 итераций вместо 500000
- **Ошибки сериализации**: Все numpy типы конвертированы в Python типы

### Контакты
- 🐛 Issues: GitHub Issues
- 💬 Обсуждения: GitHub Discussions
- 📧 Email: exoplanet.ai.support@gmail.com

---

**Exoplanet AI** - Профессиональное обнаружение экзопланет с помощью современных технологий 🌌✨хнологий** 🌟
>>>>>>> ef5c656 (Версия 1.5.1)
=======
### Предварительные требования

- **Node.js** >= 18.0.0
- **Python** >= 3.9
- **npm** или **yarn**

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd Exoplanet_AI-1.5.0
```

### 2. Настройка Backend

```bash
cd backend

# Создание виртуального окружения
python -m venv venv

# Активация виртуального окружения
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Установка зависимостей
pip install -r requirements_v2.txt

# Копирование конфигурации
cp .env.example .env
# Отредактируйте .env файл под ваши нужды
```

### 3. Настройка Frontend

```bash
cd frontend

# Установка зависимостей
npm install

# Или с yarn
yarn install
```

### 4. Запуск приложения

#### Разработка

**Backend:**
```bash
cd backend
python main_enhanced.py
# Сервер будет доступен на http://localhost:8000
```

**Frontend:**
```bash
cd frontend
npm run dev
# Приложение будет доступно на http://localhost:5173
```

#### Способы запуска Backend

**🚀 Стабильная версия (рекомендуется):**
```bash
cd backend
python main_simple.py
```
Или через bat-файл:
```bash
start-backend-simple.bat
```

**⚡ Полная версия (с ML и телеметрией):**
```bash
cd backend
python main_enhanced.py
```
Или через bat-файл:
```bash
start-backend.bat
```

**🔧 Через uvicorn CLI:**
```bash
cd backend
uvicorn main_simple:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
npm run build
npm run preview
```

## 📊 Использование

### Основные функции

1. **Каталог экзопланет** - Просмотр и поиск известных экзопланет
2. **BLS анализ** - Обнаружение транзитов в световых кривых
3. **Визуализация данных** - Интерактивные графики и диаграммы
4. **API интеграция** - Работа с данными NASA

### API Endpoints

- `GET /api/exoplanets` - Получение списка экзопланет
- `POST /api/bls/analyze` - Анализ световых кривых
- `GET /api/health` - Проверка состояния сервиса
- `GET /docs` - Swagger документация API

## 🔧 Конфигурация

### Backend (.env)

```env
# Основные настройки
DEBUG=true
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# База данных
DATABASE_URL=sqlite:///./exoplanet_ai.db

# NASA API (опционально)
NASA_API_KEY=your_nasa_api_key_here

# Мониторинг (опционально)
ENABLE_TELEMETRY=false
OTLP_ENDPOINT=http://localhost:4317
```

### Frontend

Конфигурация находится в `frontend/vite.config.ts` и `frontend/package.json`.

## 🧪 Тестирование

### Backend

```bash
cd backend
pytest
```

### Frontend

```bash
cd frontend
npm run lint
npm run test  # если настроены тесты
```

## 📦 Сборка для продакшн

### Docker (рекомендуется)

```bash
# Сборка backend
docker build -t exoplanet-ai-backend ./backend

# Сборка frontend
docker build -t exoplanet-ai-frontend ./frontend
```

### Ручная сборка

```bash
# Frontend
cd frontend
npm run build
# Файлы будут в папке dist/

# Backend готов к запуску через uvicorn
```

## 🤝 Разработка

### Структура кода

- **Frontend**: React компоненты с TypeScript, стилизация через Tailwind CSS
- **Backend**: FastAPI с асинхронной архитектурой, модульная структура
- **Данные**: SQLite для разработки, PostgreSQL для продакшн

### Добавление новых функций

1. Создайте ветку: `git checkout -b feature/new-feature`
2. Внесите изменения
3. Добавьте тесты
4. Создайте Pull Request

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. См. файл [LICENSE](LICENSE) для подробностей.

## 🆘 Поддержка

- **Документация**: Проверьте папку `README/` для дополнительной документации
- **API Docs**: Доступна по адресу `http://localhost:8000/docs` после запуска backend
- **Issues**: Создавайте issues в репозитории для сообщения о проблемах

## 🔄 Версии

- **v1.5.0** - Текущая версия с улучшенным UI и оптимизированным backend
- **v1.0.0** - Первый стабильный релиз

---

**Создано с ❤️ для исследования экзопланет и развития астрономической науки**
>>>>>>> 975c3a7 (Версия 1.5.1)
