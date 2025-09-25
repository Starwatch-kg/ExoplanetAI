# 🌌 Exoplanet AI - Advanced Transit Detection System

Современная система искусственного интеллекта для автоматического обнаружения экзопланет в данных космических миссий TESS, Kepler и K2.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![React](https://img.shields.io/badge/react-18+-blue.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.100+-green.svg)

## ✨ Возможности

### 🧠 Искусственный интеллект
- **Ансамбль нейронных сетей**: CNN, LSTM, Transformers для максимальной точности
- **Transfer Learning**: обучение на данных Kepler с адаптацией для TESS
- **Active Learning**: улучшение моделей через пользовательскую обратную связь
- **Embeddings**: быстрый поиск похожих объектов и кэширование результатов

### 📊 Анализ данных
- **BLS алгоритм**: оптимизированный Box Least Squares для поиска транзитов
- **Автоматическая предобработка**: нормализация, удаление выбросов, ресэмплинг
- **Статистическая валидация**: оценка значимости и ложных срабатываний
- **Физические параметры**: расчет радиуса планеты, температуры, обитаемости

### 🎨 Современный интерфейс
- **React 18 + TypeScript**: надежный и типизированный фронтенд
- **Интерактивные графики**: Plotly.js для научной визуализации
- **Адаптивный дизайн**: Tailwind CSS с космической тематикой
- **Плавные анимации**: Framer Motion для улучшенного UX

### 🚀 Производительность
- **Асинхронная обработка**: FastAPI с высокой производительностью
- **Кэширование**: Redis для быстрого доступа к результатам
- **База данных**: PostgreSQL для надежного хранения данных
- **Масштабируемость**: Docker и микросервисная архитектура

## 🏗️ Архитектура

```
Exoplanet_AI/
├── frontend/           # React 18 + TypeScript + Tailwind CSS
│   ├── src/
│   │   ├── components/ # UI компоненты и графики
│   │   ├── pages/      # Страницы приложения
│   │   ├── services/   # API интеграция
│   │   ├── store/      # Управление состоянием (Zustand)
│   │   └── types/      # TypeScript типы
│   └── package.json
├── backend/            # FastAPI + AI модули
│   ├── ai/            # Нейронные сети и ML модели
│   │   ├── models/    # CNN, LSTM, Transformers
│   │   ├── ensemble.py # Ансамбль моделей
│   │   ├── trainer.py  # Обучение моделей
│   │   └── predictor.py # Предсказания
│   ├── services/      # Бизнес-логика
│   │   ├── bls_service.py    # BLS анализ
│   │   ├── data_service.py   # Работа с данными
│   │   └── ai_service.py     # AI сервис
│   └── main.py        # FastAPI приложение
├── docker-compose.yml # Контейнеризация
└── README.md
```

## 🚀 Быстрый старт

### Вариант 1: Автоматический запуск (рекомендуется)

**Windows:**
```bash
./start-dev.bat
```

**Linux/macOS:**
```bash
chmod +x start-dev.sh
./start-dev.sh
```

### Вариант 2: Ручная установка

#### Требования
- Python 3.8+
- Node.js 18+
- Git

#### 1. Клонирование репозитория
```bash
git clone https://github.com/your-username/exoplanet-ai.git
cd exoplanet-ai
```

#### 2. Настройка бэкенда
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements-core.txt
```

#### 3. Настройка фронтенда
```bash
cd ../frontend
npm install
cp .env.example .env
```

#### 4. Запуск сервисов
```bash
# Терминал 1 - Backend
cd backend
python main.py

# Терминал 2 - Frontend
cd frontend
npm run dev
```

### Вариант 3: Docker (для продакшена)

#### Базовая версия (без AI)
```bash
docker-compose up exoplanet-ai
```

#### Полная версия (с AI и PostgreSQL)
```bash
docker-compose --profile full up
```

## 🌐 Доступ к приложению

После запуска приложение будет доступно по адресам:

- **Фронтенд**: http://localhost:5173
- **API документация**: http://localhost:8000/docs
- **API**: http://localhost:8000/api/

## 📖 Использование

### 1. Поиск экзопланет
1. Откройте страницу "Поиск"
2. Введите название звезды (например, "TIC 441420236")
3. Выберите каталог и миссию
4. Настройте параметры поиска (опционально)
5. Включите ИИ-анализ для повышения точности
6. Нажмите "Поиск экзопланет"

### 2. Анализ результатов
- Просмотрите интерактивную кривую блеска
- Изучите фазовые диаграммы кандидатов
- Ознакомьтесь с объяснениями ИИ
- Оцените результат для улучшения системы

### 3. История анализов
- Просмотрите все предыдущие анализы
- Фильтруйте по миссии, статусу, уверенности
- Экспортируйте результаты в CSV/JSON

## 🔧 Конфигурация

### Переменные окружения

**Backend (.env):**
```env
# Основные настройки
ENABLE_AI_FEATURES=true
ENABLE_DATABASE=true
DATABASE_URL=postgresql://user:pass@localhost/exoplanet_ai

# Опциональные
MLFLOW_TRACKING_URI=http://localhost:5000
WANDB_API_KEY=your_wandb_key
```

**Frontend (.env):**
```env
VITE_API_URL=http://localhost:8000
VITE_ENABLE_AI_FEATURES=true
VITE_ENABLE_EXPORT=true
```

### Режимы работы

1. **Минимальный** - только BLS анализ без ИИ
2. **Стандартный** - BLS + базовые ИИ модели
3. **Полный** - все возможности + база данных + мониторинг

## 🧪 API Endpoints

### Основные
- `GET /api/health` - статус системы
- `POST /api/search` - базовый поиск
- `POST /api/ai-search` - ИИ-поиск
- `GET /api/lightcurve/{target}` - данные кривой блеска

### ИИ функции
- `GET /api/ai/explanation/{target}` - объяснение результата
- `POST /api/ai/feedback` - обратная связь
- `GET /api/ai/similar/{target}` - похожие объекты
- `POST /api/ai/retrain` - переобучение модели

Полная документация: http://localhost:8000/docs

## 🔬 Научная основа

### Алгоритмы
- **BLS (Box Least Squares)**: Kovács et al. 2002
- **CNN архитектура**: адаптирована для временных рядов
- **LSTM**: для анализа долгосрочных зависимостей
- **Transformers**: attention механизм для выделения транзитов

### Данные
- **TESS**: 2-минутная и 20-секундная фотометрия
- **Kepler**: долгопериодные наблюдения
- **K2**: расширенная миссия Kepler

### Метрики качества
- **Precision/Recall**: для оценки обнаружения
- **ROC AUC**: общая производительность классификатора
- **Contamination Rate**: доля ложных срабатываний
- **Completeness**: полнота обнаружения

## 🤝 Участие в разработке

### Структура коммитов
```
feat: добавить новую функцию
fix: исправить ошибку
docs: обновить документацию
style: форматирование кода
refactor: рефакторинг
test: добавить тесты
```

### Разработка
1. Форкните репозиторий
2. Создайте ветку для функции
3. Внесите изменения
4. Добавьте тесты
5. Создайте Pull Request

## 📊 Мониторинг и логи

### Метрики
- Производительность моделей
- Время отклика API
- Использование ресурсов
- Пользовательская активность

### Логирование
```python
# Структурированные логи
logger.info("Analysis completed", extra={
    "target": target_name,
    "duration": analysis_time,
    "candidates_found": len(candidates)
})
```

## 🔒 Безопасность

- Валидация входных данных
- Rate limiting для API
- Безопасное хранение конфиденциальных данных
- CORS настройки для фронтенда

## 📈 Производительность

### Оптимизации
- Асинхронная обработка запросов
- Кэширование результатов анализа
- Ленивая загрузка компонентов
- Оптимизация SQL запросов
- Сжатие статических файлов

### Масштабирование
- Горизонтальное масштабирование через Docker
- Балансировка нагрузки с Nginx
- Кэширование с Redis
- CDN для статических файлов

## 📄 Лицензия

MIT License - см. файл [LICENSE](LICENSE) для деталей.

## 🙏 Благодарности

- **NASA MAST** за предоставление данных
- **Astropy** сообщество за астрономические библиотеки
- **Lightkurve** команда за инструменты анализа
- **Open Source** сообщество за используемые библиотеки

## 📞 Поддержка

- 📧 Email: support@exoplanet-ai.org
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/exoplanet-ai/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-username/exoplanet-ai/discussions)
- 📖 Wiki: [Project Wiki](https://github.com/your-username/exoplanet-ai/wiki)

---

**Сделано с ❤️ для астрономического сообщества**
