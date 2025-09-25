# Руководство для разработчиков

## Быстрый старт для новых разработчиков

### Предварительные требования

- **Node.js** >= 18.0.0
- **Python** >= 3.9
- **Git**
- **npm** или **yarn**

### Первоначальная настройка

1. **Клонирование репозитория**
   ```bash
   git clone <repository-url>
   cd Exoplanet_AI-1.5.0
   ```

2. **Настройка Backend**
   ```bash
   cd backend
   
   # Создание виртуального окружения
   python -m venv venv
   
   # Активация (Windows)
   venv\Scripts\activate
   # Активация (Linux/Mac)
   source venv/bin/activate
   
   # Установка зависимостей
   pip install -r requirements.txt
   
   # Копирование конфигурации
   copy .env.example .env
   ```

3. **Настройка Frontend**
   ```bash
   cd ../frontend
   npm install
   ```

### Запуск для разработки

**Backend (в одном терминале):**
```bash
cd backend
venv\Scripts\activate  # Windows
python main_enhanced.py
```

**Frontend (в другом терминале):**
```bash
cd frontend
npm run dev
```

Приложение будет доступно по адресу:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Структура проекта

```
Exoplanet_AI-1.5.0/
├── frontend/              # React TypeScript приложение
│   ├── src/
│   │   ├── components/    # Переиспользуемые компоненты
│   │   ├── pages/         # Страницы приложения
│   │   ├── services/      # API сервисы
│   │   ├── types/         # TypeScript типы
│   │   └── store/         # Состояние приложения (Zustand)
│   ├── package.json       # Зависимости и скрипты
│   └── vite.config.ts     # Конфигурация Vite
├── backend/               # FastAPI Python сервер
│   ├── ai/               # AI модули и конфигурация
│   ├── core/             # Основная логика
│   ├── services/         # Бизнес-логика
│   ├── main_enhanced.py  # Точка входа
│   ├── requirements.txt  # Python зависимости
│   └── .env.example      # Пример конфигурации
├── data/                 # Данные и кэш
├── models/               # ML модели
└── README.md             # Основная документация
```

## Технологический стек

### Frontend
- **React 18** с TypeScript
- **Vite** для сборки
- **Tailwind CSS** для стилизации
- **Framer Motion** для анимаций
- **React Query** для управления состоянием API
- **Zustand** для глобального состояния
- **React Router** для навигации

### Backend
- **FastAPI** для API
- **Python 3.9+**
- **NumPy/SciPy** для научных вычислений
- **Pandas** для обработки данных
- **Uvicorn** как ASGI сервер

## Соглашения по коду

### Frontend (TypeScript/React)
- Используйте функциональные компоненты с хуками
- Именование файлов: `PascalCase.tsx` для компонентов
- Используйте TypeScript типы для всех props и состояний
- Следуйте ESLint правилам проекта

### Backend (Python)
- Следуйте PEP 8
- Используйте type hints
- Документируйте функции с помощью docstrings
- Используйте async/await для асинхронных операций

## Тестирование

### Frontend
```bash
cd frontend
npm run lint          # Проверка линтером
npm run type-check    # Проверка типов TypeScript
```

### Backend
```bash
cd backend
pytest                # Запуск тестов
python -m py_compile main_enhanced.py  # Проверка синтаксиса
```

## Сборка для продакшн

### Frontend
```bash
cd frontend
npm run build
# Файлы будут в папке dist/
```

### Backend
```bash
cd backend
# Backend готов к запуску через uvicorn
uvicorn main_enhanced:app --host 0.0.0.0 --port 8000
```

## Отладка

### Общие проблемы

1. **Ошибки импорта в Python**
   - Убедитесь, что виртуальное окружение активировано
   - Проверьте установку зависимостей: `pip list`

2. **Ошибки TypeScript**
   - Запустите `npm run type-check` для детальной диагностики
   - Проверьте импорты и типы

3. **CORS ошибки**
   - Backend настроен для работы с localhost:5173
   - При изменении портов обновите CORS настройки

### Логирование

- Backend логи: консоль и файл `logs/app.log`
- Frontend: используйте браузерные dev tools

## Добавление новых функций

1. Создайте ветку: `git checkout -b feature/new-feature`
2. Внесите изменения
3. Добавьте тесты (если применимо)
4. Обновите документацию
5. Создайте Pull Request

## Переменные окружения

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
NASA_API_KEY=your_api_key_here

# Мониторинг (опционально)
ENABLE_TELEMETRY=false
```

## Полезные команды

```bash
# Проверка всего проекта
cd backend && python -m py_compile main_enhanced.py
cd ../frontend && npm run type-check

# Очистка кэша
cd frontend && rm -rf node_modules package-lock.json && npm install
cd backend && rm -rf __pycache__ && pip install -r requirements.txt

# Обновление зависимостей
cd frontend && npm update
cd backend && pip install --upgrade -r requirements.txt
```

## Поддержка

- Документация API: http://localhost:8000/docs
- Issues: создавайте в репозитории
- Вопросы: используйте Discussions в GitHub

---

**Удачной разработки! 🚀**
