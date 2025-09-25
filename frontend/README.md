<<<<<<< HEAD
# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default tseslint.config([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      ...tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      ...tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      ...tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default tseslint.config([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
=======
# Exoplanet AI Frontend

Современный веб-интерфейс для системы обнаружения экзопланет с использованием искусственного интеллекта.

## Технологический стек

- **React 18** - современная библиотека для создания пользовательских интерфейсов
- **TypeScript** - типизированный JavaScript для надежности кода
- **Tailwind CSS** - utility-first CSS фреймворк для быстрой стилизации
- **Vite** - быстрый сборщик и dev-сервер
- **Framer Motion** - библиотека для плавных анимаций
- **Plotly.js** - интерактивные научные графики
- **React Query** - управление состоянием сервера и кэширование
- **Zustand** - легковесное управление глобальным состоянием
- **React Hook Form** - эффективная работа с формами

## Возможности

### 🔍 Поиск экзопланет
- Интуитивный интерфейс для поиска по названию звезды
- Поддержка каталогов TIC, KIC, EPIC
- Настройка параметров BLS анализа
- Переключение между обычным и ИИ-анализом

### 📊 Интерактивная визуализация
- Динамические графики кривых блеска с Plotly.js
- Фазовые диаграммы с наложением модели транзита
- Адаптивные графики с возможностью экспорта
- Анимированные переходы и загрузочные состояния

### 🧠 ИИ-анализ
- Визуализация результатов нейронных сетей
- Объяснения решений ИИ на понятном языке
- Оценка уверенности и неопределенности
- Сравнение предсказаний разных моделей

### 📈 Управление результатами
- Таблица с историей всех анализов
- Фильтрация и сортировка результатов
- Экспорт данных в различных форматах
- Система обратной связи для улучшения ИИ

## Быстрый старт

### Установка зависимостей

```bash
cd frontend
npm install
```

### Настройка окружения

Скопируйте файл переменных окружения:

```bash
cp .env.example .env
```

Отредактируйте `.env` файл:

```env
VITE_API_URL=http://localhost:8000
VITE_NODE_ENV=development
VITE_ENABLE_AI_FEATURES=true
```

### Запуск в режиме разработки

```bash
npm run dev
```

Приложение будет доступно по адресу: http://localhost:5173

### Сборка для продакшена

```bash
npm run build
```

Собранные файлы будут в папке `dist/`

### Предварительный просмотр сборки

```bash
npm run preview
```

## Структура проекта

```
frontend/
├── public/                 # Статические файлы
├── src/
│   ├── components/        # React компоненты
│   │   ├── charts/       # Компоненты графиков
│   │   ├── layout/       # Компоненты макета
│   │   └── ui/           # UI компоненты
│   ├── pages/            # Страницы приложения
│   ├── services/         # API сервисы
│   ├── store/            # Управление состоянием
│   ├── types/            # TypeScript типы
│   ├── App.tsx           # Главный компонент
│   ├── main.tsx          # Точка входа
│   └── index.css         # Глобальные стили
├── package.json          # Зависимости и скрипты
├── tailwind.config.js    # Конфигурация Tailwind
├── tsconfig.json         # Конфигурация TypeScript
└── vite.config.ts        # Конфигурация Vite
```

## Основные компоненты

### Страницы
- **HomePage** - главная страница с обзором возможностей
- **SearchPage** - интерфейс поиска экзопланет
- **AnalysisPage** - детальный анализ результатов
- **ResultsPage** - история анализов с фильтрацией
- **AboutPage** - информация о проекте

### Компоненты графиков
- **LightCurveChart** - интерактивный график кривой блеска
- **PhaseFoldedChart** - фазовая диаграмма с моделью транзита

### UI компоненты
- **Navbar** - навигационная панель
- **LoadingSpinner** - индикаторы загрузки
- **Toaster** - система уведомлений

## API интеграция

Приложение взаимодействует с FastAPI бэкендом через следующие endpoints:

- `GET /api/health` - проверка состояния сервера
- `POST /api/search` - базовый поиск экзопланет
- `POST /api/ai-search` - ИИ-улучшенный поиск
- `GET /api/lightcurve/{target}` - получение кривой блеска
- `GET /api/catalogs` - список доступных каталогов
- `POST /api/ai/feedback` - отправка обратной связи

## Стилизация

Проект использует кастомную цветовую схему с космической тематикой:

- **Primary** - оттенки синего для основных элементов
- **Space** - темные оттенки для фона и контейнеров
- **Cosmic** - градиенты розового/фиолетового для акцентов
- **Semantic** - зеленый/красный/желтый для статусов

### Кастомные классы

```css
.btn-primary      /* Основная кнопка */
.btn-secondary    /* Вторичная кнопка */
.btn-cosmic       /* Кнопка с космическим градиентом */
.card             /* Карточка контента */
.input-field      /* Поле ввода */
.text-gradient    /* Градиентный текст */
.glass-effect     /* Эффект матового стекла */
```

## Анимации

Используется Framer Motion для создания плавных анимаций:

- Анимации появления страниц
- Переходы между состояниями
- Интерактивные hover-эффекты
- Загрузочные анимации
- Анимированные частицы на фоне

## Адаптивность

Интерфейс полностью адаптивен и оптимизирован для:

- Десктопных экранов (1920px+)
- Планшетов (768px - 1024px)
- Мобильных устройств (320px - 768px)

## Производительность

- Ленивая загрузка компонентов
- Оптимизация изображений
- Кэширование API запросов с React Query
- Минификация и сжатие в продакшене
- Tree shaking для уменьшения размера бандла

## Разработка

### Линтинг

```bash
npm run lint
```

### Форматирование кода

Рекомендуется использовать Prettier с настройками проекта.

### Отладка

- React DevTools для отладки компонентов
- React Query DevTools для мониторинга запросов
- Zustand DevTools для отслеживания состояния

## Развертывание

### Статический хостинг

Приложение можно развернуть на любом статическом хостинге:

- Netlify
- Vercel
- GitHub Pages
- AWS S3 + CloudFront

### Docker

```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## Лицензия

MIT License - см. файл LICENSE для деталей.

## Поддержка

Для вопросов и предложений создавайте issues в GitHub репозитории.
>>>>>>> ef5c656 (Версия 1.5.1)
