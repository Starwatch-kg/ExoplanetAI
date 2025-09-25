# 🚀 Инструкции по запуску Exoplanet AI

## 📋 Системные требования

- **Python**: 3.8+ (рекомендуется 3.11+)
- **RAM**: минимум 8GB (рекомендуется 16GB+)
- **Диск**: минимум 10GB свободного места
- **GPU**: опционально (для ускорения ML)

## 🔧 Установка

### 1. Клонирование репозитория
```bash
git clone <repository-url>
cd Exoplanet_AI
```

### 2. Создание виртуального окружения
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate  # Windows
```

### 3. Установка зависимостей

#### Backend (Python)
```bash
pip install -r backend/requirements.txt
```

#### Frontend (Node.js)
```bash
cd frontend
npm install
cd ..
```

## 🚀 Запуск

### Вариант 1: Веб-платформа (рекомендуется)

#### Терминал 1 - Backend
```bash
source .venv/bin/activate
./start_backend.sh
```

#### Терминал 2 - Frontend
```bash
./start_frontend.sh
```

Откройте http://localhost:5173 в браузере

### Вариант 2: CLI (командная строка)

```bash
source .venv/bin/activate
python exoplanet_search.py --tic-ids 261136679 38846515
```

### Вариант 3: Демо с синтетическими данными

```bash
source .venv/bin/activate
python demo_exoplanet_search.py
```

## 🔍 Тестирование

### Проверка установки
```bash
python -c "import sys; sys.path.append('src'); from src.exoplanet_pipeline import ExoplanetSearchPipeline; print('✅ Установка успешна!')"
```

### Тест веб-платформы
1. Запустите backend и frontend
2. Откройте http://localhost:5173
3. Введите TIC ID: `SYNTHETIC_TRANSIT`
4. Выберите модель и запустите анализ

## 📊 Использование

### Веб-платформа
1. **Dashboard**: Главная страница с описанием
2. **Загрузка данных**: Введите TIC ID или загрузите CSV
3. **Выбор модели**: Autoencoder, Classifier, Hybrid, Ensemble
4. **Анализ**: Запуск поиска экзопланет
5. **Результаты**: Интерактивные графики и таблицы

### CLI
```bash
# Поиск для конкретных звезд
python exoplanet_search.py --tic-ids 261136679 38846515

# С указанием секторов TESS
python exoplanet_search.py --tic-ids 261136679 --sectors 1 2 3

# С обучением моделей
python exoplanet_search.py --tic-ids 261136679 --train-models

# С кастомной конфигурацией
python exoplanet_search.py --tic-ids 261136679 --config custom_config.yaml
```

## 🐛 Известные проблемы и решения

### 1. Ошибка "No module named 'numpy'"
**Проблема**: Зависимости не установлены в виртуальном окружении
**Решение**: 
```bash
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Ошибка "python-cors>=1.7.0"
**Проблема**: Неправильная зависимость в requirements.txt
**Решение**: Уже исправлено в текущей версии

### 3. Недостаточно памяти
**Проблема**: PyTorch с CUDA требует много RAM
**Решение**: 
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Ошибка "No module named 'src'"
**Проблема**: Python не может найти модули src
**Решение**: Запускать из корневой директории проекта

### 5. Frontend не запускается
**Проблема**: Node.js не установлен или зависимости не установлены
**Решение**:
```bash
cd frontend
npm install
npm run dev
```

## 📁 Структура проекта

```
Exoplanet_AI/
├── backend/                 # FastAPI сервер
│   ├── main.py            # Основной файл сервера
│   └── requirements.txt   # Python зависимости
├── frontend/              # React приложение
│   ├── src/
│   │   ├── components/    # UI компоненты
│   │   ├── api/          # API клиент
│   │   └── App.tsx       # Главный компонент
│   └── package.json      # Node.js зависимости
├── src/                   # ML пайплайн
│   ├── exoplanet_pipeline.py
│   ├── hybrid_transit_search.py
│   ├── representation_learning.py
│   └── ...
├── demo_exoplanet_search.py  # Демо скрипт
├── exoplanet_search.py       # CLI скрипт
└── README.md
```

## 🎯 Основные функции

### ML Пайплайн
- **Гибридный поиск**: BLS + Neural Networks
- **Обучение представлений**: Self-supervised learning
- **Детекция аномалий**: Ensemble методов
- **Визуализация**: Интерактивные графики

### Веб-платформа
- **Интерактивный UI**: Современный React интерфейс
- **Реальное время**: Live обновления статуса
- **Графики**: Plotly.js с зумом и наведением
- **Экспорт**: JSON/CSV результаты

## 📞 Поддержка

При возникновении проблем:
1. Проверьте системные требования
2. Убедитесь, что все зависимости установлены
3. Проверьте логи в консоли
4. Создайте issue в репозитории

## 🎉 Готово!

Теперь вы можете:
- Искать экзопланеты через веб-интерфейс
- Использовать CLI для автоматизации
- Анализировать результаты с помощью графиков
- Экспортировать данные в различных форматах

Удачи в поиске новых миров! 🌌✨
