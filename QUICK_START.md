# 🚀 ExoplanetAI v2.0 - Quick Start Guide

## 🎯 Что это?

**ExoplanetAI** - система автоматического обнаружения экзопланет, использующая реальные данные NASA TESS и машинное обучение.

## ⚡ Быстрый запуск

### 1. Backend (FastAPI)
```bash
cd backend
python main.py
# → http://localhost:8001
```

### 2. Frontend (React)
```bash
cd frontend
npm run dev
# → http://localhost:5173
```

### 3. Автоматическое обнаружение
```bash
# Полный пайплайн
./start_auto_discovery.sh

# Или через CLI
python backend/cli/auto_discovery_cli.py pipeline start
```

## 📊 Что уже работает

### ✅ **Реальные данные TESS:**
- **TOI-715**: 17,687 точек, качество 76%
- **TOI-700**: 17,019 точек, качество 95%
- **TOI-1452**: 15,703 точки, качество 85%
- **TOI-849**: 16,736 точек, качество 95%

### ✅ **API эндпоинты:**
- `/api/v1/analyze` - Анализ с реальными данными NASA
- `/api/v1/exoplanets/search` - Поиск экзопланет
- `/api/v1/auto-discovery/*` - Автоматическое обнаружение
- `/docs` - Swagger документация

### ✅ **CLI команды:**
```bash
# Статус
python backend/cli/auto_discovery_cli.py status

# Загрузить данные
python backend/cli/auto_discovery_cli.py ingestion targets 441420236

# Запустить обнаружение
python backend/cli/auto_discovery_cli.py discovery cycle
```

## 🛠️ Технологии

- **Backend**: FastAPI + Python + LightGBM + Redis
- **Frontend**: React + TypeScript + Plotly.js + TailwindCSS
- **Data**: NASA TESS + MAST + Lightkurve
- **ML**: Scikit-learn + TensorFlow + Ensemble methods

## 📁 Структура

```
ExoplanetAI/
├── backend/           # FastAPI API
├── frontend/          # React UI
├── data/raw/tess/     # TESS данные (5.6MB)
├── config/            # Конфигурации
└── docs/              # Документация
```

## 🔧 Конфигурация

Основные переменные в `.env`:
```bash
BACKEND_URL=http://localhost:8001
CONFIDENCE_THRESHOLD=0.85
REDIS_URL=redis://localhost:6379
```

## 📚 Документация

- **PROJECT_AUDIT_REPORT.md** - Полный отчет о проекте
- **AUTO_DISCOVERY_GUIDE.md** - Руководство по автоматическому обнаружению
- **README.md** - Основная документация

## 🎯 Готово к production

Система оптимизирована, очищена и готова к развертыванию:
- ✅ Docker контейнеризация
- ✅ CI/CD готовность
- ✅ Мониторинг и логирование
- ✅ Безопасность (JWT + Rate limiting)
- ✅ Реальные астрономические данные

**Начните с запуска backend и frontend, затем используйте CLI для автоматического обнаружения экзопланет!** 🌟
