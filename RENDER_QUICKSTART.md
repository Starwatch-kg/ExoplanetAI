# 🚀 ExoplanetAI - Render Quick Start

Быстрое развертывание ExoplanetAI на Render за 5 минут.

## ✅ Предварительная проверка

```bash
./check_render_readiness_simple.sh
```

Если все проверки пройдены ✅ - продолжайте!

---

## 📦 Шаг 1: Загрузка на GitHub

```bash
# Если репозиторий еще не создан
git init
git add .
git commit -m "Ready for Render deployment"

# Добавьте remote (замените на ваш URL)
git remote add origin https://github.com/yourusername/exoplanet-ai.git
git push -u origin main
```

---

## ☁️ Шаг 2: Создание сервисов на Render

### Вариант A: Автоматический (Blueprint) ⭐ Рекомендуется

1. Перейдите на https://dashboard.render.com
2. Нажмите **"New +" → "Blueprint"**
3. Подключите GitHub репозиторий
4. Render автоматически обнаружит `render.yaml`
5. Нажмите **"Apply"**

### Вариант B: Ручной

#### Backend Service

1. **New + → Web Service**
2. **Настройки:**
   - Name: `exoplanet-ai-backend`
   - Environment: `Python 3`
   - Build Command: `cd backend && pip install -r requirements.txt`
   - Start Command: `cd backend && gunicorn main:app --config gunicorn_config.py`

#### Redis Service

1. **New + → Redis**
2. **Настройки:**
   - Name: `exoplanet-ai-redis`
   - Plan: `Starter` (Free)

---

## 🔐 Шаг 3: Настройка переменных окружения

В **Backend Service → Environment**:

### Обязательные

| Переменная | Значение |
|------------|----------|
| `ENVIRONMENT` | `production` |
| `LOG_LEVEL` | `INFO` |
| `CACHE_REDIS_URL` | *Из Redis service* |
| `JWT_SECRET_KEY` | *Generate* |
| `ALLOWED_ORIGINS` | `https://exoplanet-ai-frontend.onrender.com` |

### Опциональные (для NASA данных)

| Переменная | Получить |
|------------|----------|
| `NASA_API_KEY` | https://api.nasa.gov/ |

---

## 🎯 Шаг 4: Развертывание

1. **Нажмите "Create Web Service"** (или "Apply" для Blueprint)
2. **Дождитесь завершения build** (~3-5 минут)
3. **Проверьте логи** - должны увидеть:
   ```
   Application startup complete
   Uvicorn running on http://0.0.0.0:PORT
   ```

---

## ✅ Шаг 5: Проверка

### Health Check

```bash
curl https://exoplanet-ai-backend.onrender.com/api/v1/health
```

**Ожидаемый ответ:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "components": {
    "redis": "connected"
  }
}
```

### API Documentation

Откройте в браузере:
```
https://exoplanet-ai-backend.onrender.com/docs
```

---

## 🎨 Опционально: Frontend

### Static Site на Render

1. **New + → Static Site**
2. **Настройки:**
   - Name: `exoplanet-ai-frontend`
   - Build Command: `cd frontend && npm install && npm run build`
   - Publish Directory: `frontend/dist`
3. **Environment:**
   - `VITE_API_URL=https://exoplanet-ai-backend.onrender.com`

---

## 📊 Мониторинг

### Логи
- Dashboard → Service → Logs
- Real-time streaming

### Метрики
- Dashboard → Service → Metrics
- CPU, Memory, Request rate

### Alerts (опционально)
- Dashboard → Service → Settings → Alerts
- Email/Slack уведомления

---

## 🔄 Обновления

### Автоматические

Включите **Auto-Deploy**:
- Settings → Build & Deploy → Auto-Deploy: `Yes`

Теперь каждый `git push` автоматически деплоит новую версию!

### Ручные

```bash
# Локально
git add .
git commit -m "Update feature"
git push origin main

# На Render
Dashboard → Service → Manual Deploy → Deploy Latest Commit
```

---

## 🛠️ Troubleshooting

### Build fails

**Проблема:** `ERROR: Could not install packages`

**Решение:**
- Проверьте Python версию (должна быть 3.11)
- Проверьте `requirements.txt`

### Redis connection error

**Проблема:** `ConnectionError: Error connecting to Redis`

**Решение:**
- Проверьте `CACHE_REDIS_URL` в Environment
- Убедитесь, что Redis service запущен
- Система автоматически использует file cache fallback

### CORS errors

**Проблема:** `blocked by CORS policy`

**Решение:**
- Обновите `ALLOWED_ORIGINS` в Environment
- Добавьте URL фронтенда

---

## 📞 Поддержка

- **Документация:** [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md)
- **Render Docs:** https://render.com/docs
- **GitHub Issues:** https://github.com/yourusername/exoplanet-ai/issues

---

## 🎉 Готово!

Ваше приложение развернуто:

- **Backend API:** https://exoplanet-ai-backend.onrender.com
- **API Docs:** https://exoplanet-ai-backend.onrender.com/docs
- **Frontend:** https://exoplanet-ai-frontend.onrender.com

**Следующие шаги:**
1. Настройте custom domain (опционально)
2. Включите auto-scaling
3. Настройте мониторинг
4. Добавьте CI/CD тесты
