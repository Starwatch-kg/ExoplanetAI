@echo off
echo ========================================
echo 🌌 Exoplanet AI v2.0 - Enhanced Startup
echo ========================================

:: Проверяем Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не найден. Установите Python 3.8+
    pause
    exit /b 1
)

:: Проверяем Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js не найден. Установите Node.js 16+
    pause
    exit /b 1
)

echo ✅ Проверка зависимостей пройдена

:: Создаем директории
if not exist "backend\logs" mkdir "backend\logs"
if not exist "backend\cache" mkdir "backend\cache"

echo 📁 Директории созданы

:: Запуск Backend
echo.
echo 🚀 Запуск Backend (FastAPI)...
cd backend

:: Активируем виртуальное окружение если есть
if exist "venv\Scripts\activate.bat" (
    echo 🐍 Активация виртуального окружения...
    call venv\Scripts\activate.bat
)

:: Устанавливаем зависимости если нужно
if not exist "venv" (
    echo 📦 Создание виртуального окружения...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo 📦 Установка зависимостей...
    pip install -r requirements_v2.txt
)

:: Копируем конфигурацию если нет .env
if not exist ".env" (
    echo ⚙️ Создание конфигурации...
    copy ".env.production" ".env"
)

:: Запускаем новое приложение
echo 🎯 Запуск улучшенного сервера...
start "Exoplanet AI Backend" cmd /k "python app.py"

cd ..

:: Запуск Frontend
echo.
echo 🎨 Запуск Frontend (React + Vite)...
cd frontend

:: Устанавливаем зависимости если нужно
if not exist "node_modules" (
    echo 📦 Установка зависимостей...
    npm install
)

:: Запускаем фронтенд
echo 🎯 Запуск фронтенда...
start "Exoplanet AI Frontend" cmd /k "npm run dev"

cd ..

echo.
echo ========================================
echo ✅ Exoplanet AI v2.0 запущен!
echo.
echo 🔗 Доступные URL:
echo   Frontend: http://localhost:5173
echo   Backend API: http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo   Health Check: http://localhost:8000/health
echo.
echo 📊 Новые возможности v2.0:
echo   ✨ Улучшенная архитектура
echo   🚀 Оптимизированная производительность
echo   🔒 Улучшенная безопасность
echo   📈 Кэширование результатов
echo   🎯 Новые API endpoints
echo   📱 Улучшенный UI/UX
echo.
echo 💡 Для остановки закройте окна терминалов
echo ========================================

pause
