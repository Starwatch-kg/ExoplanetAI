@echo off
echo ================================================================================
echo 🚀 EXOPLANET AI - SIMPLE STABLE BACKEND
echo ================================================================================
echo.

REM Переходим в папку backend
cd /d "%~dp0backend"

REM Проверяем наличие Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python не найден! Убедитесь, что Python установлен и добавлен в PATH.
    pause
    exit /b 1
)

REM Проверяем наличие main_simple.py
if not exist "main_simple.py" (
    echo ❌ Файл main_simple.py не найден!
    pause
    exit /b 1
)

echo ✅ Python найден
echo ✅ Файл main_simple.py найден
echo.
echo 🌐 Сервер будет доступен по адресу: http://localhost:8000
echo 📊 API документация: http://localhost:8000/docs
echo 🔍 API endpoints: http://localhost:8000/api/v1/
echo 🧪 CORS Test: http://localhost:8000/api/v1/test-cors
echo.
echo ⚡ СТАБИЛЬНАЯ ВЕРСИЯ - без сложных зависимостей
echo.
echo Для остановки сервера нажмите Ctrl+C
echo.
echo ================================================================================
echo 🚀 ЗАПУСК СТАБИЛЬНОГО СЕРВЕРА...
echo ================================================================================

REM Запускаем стабильный сервер
python main_simple.py

echo.
echo ================================================================================
echo 🛑 СЕРВЕР ОСТАНОВЛЕН
echo ================================================================================
pause
