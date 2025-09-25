@echo off
echo ================================================================================
echo 🚀 EXOPLANET AI v2.0 - BACKEND SERVER
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

REM Проверяем наличие main_enhanced.py
if not exist "main_enhanced.py" (
    echo ❌ Файл main_enhanced.py не найден!
    pause
    exit /b 1
)

echo ✅ Python найден
echo ✅ Файл main_enhanced.py найден
echo.
echo 🌐 Сервер будет доступен по адресу: http://localhost:8000
echo 📊 API документация: http://localhost:8000/docs
echo 🔍 API endpoints: http://localhost:8000/api/v1/
echo.
echo Для остановки сервера нажмите Ctrl+C
echo.
echo ================================================================================
echo 🚀 ЗАПУСК СЕРВЕРА...
echo ================================================================================

REM Запускаем сервер
python main_enhanced.py

echo.
echo ================================================================================
echo 🛑 СЕРВЕР ОСТАНОВЛЕН
echo ================================================================================
pause
