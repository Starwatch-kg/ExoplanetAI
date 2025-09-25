@echo off
echo ========================================
echo    Exoplanet AI - Quick System Test
echo ========================================

:: Check if servers are running
echo [1/4] Checking if backend is running...
curl -s http://localhost:8000/api/health >nul 2>&1
if errorlevel 1 (
    echo ❌ Backend server is not running on http://localhost:8000
    echo Please run start-dev.bat first
    pause
    exit /b 1
) else (
    echo ✅ Backend server is running
)

echo [2/4] Checking if frontend is running...
curl -s http://localhost:5173 >nul 2>&1
if errorlevel 1 (
    echo ❌ Frontend server is not running on http://localhost:5173
    echo Please run start-dev.bat first
    pause
    exit /b 1
) else (
    echo ✅ Frontend server is running
)

echo [3/4] Testing API endpoints...
echo Testing /api/health...
curl -s http://localhost:8000/api/health
echo.

echo Testing /api/catalogs...
curl -s http://localhost:8000/api/catalogs
echo.

echo [4/4] Opening application in browser...
start http://localhost:5173
start http://localhost:8000/docs

echo.
echo ========================================
echo    System Test Complete!
echo ========================================
echo Frontend: http://localhost:5173
echo Backend:  http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo ========================================

pause
