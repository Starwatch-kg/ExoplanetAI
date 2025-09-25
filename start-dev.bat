@echo off
echo ========================================
echo    Exoplanet AI - Development Setup
echo ========================================

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

:: Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
)

echo [1/5] Setting up Python virtual environment...
cd backend
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate

echo [2/5] Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements-simple.txt

echo [3/5] Setting up frontend...
cd ..\frontend
if not exist node_modules (
    echo Installing Node.js dependencies...
    npm install
)

echo [4/5] Creating environment files...
if not exist .env (
    echo VITE_API_URL=http://localhost:8000 > .env
    echo VITE_NODE_ENV=development >> .env
    echo VITE_ENABLE_AI_FEATURES=true >> .env
    echo VITE_ENABLE_EXPORT=true >> .env
    echo VITE_ENABLE_FEEDBACK=true >> .env
    echo Created frontend/.env with default settings
)

cd ..\backend
if not exist .env (
    echo ENABLE_AI_FEATURES=false > .env
    echo ENABLE_DATABASE=true >> .env
    echo DATABASE_URL=sqlite:///./exoplanet_ai.db >> .env
    echo Created backend/.env with SQLite database
)

echo [5/5] Starting services...
echo.
echo ========================================
echo    Starting Exoplanet AI Services
echo ========================================
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Press Ctrl+C in any window to stop servers
echo ========================================
echo.

:: Get current directory
set "PROJECT_DIR=%cd%"

:: Start backend server in new window
echo Starting backend server...
start "Exoplanet AI - Backend Server" cmd /k "cd /d "%PROJECT_DIR%\backend" && call venv\Scripts\activate && echo Backend server starting... && uvicorn main_enhanced:app --reload --host 0.0.0.0 --port 8000"

:: Wait for backend to initialize
echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

:: Start frontend server in new window
echo Starting frontend server...
start "Exoplanet AI - Frontend Server" cmd /k "cd /d "%PROJECT_DIR%\frontend" && echo Frontend server starting... && npm run dev"

:: Wait a bit more for frontend to start
timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo    Both servers are starting up!
echo ========================================
echo.
echo Backend API:     http://localhost:8000
echo Frontend App:    http://localhost:5173
echo API Health:      http://localhost:8000/api/health
echo API Docs:        http://localhost:8000/docs
echo.
echo Close this window or press any key to exit setup
echo (Servers will continue running in separate windows)
echo ========================================

pause >nul
