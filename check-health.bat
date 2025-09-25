@echo off
echo ========================================
echo   Exoplanet AI - Health Check
echo ========================================
echo.

echo [1/4] Checking Python syntax...
cd backend
python -m py_compile main_enhanced.py
if %errorlevel% neq 0 (
    echo ERROR: Python syntax check failed!
    pause
    exit /b 1
)
echo ✓ Python syntax OK

echo.
echo [2/4] Checking TypeScript compilation...
cd ..\frontend
call npx tsc --noEmit
if %errorlevel% neq 0 (
    echo ERROR: TypeScript compilation failed!
    pause
    exit /b 1
)
echo ✓ TypeScript compilation OK

echo.
echo [3/4] Checking dependencies...
if not exist node_modules (
    echo WARNING: Frontend dependencies not installed
    echo Run: cd frontend && npm install
) else (
    echo ✓ Frontend dependencies OK
)

cd ..\backend
if not exist venv (
    echo WARNING: Python virtual environment not found
    echo Run: cd backend && python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt
) else (
    echo ✓ Python environment OK
)

echo.
echo [4/4] Project structure check...
cd ..
if exist README.md (
    echo ✓ README.md exists
) else (
    echo WARNING: README.md missing
)

if exist .gitignore (
    echo ✓ .gitignore exists
) else (
    echo WARNING: .gitignore missing
)

if exist CONTRIBUTING.md (
    echo ✓ CONTRIBUTING.md exists
) else (
    echo WARNING: CONTRIBUTING.md missing
)

echo.
echo ========================================
echo   Health Check Complete!
echo ========================================
echo.
echo To start the application:
echo 1. Backend:  cd backend && venv\Scripts\activate && python main_enhanced.py
echo 2. Frontend: cd frontend && npm run dev
echo.
pause
