#!/bin/bash

# ExoplanetAI Startup Script
# Скрипт запуска ExoplanetAI

set -e

echo "🚀 Запуск ExoplanetAI..."
echo "========================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    print_error "Пожалуйста, запустите этот скрипт из корневой директории проекта"
    exit 1
fi

# Activate virtual environment
if [ -d "venv" ]; then
    print_status "Активация виртуального окружения..."
    source venv/bin/activate
    print_success "Виртуальное окружение активировано"
else
    print_warning "Виртуальное окружение не найдено, используем системный Python"
fi

# Function to check if a process is running
is_running() {
    pgrep -f "$1" > /dev/null
}

# Function to kill a process
kill_process() {
    pkill -f "$1" 2>/dev/null || true
}

# Stop any existing processes
print_status "Остановка существующих процессов..."
kill_process "uvicorn.*main:app"
kill_process "npm.*run.*dev"
print_success "Существующие процессы остановлены"

# Start backend in background
print_status "Запуск backend сервера..."
cd Exoplanet_AI/exoplanet-ai/backend
nohup uvicorn main:app --host 0.0.0.0 --port 8001 --reload > ../../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ../..

# Wait a moment for backend to start
sleep 3

# Check if backend started successfully
if is_running "uvicorn.*main:app"; then
    print_success "Backend сервер запущен (PID: $BACKEND_PID)"
else
    print_error "Не удалось запустить backend сервер"
    exit 1
fi

# Start frontend in background (if npm is available)
print_status "Проверка наличия npm для запуска frontend..."
if command -v npm >/dev/null 2>&1; then
    if [ -d "Exoplanet_AI/exoplanet-ai/frontend" ]; then
        print_status "Запуск frontend сервера..."
        cd Exoplanet_AI/exoplanet-ai/frontend
        nohup npm run dev > ../../logs/frontend.log 2>&1 &
        FRONTEND_PID=$!
        cd ../..
        
        # Wait a moment for frontend to start
        sleep 3
        
        # Check if frontend started successfully
        if is_running "npm.*run.*dev"; then
            print_success "Frontend сервер запущен (PID: $FRONTEND_PID)"
        else
            print_warning "Не удалось запустить frontend сервер"
            print_warning "Вы можете запустить frontend вручную с помощью Docker:"
            print_warning "  docker-compose up -d"
        fi
    else
        print_warning "Директория frontend не найдена"
        print_warning "Запускаем только backend"
    fi
else
    print_warning "npm не найден, запускаем только backend"
    print_warning "Вы можете:"
    print_warning "  1. Установить Node.js и npm вручную"
    print_warning "  2. Использовать Docker для запуска frontend:"
    print_warning "     docker-compose up -d"
    print_warning "  3. Работать только с API backend"
fi

print_success "Сервисы запущены!"
echo ""
echo "Доступ к приложению:"
echo "  🌐 Frontend: http://localhost:5173 (если запущен)"
echo "  🔧 Backend API: http://localhost:8001/api/v1"
echo "  🩺 Health Check: http://localhost:8001/health"
echo ""
echo "Если frontend не запущен, вы можете:"
echo "  1. Установить Node.js и npm и перезапустить скрипт"
echo "  2. Использовать Docker: docker-compose up -d"
echo "  3. Работать только с API backend"
echo ""
echo "Логи:"
echo "  📄 Backend: logs/backend.log"
if command -v npm >/dev/null 2>&1 && [ -d "Exoplanet_AI/exoplanet-ai/frontend" ]; then
    echo "  📄 Frontend: logs/frontend.log"
fi
echo ""
echo "Для остановки сервисов используйте: ./scripts/stop_all.sh"

# Function to cleanup on exit
cleanup() {
    echo ""
    print_status "Остановка сервисов..."
    kill_process "uvicorn.*main:app"
    kill_process "npm.*run.*dev"
    print_success "Все сервисы остановлены"
    exit 0
}

# Trap SIGINT and SIGTERM
trap cleanup SIGINT SIGTERM

# Keep script running
while true; do
    sleep 1
done