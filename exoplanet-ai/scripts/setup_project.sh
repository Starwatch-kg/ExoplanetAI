#!/bin/bash

# ExoplanetAI Project Setup Script
# Скрипт настройки проекта ExoplanetAI

set -e  # Exit on any error

echo "🚀 Настройка проекта ExoplanetAI..."
echo "==================================="

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

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Создание виртуального окружения..."
    python3 -m venv venv
    print_success "Виртуальное окружение создано"
else
    print_warning "Виртуальное окружение уже существует"
fi

# Activate virtual environment
print_status "Активация виртуального окружения..."
source venv/bin/activate

# Upgrade pip
print_status "Обновление pip..."
pip install --upgrade pip

# Install backend dependencies
if [ -f "Exoplanet_AI/exoplanet-ai/backend/requirements.txt" ]; then
    print_status "Установка зависимостей backend..."
    pip install -r Exoplanet_AI/exoplanet-ai/backend/requirements.txt
    print_success "Зависимости backend установлены"
elif [ -f "backend/requirements.txt" ]; then
    print_status "Установка зависимостей backend..."
    pip install -r backend/requirements.txt
    print_success "Зависимости backend установлены"
else
    print_error "Файл requirements.txt не найден"
    exit 1
fi

# Install frontend dependencies
if [ -f "Exoplanet_AI/exoplanet-ai/frontend/package.json" ]; then
    print_status "Проверка наличия npm..."
    if command -v npm >/dev/null 2>&1; then
        print_status "Установка зависимостей frontend..."
        cd Exoplanet_AI/exoplanet-ai/frontend
        npm install
        cd ../..
        print_success "Зависимости frontend установлены"
    else
        print_warning "npm не найден в системе"
        print_warning "Вы можете использовать Docker для запуска frontend:"
        print_warning "  docker-compose up -d"
        print_warning "Или установить Node.js и npm вручную"
        print_warning "Ссылка для установки Node.js: https://nodejs.org/"
    fi
elif [ -f "frontend/package.json" ]; then
    print_status "Проверка наличия npm..."
    if command -v npm >/dev/null 2>&1; then
        print_status "Установка зависимостей frontend..."
        cd frontend
        npm install
        cd ..
        print_success "Зависимости frontend установлены"
    else
        print_warning "npm не найден в системе"
        print_warning "Вы можете использовать Docker для запуска frontend:"
        print_warning "  docker-compose up -d"
        print_warning "Или установить Node.js и npm вручную"
        print_warning "Ссылка для установки Node.js: https://nodejs.org/"
    fi
else
    print_warning "Файл frontend/package.json не найден"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Создание файла .env..."
    cat > .env << EOF
# ExoplanetAI Configuration
# Конфигурация ExoplanetAI

# API Configuration
API_V1_STR=/api/v1
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# Database
DATABASE_URL=sqlite:///./exoplanet_ai.db
REDIS_URL=redis://localhost:6379/0

# External APIs
NASA_API_KEY=your-nasa-api-key
MAST_API_TOKEN=your-mast-token

# AI Configuration
ENABLE_AI_FEATURES=true
MODEL_CACHE_DIR=./ml_models
MAX_MODEL_MEMORY_MB=2048

# Monitoring
SENTRY_DSN=your-sentry-dsn
LOG_LEVEL=INFO

# Performance
MAX_REQUEST_SIZE=10485760
REQUEST_TIMEOUT=300

# Security
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000
EOF
    print_success "Файл .env создан"
else
    print_warning "Файл .env уже существует"
fi

# Create logs directory
mkdir -p logs
print_success "Директория logs создана"

# Create data directories
mkdir -p temp_data ml_models
print_success "Директории данных созданы"

print_success "Настройка проекта завершена!"
echo ""
echo "Чтобы активировать виртуальное окружение, выполните:"
echo "  source venv/bin/activate"
echo ""
echo "Чтобы запустить backend:"
echo "  cd Exoplanet_AI/exoplanet-ai/backend && uvicorn main:app --host 0.0.0.0 --port 8001 --reload"
echo ""
echo "Чтобы запустить frontend (если установлен npm):"
echo "  cd Exoplanet_AI/exoplanet-ai/frontend && npm run dev"
echo ""
echo "Чтобы запустить всё с помощью Docker (рекомендуется):"
echo "  docker-compose up -d"
echo ""
echo "Если npm не установлен, вы можете:"
echo "  1. Установить Node.js и npm вручную с https://nodejs.org/"
echo "  2. Использовать Docker для запуска frontend"
echo "  3. Запустить только backend и использовать API напрямую"