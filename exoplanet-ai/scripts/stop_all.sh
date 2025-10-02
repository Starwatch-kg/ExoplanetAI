#!/bin/bash

# ExoplanetAI Stop Script
# Скрипт остановки ExoplanetAI

set -e

echo "🛑 Остановка ExoplanetAI..."
echo "=========================="

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

# Function to kill a process
kill_process() {
    pkill -f "$1" 2>/dev/null || true
}

# Stop backend processes
print_status "Остановка backend сервера..."
kill_process "uvicorn.*main:app"
print_success "Backend сервер остановлен"

# Stop frontend processes
print_status "Остановка frontend сервера..."
kill_process "npm.*run.*dev"
print_success "Frontend сервер остановлен"

# Stop any remaining processes
print_status "Остановка остальных процессов..."
kill_process "python.*main.py"
kill_process "node.*server.js"
print_success "Остальные процессы остановлены"

print_success "Все сервисы остановлены!"
echo ""
echo "Для запуска снова используйте:"
echo "  ./scripts/start_all.sh"
echo ""
echo "Или с помощью Docker:"
echo "  docker-compose up -d"