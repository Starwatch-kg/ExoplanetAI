#!/bin/bash

# 🚀 Production Build Script
echo "🌌 Создание production сборки Exoplanet AI..."

# Цвета для вывода
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Проверяем зависимости
echo -e "${BLUE}📦 Проверка зависимостей...${NC}"

# Backend зависимости
echo "Проверка backend зависимостей..."
cd backend
if [ ! -d ".venv" ]; then
    echo "Создание виртуального окружения..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt

# Оптимизация Python кода
echo -e "${BLUE}🐍 Оптимизация Python кода...${NC}"
python -m py_compile main.py nasa_api.py

cd ..

# Frontend сборка
echo -e "${BLUE}⚛️ Сборка frontend...${NC}"
cd frontend

# Установка зависимостей
npm ci --production=false

# Production сборка с оптимизациями
echo "Создание оптимизированной сборки..."
npm run build

# Анализ размера бандла
echo -e "${BLUE}📊 Анализ размера бандла...${NC}"
du -sh dist/

cd ..

# Создание архива
echo -e "${BLUE}📦 Создание архива для деплоя...${NC}"
tar -czf exoplanet-ai-production.tar.gz \
    --exclude=node_modules \
    --exclude=.git \
    --exclude=__pycache__ \
    --exclude=.venv \
    backend/ \
    frontend/dist/ \
    *.md \
    *.sh

echo -e "${GREEN}✅ Production сборка готова!${NC}"
echo -e "${GREEN}📦 Архив: exoplanet-ai-production.tar.gz${NC}"
echo -e "${BLUE}🚀 Готово к деплою!${NC}"
