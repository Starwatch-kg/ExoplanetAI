#!/bin/bash

# Скрипт для запуска frontend приложения

echo "🚀 Запуск Exoplanet AI Frontend..."

# Переходим в директорию frontend относительно расположения скрипта
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/frontend"

# Проверяем наличие node_modules
if [ ! -d "node_modules" ]; then
    echo "📦 Установка зависимостей..."
    npm install
fi

# Запускаем dev сервер
echo "🌐 Запуск React приложения на http://localhost:5173"
npm run dev
