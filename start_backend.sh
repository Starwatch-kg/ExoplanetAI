#!/bin/bash

# Скрипт для запуска backend сервера

echo "🚀 Запуск Exoplanet AI Backend..."

# Переходим в директорию backend относительно расположения скрипта
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/backend"

# Проверяем наличие виртуального окружения
if [ ! -d "venv" ]; then
    echo "📦 Создание виртуального окружения..."
    python3 -m venv venv
fi

# Активируем виртуальное окружение
echo "🔧 Активация виртуального окружения..."
source venv/bin/activate

# Устанавливаем зависимости
echo "📥 Установка зависимостей..."
pip install -r requirements.txt

# Запускаем сервер на порту 8001
echo "🌐 Запуск FastAPI сервера на http://localhost:8001"
uvicorn main:app --reload --host 0.0.0.0 --port 8001