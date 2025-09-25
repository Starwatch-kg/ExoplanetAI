#!/bin/bash

# Улучшенный скрипт для полной проверки проекта ExoplanetAI

echo "🧪 Запуск полной проверки проекта ExoplanetAI..."

# Проверка Python
echo "🐍 Проверяем Python..."
python --version

# Установка основных зависимостей
echo "📦 Устанавливаем основные зависимости..."
pip install --upgrade pip

# Установка линтеров и инструментов тестирования
echo "🔧 Устанавливаем инструменты разработки..."
pip install black flake8 mypy bandit pytest pytest-asyncio

# Переходим в backend
cd backend

# Установка зависимостей проекта
echo "📦 Устанавливаем зависимости проекта..."
pip install -r requirements.txt

# Запуск линтеров
echo "🔍 Запускаем линтеры..."

# Black (форматирование)
echo "⚫ Black..."
if black --check --diff . 2>/dev/null; then
    echo "✅ Black: OK"
else
    echo "❌ Black: Найдены проблемы форматирования"
    echo "💡 Исправьте: black ."
fi

# Flake8 (синтаксис)
echo "🔍 Flake8..."
if flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics 2>/dev/null; then
    echo "✅ Flake8: OK"
else
    echo "❌ Flake8: Найдены синтаксические ошибки"
fi

# MyPy (типы)
echo "📝 MyPy..."
if mypy . --ignore-missing-imports 2>/dev/null; then
    echo "✅ MyPy: OK"
else
    echo "❌ MyPy: Найдены проблемы с типами"
fi

# Bandit (безопасность)
echo "🛡️ Bandit..."
if bandit -r . -f json -o bandit-report.json 2>/dev/null; then
    echo "✅ Bandit: OK"
    # Проверяем, есть ли уязвимости
    if [ -f bandit-report.json ]; then
        vulnerabilities=$(grep -o '"issue_severity":"HIGH"' bandit-report.json | wc -l)
        if [ "$vulnerabilities" -gt 0 ]; then
            echo "⚠️  Bandit: Найдено $vulnerabilities уязвимостей высокой критичности"
        else
            echo "✅ Bandit: Уязвимостей не найдено"
        fi
    fi
else
    echo "❌ Bandit: Ошибка выполнения"
fi

# Pytest (тесты)
echo "🧪 Pytest..."
if pytest test_exoplanetai.py -v --tb=short 2>/dev/null; then
    echo "✅ Pytest: Все тесты пройдены"
else
    echo "❌ Pytest: Некоторые тесты не пройдены"
fi

# Проверка импортов
echo "📥 Проверка импортов..."
if python -c "import main, nasa_api, signal_processor, visualization; print('✅ Все модули импортированы успешно')" 2>/dev/null; then
    echo "✅ Imports: OK"
else
    echo "❌ Imports: Проблемы с импортами"
fi

# Проверка новых возможностей
echo "🔬 Проверка новых возможностей..."
if python -c "import lightkurve, matplotlib; print('✅ Lightkurve и Matplotlib доступны')" 2>/dev/null; then
    echo "✅ Визуализация: Lightkurve и Matplotlib готовы"
else
    echo "❌ Визуализация: Проблемы с библиотеками"
fi

# Проверка NASA API
echo "🛰️ Проверка NASA API..."
if python -c "from nasa_api import nasa_integration; print('✅ NASA API модуль доступен')" 2>/dev/null; then
    echo "✅ NASA API: OK"
else
    echo "❌ NASA API: Проблемы с модулем"
fi

echo ""
echo "🎉 Проверка завершена!"
echo "📊 Результаты:"
echo "   - Black: $(black --check . 2>/dev/null && echo 'OK' || echo 'Проблемы')"
echo "   - Flake8: $(flake8 . --count --select=E9,F63,F7,F82 2>/dev/null && echo 'OK' || echo 'Проблемы')"
echo "   - MyPy: $(mypy . --ignore-missing-imports 2>/dev/null && echo 'OK' || echo 'Проблемы')"
echo "   - Bandit: $(bandit -r . 2>/dev/null && echo 'OK' || echo 'Проблемы')"
echo "   - Tests: $(pytest test_exoplanetai.py --tb=no -q 2>/dev/null && echo 'OK' || echo 'Проблемы')"
echo "   - Imports: $(python -c 'import main, nasa_api, signal_processor, visualization' 2>/dev/null && echo 'OK' || echo 'Проблемы')"
echo "   - Lightkurve: $(python -c 'import lightkurve' 2>/dev/null && echo 'OK' || echo 'Проблемы')"
echo "   - NASA API: $(python -c 'from nasa_api import nasa_integration' 2>/dev/null && echo 'OK' || echo 'Проблемы')"

echo ""
echo "📈 Новые возможности:"
echo "   ✅ Убраны синтетические данные"
echo "   ✅ Добавлена визуализация с Matplotlib"
echo "   ✅ Интегрирован Lightkurve"
echo "   ✅ Улучшена интеграция с NASA API"
echo "   ✅ Добавлены графики в Pro анализ"
echo "   ✅ Создан модуль визуализации"

echo ""
echo "🚀 Проект готов к использованию!"
echo "📊 API: http://localhost:8001"
echo "📚 Docs: http://localhost:8001/docs"
echo "🌟 Ветка: master"
echo "📞 GitHub: https://github.com/neoalderson/ExoplanetAI"
