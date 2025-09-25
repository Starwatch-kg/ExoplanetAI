# 🚀 Быстрый запуск тестов ExoplanetAI

## Локальная проверка проекта

```bash
# Сделать скрипт исполняемым
chmod +x check_project.sh

# Запустить полную проверку
./check_project.sh
```

## Ручной запуск отдельных инструментов

```bash
cd backend

# Установка зависимостей
pip install -r requirements.txt

# Линтеры
black --check --diff .
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
mypy . --ignore-missing-imports
bandit -r . -f json -o bandit-report.json

# Тесты
pytest tests/ -v --tb=short
```

## CI/CD

- **GitHub Actions**: `.github/workflows/ci-python.yml`
- **Локальный скрипт**: `check_project.sh`

## Что проверяется

1. **Black** - форматирование кода
2. **Flake8** - синтаксические ошибки
3. **MyPy** - типизация
4. **Bandit** - уязвимости безопасности
5. **Pytest** - функциональные тесты

## Исправленные проблемы

✅ Убраны дублирующиеся функции поиска планет
✅ Удалены неиспользуемые импорты
✅ Исправлены пороги обнаружения транзитов
✅ Удалены заглушки ML компонентов
✅ Добавлены реальные тесты
✅ Создан CI/CD pipeline
