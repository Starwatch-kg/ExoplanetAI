"""
Тесты для ExoplanetAI системы
"""
import pytest
import asyncio
from main import app
from fastapi.testclient import TestClient
import json

client = TestClient(app)

def test_health_check():
    """Тест проверки состояния API"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root_endpoint():
    """Тест корневого эндпоинта"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Exoplanet AI API" in response.json()["message"]

def test_nasa_stats():
    """Тест получения статистики NASA"""
    response = client.get("/api/nasa/stats")
    assert response.status_code == 200
    data = response.json()
    assert "totalPlanets" in data
    assert "totalHosts" in data

def test_load_tic_real_data():
    """Тест загрузки реальных данных TIC"""
    # Тестируем с известным TIC ID, который имеет планету
    response = client.post("/load-tic",
                         json={"tic_id": "TIC 307210830"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "data" in data

def test_load_tic_invalid():
    """Тест загрузки данных с некорректным TIC ID"""
    response = client.post("/load-tic",
                         json={"tic_id": "INVALID"})
    # Должна быть ошибка, так как нет синтетических данных
    assert response.status_code == 500

def test_amateur_analysis():
    """Тест любительского анализа"""
    response = client.post("/amateur/analyze",
                         json={"tic_id": "TIC 307210830"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True

def test_pro_analysis():
    """Тест профессионального анализа"""
    # Используем реальные данные для теста
    test_data = {
        "lightcurve_data": {
            "tic_id": "TIC 307210830",
            "times": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "fluxes": [1.0, 0.99, 1.01, 0.98, 1.02, 1.0, 0.99, 1.01, 0.98, 1.0]
        },
        "model_type": "detector"
    }

    response = client.post("/pro/analyze", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "plots_data" in data
    assert "detailed_analysis" in data

def test_get_latest_analyses():
    """Тест получения последних анализов"""
    response = client.get("/api/latest-analyses")
    assert response.status_code == 200
    data = response.json()
    assert "success" in data

def test_analysis_with_synthetic_data_removed():
    """Тест что синтетические данные действительно удалены"""
    # Проверяем что нет fallback синтетических данных
    response = client.post("/load-tic",
                         json={"tic_id": "NONEXISTENT"})
    assert response.status_code == 500  # Должна быть ошибка, нет синтетики

if __name__ == "__main__":
    # Запуск тестов
    print("🧪 Запуск тестов ExoplanetAI...")

    test_health_check()
    print("✅ Health check passed")

    test_root_endpoint()
    print("✅ Root endpoint passed")

    test_nasa_stats()
    print("✅ NASA stats passed")

    test_load_tic_real_data()
    print("✅ Real TIC data loading passed")

    test_load_tic_invalid()
    print("✅ Invalid TIC handling passed")

    test_amateur_analysis()
    print("✅ Amateur analysis passed")

    test_pro_analysis()
    print("✅ Pro analysis passed")

    test_get_latest_analyses()
    print("✅ Latest analyses passed")

    test_analysis_with_synthetic_data_removed()
    print("✅ Synthetic data removal verified")

    print("\n🎉 Все тесты пройдены успешно!")
    print("📊 ExoplanetAI готов к использованию с реальными данными NASA")
