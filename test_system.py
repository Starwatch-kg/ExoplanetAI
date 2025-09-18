#!/usr/bin/env python3
"""
🧪 Системный тест Exoplanet AI
Проверяет работоспособность всех основных компонентов системы
"""

import requests
import time
import json
from typing import Dict, Any, Optional
import sys

class ExoplanetSystemTester:
    """Класс для тестирования системы Exoplanet AI"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 30
        self.results = {}
    
    def print_status(self, message: str, status: str = "INFO"):
        """Вывод статуса с цветами"""
        colors = {
            "INFO": "\033[94m",      # Синий
            "SUCCESS": "\033[92m",   # Зеленый
            "WARNING": "\033[93m",   # Желтый
            "ERROR": "\033[91m",     # Красный
            "RESET": "\033[0m"       # Сброс
        }
        
        color = colors.get(status, colors["INFO"])
        reset = colors["RESET"]
        
        icons = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌"
        }
        
        icon = icons.get(status, "ℹ️")
        print(f"{color}{icon} {message}{reset}")
    
    def test_endpoint(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Тестирование одного endpoint"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            start_time = time.time()
            
            if method.upper() == "GET":
                response = self.session.get(url)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Неподдерживаемый метод: {method}")
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Проверяем статус код
            response.raise_for_status()
            
            # Парсим JSON
            try:
                json_data = response.json()
            except ValueError:
                json_data = {"raw_response": response.text}
            
            return {
                "success": True,
                "status_code": response.status_code,
                "response_time": response_time,
                "data": json_data,
                "error": None
            }
            
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Ошибка подключения к серверу",
                "status_code": None,
                "response_time": None,
                "data": None
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Превышено время ожидания",
                "status_code": None,
                "response_time": None,
                "data": None
            }
        except requests.exceptions.HTTPError as e:
            return {
                "success": False,
                "error": f"HTTP ошибка: {e}",
                "status_code": response.status_code,
                "response_time": response_time,
                "data": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Неожиданная ошибка: {e}",
                "status_code": None,
                "response_time": None,
                "data": None
            }
    
    def test_basic_endpoints(self):
        """Тестирование базовых endpoints"""
        self.print_status("🏠 Тестирование базовых endpoints", "INFO")
        
        tests = [
            ("GET", "/", "Главная страница"),
            ("GET", "/health", "Health check"),
            ("GET", "/api/nasa/stats", "NASA статистика")
        ]
        
        for method, endpoint, description in tests:
            result = self.test_endpoint(method, endpoint)
            self.results[endpoint] = result
            
            if result["success"]:
                self.print_status(
                    f"{description}: {result['response_time']:.3f}s", 
                    "SUCCESS"
                )
            else:
                self.print_status(
                    f"{description}: {result['error']}", 
                    "ERROR"
                )
    
    def test_data_loading(self):
        """Тестирование загрузки данных"""
        self.print_status("🛰️ Тестирование загрузки данных TESS", "INFO")
        
        test_data = {
            "tic_id": "261136679",
            "sectors": [1, 2]
        }
        
        result = self.test_endpoint("POST", "/load-tic", test_data)
        self.results["/load-tic"] = result
        
        if result["success"]:
            data = result["data"]
            if data.get("success") and "data" in data:
                lightcurve = data["data"]
                points = len(lightcurve.get("times", []))
                self.print_status(
                    f"Загрузка данных: {points} точек за {result['response_time']:.3f}s", 
                    "SUCCESS"
                )
                return lightcurve
            else:
                self.print_status("Загрузка данных: некорректный ответ", "WARNING")
                return None
        else:
            self.print_status(f"Загрузка данных: {result['error']}", "ERROR")
            return None
    
    def test_analysis(self, lightcurve_data: Optional[Dict] = None):
        """Тестирование анализа"""
        self.print_status("🔬 Тестирование анализа экзопланет", "INFO")
        
        if not lightcurve_data:
            # Используем тестовые данные
            lightcurve_data = {
                "tic_id": "TEST",
                "times": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "fluxes": [1.0, 0.99, 0.98, 0.99, 1.0, 0.99]
            }
        
        analysis_request = {
            "lightcurve_data": lightcurve_data,
            "model_type": "simple",
            "parameters": {"threshold": 0.01}
        }
        
        result = self.test_endpoint("POST", "/analyze", analysis_request)
        self.results["/analyze"] = result
        
        if result["success"]:
            data = result["data"]
            if data.get("success"):
                candidates = data.get("candidates", [])
                processing_time = data.get("processing_time", 0)
                self.print_status(
                    f"Анализ: {len(candidates)} кандидатов за {processing_time:.3f}s", 
                    "SUCCESS"
                )
                return data
            else:
                error_msg = data.get("error", "Неизвестная ошибка")
                self.print_status(f"Анализ: {error_msg}", "WARNING")
                return None
        else:
            self.print_status(f"Анализ: {result['error']}", "ERROR")
            return None
    
    def test_performance(self):
        """Тестирование производительности"""
        self.print_status("⚡ Тестирование производительности", "INFO")
        
        # Тестируем health endpoint несколько раз
        times = []
        for i in range(5):
            result = self.test_endpoint("GET", "/health")
            if result["success"]:
                times.append(result["response_time"])
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            self.print_status(
                f"Health endpoint: avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s",
                "SUCCESS" if avg_time < 1.0 else "WARNING"
            )
        else:
            self.print_status("Не удалось измерить производительность", "ERROR")
    
    def run_full_test(self):
        """Запуск полного тестирования"""
        self.print_status("🧪 Запуск полного системного теста", "INFO")
        print("=" * 60)
        
        # 1. Базовые endpoints
        self.test_basic_endpoints()
        print()
        
        # 2. Загрузка данных
        lightcurve = self.test_data_loading()
        print()
        
        # 3. Анализ
        analysis_result = self.test_analysis(lightcurve)
        print()
        
        # 4. Производительность
        self.test_performance()
        print()
        
        # 5. Итоговый отчет
        self.print_report()
    
    def print_report(self):
        """Печать итогового отчета"""
        self.print_status("📊 Итоговый отчет", "INFO")
        print("=" * 60)
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r["success"])
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📈 Всего тестов: {total_tests}")
        print(f"✅ Успешных: {successful_tests}")
        print(f"❌ Неудачных: {total_tests - successful_tests}")
        print(f"📊 Успешность: {success_rate:.1f}%")
        print()
        
        # Детальная информация по каждому тесту
        for endpoint, result in self.results.items():
            status = "✅" if result["success"] else "❌"
            time_info = f" ({result['response_time']:.3f}s)" if result["response_time"] else ""
            error_info = f" - {result['error']}" if result["error"] else ""
            
            print(f"{status} {endpoint}{time_info}{error_info}")
        
        print("=" * 60)
        
        if success_rate >= 80:
            self.print_status("🎉 Система работает отлично!", "SUCCESS")
        elif success_rate >= 60:
            self.print_status("⚠️ Система работает с предупреждениями", "WARNING")
        else:
            self.print_status("❌ Система имеет серьезные проблемы", "ERROR")
            return 1
        
        return 0

def main():
    """Главная функция"""
    print("🌌 Exoplanet AI - Системный тест")
    print("=" * 60)
    
    # Проверяем аргументы командной строки
    base_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"🔗 Тестируем: {base_url}")
    print()
    
    # Создаем тестер и запускаем тесты
    tester = ExoplanetSystemTester(base_url)
    exit_code = tester.run_full_test()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
