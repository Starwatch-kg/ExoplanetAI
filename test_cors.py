#!/usr/bin/env python3
"""
Тестовый скрипт для проверки CORS и API endpoints
"""

import requests
import json
import time
from typing import Dict, Any

# Конфигурация
API_BASE_URL = "http://localhost:8000"
API_V1_URL = f"{API_BASE_URL}/api/v1"

def test_cors_headers(url: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
    """
    Тестирует CORS заголовки для указанного URL
    """
    print(f"\n🔍 Testing CORS for {method} {url}")
    
    headers = {
        "Origin": "http://localhost:5173",
        "Content-Type": "application/json"
    }
    
    try:
        if method == "OPTIONS":
            response = requests.options(url, headers=headers)
        elif method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        else:
            print(f"❌ Unsupported method: {method}")
            return {"error": f"Unsupported method: {method}"}
        
        result = {
            "status_code": response.status_code,
            "cors_headers": {},
            "response_data": None,
            "error": None
        }
        
        # Проверяем CORS заголовки
        cors_headers = [
            "Access-Control-Allow-Origin",
            "Access-Control-Allow-Methods", 
            "Access-Control-Allow-Headers",
            "Access-Control-Allow-Credentials",
            "Access-Control-Expose-Headers"
        ]
        
        for header in cors_headers:
            if header in response.headers:
                result["cors_headers"][header] = response.headers[header]
        
        # Получаем данные ответа
        try:
            if response.headers.get("content-type", "").startswith("application/json"):
                result["response_data"] = response.json()
            else:
                result["response_data"] = response.text[:200]  # Первые 200 символов
        except:
            result["response_data"] = "Could not parse response"
        
        # Статус
        if response.status_code < 400:
            print(f"✅ Success: {response.status_code}")
        else:
            print(f"❌ Error: {response.status_code}")
            result["error"] = f"HTTP {response.status_code}"
        
        # Выводим CORS заголовки
        if result["cors_headers"]:
            print("📋 CORS Headers:")
            for header, value in result["cors_headers"].items():
                print(f"   {header}: {value}")
        else:
            print("⚠️  No CORS headers found")
        
        return result
        
    except requests.exceptions.ConnectionError:
        error_msg = f"❌ Connection failed to {url}"
        print(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"❌ Request failed: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

def test_api_endpoints():
    """
    Тестирует основные API endpoints
    """
    print("=" * 80)
    print("🚀 TESTING EXOPLANET AI API ENDPOINTS")
    print("=" * 80)
    
    # Список endpoints для тестирования
    endpoints = [
        # Health checks
        {"url": f"{API_BASE_URL}/", "method": "GET", "name": "Root endpoint"},
        {"url": f"{API_V1_URL}/health", "method": "GET", "name": "Health check"},
        {"url": f"{API_V1_URL}/test-cors", "method": "GET", "name": "CORS test"},
        
        # OPTIONS requests (preflight)
        {"url": f"{API_V1_URL}/health", "method": "OPTIONS", "name": "Health OPTIONS"},
        {"url": f"{API_V1_URL}/bls", "method": "OPTIONS", "name": "BLS OPTIONS"},
        
        # BLS endpoint
        {
            "url": f"{API_V1_URL}/bls", 
            "method": "POST", 
            "name": "BLS Analysis",
            "data": {
                "target_name": "TIC 123456789",
                "catalog": "TIC",
                "mission": "TESS",
                "period_min": 1.0,
                "period_max": 10.0,
                "snr_threshold": 7.0,
                "use_enhanced": True
            }
        },
        
        # Search endpoint
        {
            "url": f"{API_V1_URL}/search", 
            "method": "POST", 
            "name": "Exoplanet Search",
            "data": {
                "target_name": "TIC 987654321",
                "catalog": "TIC",
                "mission": "TESS",
                "use_bls": True,
                "use_ai": True,
                "period_min": 1.0,
                "period_max": 10.0,
                "snr_threshold": 7.0
            }
        }
    ]
    
    results = []
    
    for endpoint in endpoints:
        result = test_cors_headers(
            url=endpoint["url"],
            method=endpoint["method"],
            data=endpoint.get("data")
        )
        result["endpoint_name"] = endpoint["name"]
        results.append(result)
        
        time.sleep(0.5)  # Небольшая пауза между запросами
    
    # Сводка результатов
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    success_count = 0
    total_count = len(results)
    
    for result in results:
        name = result["endpoint_name"]
        if result.get("error"):
            print(f"❌ {name}: {result['error']}")
        elif result["status_code"] < 400:
            print(f"✅ {name}: HTTP {result['status_code']}")
            success_count += 1
        else:
            print(f"⚠️  {name}: HTTP {result['status_code']}")
    
    print(f"\n📈 Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    # Проверка CORS
    cors_working = any(
        result.get("cors_headers", {}).get("Access-Control-Allow-Origin") 
        for result in results
    )
    
    if cors_working:
        print("✅ CORS: Working")
    else:
        print("❌ CORS: Not working properly")
    
    return results

def main():
    """
    Главная функция тестирования
    """
    print("🧪 Exoplanet AI CORS & API Tester")
    print(f"🌐 Testing API at: {API_BASE_URL}")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = test_api_endpoints()
    
    # Сохраняем результаты в файл
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"test_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    print("\n🏁 Testing completed!")

if __name__ == "__main__":
    main()
