#!/usr/bin/env python3
"""
Простой тест API для проверки работоспособности
"""
import requests
import json
import time

def test_api():
    base_url = "http://localhost:8000"
    
    print("🔍 Тестирование API...")
    
    # 1. Проверяем health endpoint
    try:
        print("1. Проверяем /api/health...")
        response = requests.get(f"{base_url}/api/health", timeout=10)
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ Health OK: {response.json()}")
        else:
            print(f"   ❌ Health failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Health error: {e}")
    
    # 2. Проверяем CORS endpoint
    try:
        print("2. Проверяем /api/test-cors...")
        response = requests.get(f"{base_url}/api/test-cors", timeout=10)
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ CORS OK: {response.json()}")
        else:
            print(f"   ❌ CORS failed: {response.text}")
    except Exception as e:
        print(f"   ❌ CORS error: {e}")
    
    # 3. Тестируем простой поиск
    try:
        print("3. Тестируем /api/search...")
        test_data = {
            "target_name": "167692429",
            "catalog": "TIC",
            "mission": "TESS",
            "period_min": 0.5,
            "period_max": 20.0,
            "duration_min": 0.05,
            "duration_max": 0.3,
            "snr_threshold": 7.0
        }
        
        response = requests.post(
            f"{base_url}/api/search", 
            json=test_data,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Search OK: найдено {len(result.get('candidates', []))} кандидатов")
        else:
            print(f"   ❌ Search failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Search error: {e}")
    
    # 4. Тестируем AI поиск
    try:
        print("4. Тестируем /api/ai-search...")
        response = requests.post(
            f"{base_url}/api/ai-search", 
            json=test_data,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ AI Search OK: найдено {len(result.get('candidates', []))} кандидатов")
            if result.get('ai_analysis'):
                print(f"   🤖 AI анализ: уверенность {result['ai_analysis'].get('confidence', 0):.3f}")
        else:
            print(f"   ❌ AI Search failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ AI Search error: {e}")
    
    # 5. Тестируем NASA Data Browser
    try:
        print("5. Тестируем /api/nasa-data...")
        response = requests.get(
            f"{base_url}/api/nasa-data/441420236?catalog=TIC&mission=TESS",
            timeout=30
        )
        
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ NASA Data OK: {result.get('data_source', 'Unknown')}")
            if result.get('confirmed_planets'):
                print(f"   🪐 Найдено {len(result['confirmed_planets'])} подтвержденных планет")
        else:
            print(f"   ❌ NASA Data failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ NASA Data error: {e}")
    
    # 6. Тестируем получение подтвержденных планет
    try:
        print("6. Тестируем /api/confirmed-planets...")
        response = requests.get(
            f"{base_url}/api/confirmed-planets/441420236",
            timeout=30
        )
        
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Confirmed Planets OK: найдено {result.get('count', 0)} планет")
        else:
            print(f"   ❌ Confirmed Planets failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Confirmed Planets error: {e}")
    
    # 7. Тестируем каталоги
    try:
        print("7. Тестируем /api/catalogs...")
        response = requests.get(f"{base_url}/api/catalogs", timeout=10)
        
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Catalogs OK: {len(result.get('catalogs', []))} каталогов")
        else:
            print(f"   ❌ Catalogs failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Catalogs error: {e}")
    
    # 8. Тестируем получение кривой блеска
    try:
        print("8. Тестируем /api/lightcurve...")
        response = requests.get(
            f"{base_url}/api/lightcurve/441420236?mission=TESS",
            timeout=30
        )
        
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Lightcurve OK: {len(result.get('time', []))} точек данных")
        else:
            print(f"   ❌ Lightcurve failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Lightcurve error: {e}")
    
    print("\n🎯 Тестирование завершено!")

def test_enhanced_detector():
    """Тестирование усиленного детектора"""
    print("\n🚀 Тестирование усиленного детектора транзитов...")
    
    try:
        # Импорт и запуск тестов
        import subprocess
        import sys
        
        result = subprocess.run([
            sys.executable, "test_enhanced_detector.py"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Усиленный детектор работает корректно")
            print("📊 Основные результаты:")
            # Выводим последние строки с результатами
            lines = result.stdout.split('\n')
            for line in lines[-10:]:
                if line.strip():
                    print(f"   {line}")
        else:
            print("❌ Ошибка в усиленном детекторе:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Тест усиленного детектора превысил лимит времени")
    except Exception as e:
        print(f"❌ Ошибка при тестировании усиленного детектора: {e}")

if __name__ == "__main__":
    test_api()
    test_enhanced_detector()
