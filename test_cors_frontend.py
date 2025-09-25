#!/usr/bin/env python3
"""
Тест CORS для frontend
"""

import requests
import json

def test_cors_from_frontend():
    """Тестируем CORS так, как это делает frontend"""
    
    print("🧪 Тестируем CORS для frontend...")
    print("=" * 50)
    
    # Заголовки, которые отправляет браузер
    headers = {
        'Origin': 'http://localhost:5173',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    # 1. Тест OPTIONS запроса (preflight)
    print("1. Тестируем OPTIONS preflight запрос...")
    try:
        response = requests.options(
            'http://localhost:8000/api/v1/search',
            headers={
                'Origin': 'http://localhost:5173',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'Content-Type'
            }
        )
        print(f"   Status: {response.status_code}")
        print(f"   CORS Headers:")
        for key, value in response.headers.items():
            if 'access-control' in key.lower():
                print(f"     {key}: {value}")
        print()
    except Exception as e:
        print(f"   ❌ OPTIONS failed: {e}")
        print()
    
    # 2. Тест POST запроса (как frontend)
    print("2. Тестируем POST запрос как frontend...")
    try:
        data = {
            "target_name": "307210830",
            "catalog": "TIC",
            "mission": "TESS",
            "use_bls": True,
            "use_ai": True,
            "use_ensemble": True,
            "search_mode": "ensemble",
            "period_min": 0.5,
            "period_max": 20,
            "snr_threshold": 7
        }
        
        response = requests.post(
            'http://localhost:8000/api/v1/search',
            headers=headers,
            json=data
        )
        
        print(f"   Status: {response.status_code}")
        print(f"   CORS Headers:")
        for key, value in response.headers.items():
            if 'access-control' in key.lower():
                print(f"     {key}: {value}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success!")
            print(f"   Target: {result.get('target_name')}")
            print(f"   Candidates: {result.get('candidates_found')}")
        else:
            print(f"   ❌ Failed!")
            print(f"   Error: {response.text}")
        print()
        
    except Exception as e:
        print(f"   ❌ POST failed: {e}")
        print()
    
    # 3. Тест простого GET запроса
    print("3. Тестируем GET health запрос...")
    try:
        response = requests.get(
            'http://localhost:8000/api/v1/health',
            headers={'Origin': 'http://localhost:5173'}
        )
        print(f"   Status: {response.status_code}")
        print(f"   CORS Headers:")
        for key, value in response.headers.items():
            if 'access-control' in key.lower():
                print(f"     {key}: {value}")
        print()
    except Exception as e:
        print(f"   ❌ GET failed: {e}")
        print()

if __name__ == "__main__":
    test_cors_from_frontend()
