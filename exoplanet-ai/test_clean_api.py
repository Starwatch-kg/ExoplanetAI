#!/usr/bin/env python3
"""
Test script for Clean ExoplanetAI API
Тестовый скрипт для очищенного API
"""

import requests
import json
import time

API_BASE = "http://localhost:8001"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/api/v1/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Version: {data['version']}")
            print(f"   Services: {data['services']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_catalogs():
    """Test catalogs endpoint"""
    print("\n📚 Testing catalogs endpoint...")
    try:
        response = requests.get(f"{API_BASE}/api/v1/catalogs", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Catalogs retrieved successfully")
            print(f"   Available catalogs: {data['catalogs']}")
            print(f"   Available missions: {data['missions']}")
            return True
        else:
            print(f"❌ Catalogs failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Catalogs error: {e}")
        return False

def test_search():
    """Test search endpoint with real data"""
    print("\n🔍 Testing search endpoint...")
    
    # Test with a well-known exoplanet host star
    search_request = {
        "target_name": "TIC 307210830",  # TOI-715 - known exoplanet host
        "catalog": "TIC",
        "mission": "TESS",
        "period_min": 0.5,
        "period_max": 20.0,
        "snr_threshold": 7.0
    }
    
    try:
        print(f"   Searching for: {search_request['target_name']}")
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE}/api/v1/search", 
            json=search_request,
            timeout=120  # 2 minutes for real NASA data
        )
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Search completed in {duration:.1f}s")
            print(f"   Target: {data['target_name']}")
            print(f"   Candidates found: {data['candidates_found']}")
            print(f"   Processing time: {data['processing_time_ms']:.1f}ms")
            
            if data['bls_result']:
                bls = data['bls_result']
                print(f"   BLS Results:")
                print(f"     Period: {bls['best_period']:.3f} days")
                print(f"     SNR: {bls['snr']:.1f}")
                print(f"     Depth: {bls['depth']:.6f}")
                print(f"     Significant: {bls['is_significant']}")
            
            lightcurve = data['lightcurve_info']
            print(f"   Lightcurve: {lightcurve['points_count']} points")
            print(f"   Data source: {lightcurve['data_source']}")
            
            star = data['star_info']
            print(f"   Star: mag={star['magnitude']:.1f}")
            
            return True
        else:
            print(f"❌ Search failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"   Raw response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Search timed out (this is normal for real NASA data)")
        return False
    except Exception as e:
        print(f"❌ Search error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing ExoplanetAI Clean API")
    print("=" * 50)
    
    # Test basic endpoints
    health_ok = test_health()
    catalogs_ok = test_catalogs()
    
    if not health_ok:
        print("\n❌ Basic health check failed, skipping search test")
        return
    
    # Test search (this might take a while with real NASA data)
    print("\n⚠️  Search test uses real NASA data and may take 1-2 minutes...")
    search_ok = test_search()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Health: {'✅' if health_ok else '❌'}")
    print(f"   Catalogs: {'✅' if catalogs_ok else '❌'}")
    print(f"   Search: {'✅' if search_ok else '❌'}")
    
    if health_ok and catalogs_ok:
        print("\n🎉 Basic API functionality working!")
        if search_ok:
            print("🌟 Full API including real NASA data working!")
        else:
            print("⚠️  Search may need more time or different target")
    else:
        print("\n❌ API has issues, check server logs")

if __name__ == "__main__":
    main()
