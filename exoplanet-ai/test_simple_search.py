#!/usr/bin/env python3
"""
Simple search test for ExoplanetAI
"""

import requests
import json

def test_simple_search():
    """Test search with a simple request"""
    
    search_request = {
        "target_name": "TIC 307210830",
        "catalog": "TIC", 
        "mission": "TESS",
        "period_min": 1.0,
        "period_max": 10.0,
        "snr_threshold": 5.0
    }
    
    print("🔍 Testing simple search...")
    print(f"Request: {json.dumps(search_request, indent=2)}")
    
    try:
        response = requests.post(
            "http://localhost:8001/api/v1/search",
            json=search_request,
            timeout=60
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Search successful!")
            print(f"Target: {data['target_name']}")
            print(f"Candidates: {data['candidates_found']}")
            print(f"Processing time: {data['processing_time_ms']:.1f}ms")
            
            if data.get('bls_result'):
                bls = data['bls_result']
                print(f"BLS Period: {bls['best_period']:.3f} days")
                print(f"BLS SNR: {bls['snr']:.1f}")
                
        else:
            print(f"❌ Search failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_simple_search()
