#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест enhanced_bls модуля
"""

import sys
import os

# Добавляем backend в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

try:
    print("Testing enhanced_bls import...")
    from enhanced_bls import EnhancedBLS
    print("✅ Enhanced BLS imported successfully")
    
    # Тестируем создание объекта
    print("Testing EnhancedBLS creation...")
    bls = EnhancedBLS(
        minimum_period=1.0,
        maximum_period=10.0,
        frequency_factor=3.0,
        minimum_n_transit=2,
        maximum_duration_factor=0.3,
        enable_ml_validation=True
    )
    print("✅ EnhancedBLS object created successfully")
    
    # Тестируем простой поиск
    print("Testing BLS search...")
    import numpy as np
    
    # Создаем тестовые данные
    time = np.linspace(0, 30, 1000)
    flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))
    
    # Добавляем простой транзит
    period = 3.14159
    depth = 0.002
    duration = 0.1
    t0 = 1.5
    
    for i in range(int(30 / period) + 1):
        transit_time = t0 + i * period
        if transit_time > 30:
            break
        in_transit = np.abs(time - transit_time) < duration / 2
        flux[in_transit] -= depth
    
    # Запускаем поиск
    result = bls.search(time, flux, target_name="TEST")
    print("✅ BLS search completed successfully")
    print(f"📊 Best period: {result['best_period']:.3f} days")
    print(f"📊 SNR: {result['snr']:.1f}")
    print(f"📊 Significance: {result['is_significant']}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("🎉 All tests passed!")
