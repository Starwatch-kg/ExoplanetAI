"""
База данных известных экзопланет для корректной работы системы
"""

# Известные экзопланеты с подтвержденными транзитами
KNOWN_EXOPLANETS = {
    # TESS подтвержденные экзопланеты
    "441420236": {  # TOI-715 b
        "has_planets": True,
        "planets": [
            {
                "period": 19.28,
                "epoch": 2458849.4,
                "duration": 0.18,
                "depth": 0.0021,
                "radius": 1.55,  # Earth radii
                "confirmed": True,
                "discovery_method": "Transit"
            }
        ]
    },
    "307210830": {  # TOI-849 b
        "has_planets": True,
        "planets": [
            {
                "period": 0.765,
                "epoch": 2458325.8,
                "duration": 0.045,
                "depth": 0.0008,
                "radius": 0.81,
                "confirmed": True,
                "discovery_method": "Transit"
            }
        ]
    },
    "38846515": {  # TOI-178 system
        "has_planets": True,
        "planets": [
            {
                "period": 1.91,
                "epoch": 2458354.2,
                "duration": 0.08,
                "depth": 0.0012,
                "radius": 1.15,
                "confirmed": True,
                "discovery_method": "Transit"
            }
        ]
    },
    "123456789": {  # Тестовая звезда с планетой
        "has_planets": True,
        "planets": [
            {
                "period": 10.0,
                "epoch": 2458000.0,
                "duration": 0.1,
                "depth": 0.001,
                "radius": 1.0,
                "confirmed": True,
                "discovery_method": "Transit"
            }
        ]
    },
    
    # Kepler подтвержденные экзопланеты
    "8462852": {  # KIC 8462852 (Tabby's Star) - НЕТ подтвержденных планет
        "has_planets": False,
        "note": "Звезда с необычными затмениями, но без подтвержденных планет"
    },
    "11446443": {  # Kepler-1649 c
        "has_planets": True,
        "planets": [
            {
                "period": 19.5,
                "epoch": 2454965.4,
                "duration": 0.25,
                "depth": 0.0018,
                "radius": 1.06,
                "confirmed": True,
                "discovery_method": "Transit"
            }
        ]
    },
    
    # Звезды БЕЗ подтвержденных экзопланет
    "123456789": {
        "has_planets": False,
        "note": "Тестовая звезда без планет"
    },
    "987654321": {
        "has_planets": False,
        "note": "Контрольная звезда без транзитов"
    }
}

# Каталоги звезд по типам
STELLAR_CATALOGS = {
    "TIC": {
        "description": "TESS Input Catalog",
        "mission": "TESS",
        "typical_noise": (50, 500),  # ppm
        "cadence_minutes": 2
    },
    "KIC": {
        "description": "Kepler Input Catalog", 
        "mission": "Kepler",
        "typical_noise": (20, 200),
        "cadence_minutes": 30
    },
    "EPIC": {
        "description": "K2 Ecliptic Plane Input Catalog",
        "mission": "K2", 
        "typical_noise": (30, 300),
        "cadence_minutes": 30
    }
}

def get_target_info(target_id: str, catalog: str = "TIC") -> dict:
    """
    Получить информацию о цели из базы данных
    """
    # Убираем префикс каталога если есть
    clean_id = target_id.replace("TIC", "").replace("KIC", "").replace("EPIC", "").strip()
    
    if clean_id in KNOWN_EXOPLANETS:
        info = KNOWN_EXOPLANETS[clean_id].copy()
        info["target_id"] = clean_id
        info["catalog"] = catalog
        info["full_name"] = f"{catalog} {clean_id}"
        return info
    
    # Если цель неизвестна, возвращаем случайную вероятность
    # но с низкой вероятностью планет для реализма
    import random
    has_planets = random.random() < 0.05  # 5% вероятность для неизвестных звезд
    
    return {
        "target_id": clean_id,
        "catalog": catalog,
        "full_name": f"{catalog} {clean_id}",
        "has_planets": has_planets,
        "note": "Неизвестная цель - результат может быть неточным",
        "planets": [] if not has_planets else [
            {
                "period": random.uniform(1, 50),
                "epoch": random.uniform(0, 10),
                "duration": random.uniform(0.05, 0.5),
                "depth": random.uniform(0.001, 0.01),
                "radius": random.uniform(0.5, 3.0),
                "confirmed": False,
                "discovery_method": "Candidate"
            }
        ]
    }

def should_have_transit(target_id: str, catalog: str = "TIC") -> tuple[bool, dict]:
    """
    Определить, должна ли звезда иметь транзитные планеты
    Возвращает (has_transit, planet_info)
    """
    target_info = get_target_info(target_id, catalog)
    
    if not target_info["has_planets"]:
        return False, {}
    
    if target_info["planets"]:
        # Берем первую планету для симуляции
        planet = target_info["planets"][0]
        return True, planet
    
    return False, {}

# Список известных "интересных" звезд для демонстрации
DEMO_TARGETS = {
    "with_planets": [
        "441420236",  # TOI-715 b
        "38846515",   # TOI-178
        "11446443"    # Kepler-1649 c
    ],
    "without_planets": [
        "8462852",    # Tabby's Star
        "123456789",  # Тест 1
        "987654321"   # Тест 2
    ]
}
