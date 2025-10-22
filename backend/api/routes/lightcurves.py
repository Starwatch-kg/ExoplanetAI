"""
Light curves API routes - REAL NASA DATA ONLY
Маршруты API кривых блеска - ТОЛЬКО РЕАЛЬНЫЕ ДАННЫЕ NASA
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query

try:
    import lightkurve as lk
    from astroquery.mast import Catalogs, Observations
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    LIGHTKURVE_AVAILABLE = True
except ImportError:
    LIGHTKURVE_AVAILABLE = False
    logging.warning("lightkurve/astroquery not available - real NASA data disabled")

from auth.dependencies import require_researcher
from auth.models import User
from core.cache import get_cache
from data_sources.registry import get_registry
from schemas.response import ErrorCode, create_error_response, create_success_response

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/real/{target_name}")
async def get_real_lightcurve(
    target_name: str = Path(..., description="Target name (e.g., TOI-715, TIC-441420236)"),
    mission: str = Query("TESS", description="Mission name (TESS, Kepler, K2)"),
    sector: Optional[int] = Query(None, description="TESS sector number"),
):
    """
    Get REAL lightcurve data from NASA archives
    Получить РЕАЛЬНЫЕ данные кривой блеска из архивов NASA
    """
    if not LIGHTKURVE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Real NASA data service unavailable - lightkurve not installed"
        )
    
    try:
        logger.info(f"Fetching REAL NASA data for {target_name} from {mission}")
        
        # Search for real lightcurve data
        search_result = lk.search_lightcurve(
            target_name, 
            mission=mission.upper(),
            sector=sector
        )
        
        if len(search_result) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No real {mission} data found for target {target_name}"
            )
        
        # Download the first available lightcurve
        lc = search_result[0].download()
        
        if lc is None:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to download real data for {target_name}"
            )
        
        # Clean and normalize the real data
        lc = lc.normalize().remove_outliers(sigma=5)
        
        # Extract real data
        time_data = lc.time.value.tolist()
        flux_data = lc.flux.value.tolist()
        quality_data = lc.quality.value.tolist() if hasattr(lc, 'quality') else [0] * len(time_data)
        
        # Get real metadata
        metadata = {
            "mission": mission.upper(),
            "target_name": target_name,
            "sector": getattr(lc, 'sector', sector),
            "camera": getattr(lc, 'camera', None),
            "ccd": getattr(lc, 'ccd', None),
            "data_points": len(time_data),
            "time_span_days": float(max(time_data) - min(time_data)),
            "cadence_minutes": float(lc.time[1].value - lc.time[0].value) * 24 * 60,
            "data_source": "NASA MAST Archive",
            "real_data": True
        }
        
        return create_success_response({
            "target_name": target_name,
            "mission": mission.upper(),
            "lightcurve": {
                "time": time_data,
                "flux": flux_data,
                "quality": quality_data,
                "time_data": time_data,  # Для совместимости с фронтендом
                "flux_data": flux_data   # Для совместимости с фронтендом
            },
            "metadata": metadata,
            "data_quality": {
                "total_points": len(time_data),
                "good_quality_points": sum(1 for q in quality_data if q == 0),
                "outliers_removed": True,
                "normalized": True
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching real NASA data for {target_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch real NASA data: {str(e)}"
        )


@router.get("/search/{target_name}")
async def search_real_targets(
    target_name: str = Path(..., description="Target name to search"),
    mission: str = Query("TESS", description="Mission name"),
    limit: int = Query(10, description="Maximum results"),
):
    """
    Search for real targets in NASA archives
    Поиск реальных целей в архивах NASA
    """
    if not LIGHTKURVE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Real NASA search unavailable - lightkurve not installed"
        )
    
    try:
        logger.info(f"Searching real NASA targets for {target_name}")
        
        # Search for real observations
        search_result = lk.search_lightcurve(target_name, mission=mission.upper())
        
        if len(search_result) == 0:
            return create_success_response({
                "query": target_name,
                "mission": mission.upper(),
                "results": [],
                "total_found": 0,
                "message": f"No real {mission} observations found for {target_name}"
            })
        
        # Limit results
        search_result = search_result[:limit]
        
        results = []
        for obs in search_result:
            result = {
                "target_name": str(obs.target_name),
                "mission": str(obs.mission),
                "sector": getattr(obs, 'sector', None),
                "camera": getattr(obs, 'camera', None),
                "ccd": getattr(obs, 'ccd', None),
                "exptime": getattr(obs, 'exptime', None),
                "distance": getattr(obs, 'distance', None),
                "data_available": True,
                "real_nasa_data": True
            }
            results.append(result)
        
        return create_success_response({
            "query": target_name,
            "mission": mission.upper(),
            "results": results,
            "total_found": len(results),
            "data_source": "NASA MAST Archive"
        })
        
    except Exception as e:
        logger.error(f"Error searching real NASA targets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search real NASA data: {str(e)}"
        )


@router.get("/demo/{target_name}")
async def get_demo_lightcurve(
    target_name: str = Path(..., description="Target name for demo data"),
    mission: str = Query("TESS", description="Mission name (TESS, Kepler, K2)"),
):
    """
    Get demo lightcurve data for testing (no authentication required)
    Получить demo данные кривой блеска для тестирования
    """
    import numpy as np
    import hashlib
    
    try:
        logger.info(f"Generating demo lightcurve for {target_name} from {mission}")
        
        # Use target name hash for deterministic results
        seed = int(hashlib.md5(target_name.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        
        # Generate time series (TESS sector = 27.4 days)
        if mission.upper() == "TESS":
            duration_days = 27.4
            cadence_minutes = 2.0  # TESS 2-minute cadence
        elif mission.upper() in ["KEPLER", "K2"]:
            duration_days = 90.0
            cadence_minutes = 29.4  # Kepler long cadence
        else:
            duration_days = 30.0
            cadence_minutes = 10.0
        
        # Generate time array
        total_points = int((duration_days * 24 * 60) / cadence_minutes)
        time = np.linspace(0, duration_days, total_points)
        
        # Generate base flux with realistic noise
        base_flux = np.ones_like(time)
        noise_level = 1000e-6  # 1000 ppm noise
        noise = np.random.normal(0, noise_level, len(time))
        flux = base_flux + noise
        
        # Add transit signals for known targets
        if any(prefix in target_name.upper() for prefix in ['TOI', 'TIC', 'KEPLER', 'KOI']):
            # Add realistic transit
            period = np.random.uniform(1.0, 50.0)  # 1-50 day period
            depth = np.random.uniform(0.001, 0.01)  # 0.1-1% depth
            duration = np.random.uniform(0.05, 0.2) * period  # 5-20% of period
            
            # Add multiple transits
            num_transits = int(duration_days / period)
            for i in range(num_transits):
                transit_center = (i + 0.5) * period
                if transit_center < duration_days:
                    # Simple box transit model
                    transit_mask = np.abs(time - transit_center) < (duration / 2)
                    flux[transit_mask] -= depth
        
        # Add stellar variability
        stellar_period = np.random.uniform(5.0, 30.0)  # Stellar rotation
        stellar_amplitude = np.random.uniform(0.0005, 0.005)  # 0.05-0.5%
        stellar_variation = stellar_amplitude * np.sin(2 * np.pi * time / stellar_period)
        flux += stellar_variation
        
        # Quality flags (0 = good, >0 = bad)
        quality = np.zeros_like(time, dtype=int)
        # Add some bad data points
        bad_indices = np.random.choice(len(time), size=int(0.05 * len(time)), replace=False)
        quality[bad_indices] = 1
        
        return create_success_response({
            "target_name": target_name,
            "mission": mission.upper(),
            "lightcurve": {
                "time": time.tolist(),
                "flux": flux.tolist(),
                "quality": quality.tolist(),
                "time_data": time.tolist(),  # Для совместимости с фронтендом
                "flux_data": flux.tolist()   # Для совместимости с фронтендом
            },
            "metadata": {
                "total_points": len(time),
                "duration_days": duration_days,
                "cadence_minutes": cadence_minutes,
                "noise_level_ppm": int(noise_level * 1e6),
                "data_source": "Demo Data Generator",
                "real_data": False
            },
            "message": f"Demo lightcurve generated for {target_name}"
        })
        
    except Exception as e:
        logger.error(f"Error generating demo lightcurve: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate demo lightcurve: {str(e)}"
        )
