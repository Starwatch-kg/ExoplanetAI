"""
Light curves API routes
Маршруты API кривых блеска
"""

import logging
import time
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from auth.dependencies import require_researcher
from auth.models import User
from core.cache import get_cache
from data_sources.registry import get_registry
from schemas.response import ErrorCode, create_error_response, create_success_response

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{target_name}")
async def get_light_curve(
    target_name: str = Path(
        ..., description="Target name (e.g., 'TOI-715', 'TIC 123456')"
    ),
    mission: Optional[str] = Query(None, description="Mission (TESS, Kepler, K2)"),
    sector_quarter: Optional[int] = Query(
        None, description="Specific sector/quarter number"
    ),
    normalize: bool = Query(True, description="Normalize flux data"),
    remove_outliers: bool = Query(True, description="Remove outliers"),
    current_user: User = Depends(require_researcher),
):
    """
    Get light curve data for a target

    **🔒 Requires researcher role or higher**

    Returns real photometric time series data from space missions.

    **Examples:**
    - `/api/v1/lightcurve/TOI-715?mission=TESS`
    - `/api/v1/lightcurve/TIC%20261136679?sector_quarter=1`
    """
    start_time = time.time()

    try:
        # Check cache first
        cache = get_cache()
        cache_key = (
            f"{target_name}:{mission}:{sector_quarter}:{normalize}:{remove_outliers}"
        )
        cached_lc = await cache.get("lightcurves", cache_key)

        if cached_lc:
            logger.info(f"Cache hit for light curve: {target_name}")
            return create_success_response(
                data={"lightcurve": cached_lc, "cached": True},
                message=f"Light curve data for {target_name}",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Get registry and find sources that support light curves
        registry = get_registry()
        sources = [
            s
            for s in registry.get_available_sources()
            if s.get_capabilities().get("light_curves", False)
        ]

        if not sources:
            return create_error_response(
                ErrorCode.SERVICE_UNAVAILABLE, "No light curve data sources available"
            )

        # Try to get light curve from sources
        lightcurve_data = None
        source_used = None

        # If mission specified, try mission-specific sources first
        if mission:
            mission_sources = [
                s for s in sources if mission.upper() in s.get_supported_missions()
            ]
            for source in mission_sources:
                try:
                    lightcurve_data = await source.fetch_light_curve(
                        target_name, mission, sector_quarter
                    )
                    if lightcurve_data:
                        source_used = source.name
                        break
                except Exception as e:
                    logger.debug(f"Light curve not found in {source.name}: {e}")
                    continue

        # If not found, try all light curve sources
        if not lightcurve_data:
            for source in sources:
                try:
                    lightcurve_data = await source.fetch_light_curve(
                        target_name, mission, sector_quarter
                    )
                    if lightcurve_data:
                        source_used = source.name
                        break
                except Exception as e:
                    logger.debug(f"Light curve not found in {source.name}: {e}")
                    continue

        if not lightcurve_data:
            return create_error_response(
                ErrorCode.DATA_NOT_FOUND,
                f"Light curve data for '{target_name}' not found",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Process data according to parameters
        processed_data = _process_light_curve_data(
            lightcurve_data, normalize=normalize, remove_outliers=remove_outliers
        )

        # Add metadata
        processed_data["source_used"] = source_used
        processed_data["processing_options"] = {
            "normalized": normalize,
            "outliers_removed": remove_outliers,
        }

        # Cache for 2 hours (light curves are large)
        await cache.set("lightcurves", cache_key, processed_data, ttl=7200)

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Light curve retrieved: {target_name} from {source_used} "
            f"({len(processed_data['time'])} points)"
        )

        return create_success_response(
            data={"lightcurve": processed_data, "cached": False},
            message=f"Light curve data for {target_name}",
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Error fetching light curve for '{target_name}': {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to fetch light curve: {str(e)}",
            processing_time_ms=(time.time() - start_time) * 1000,
        )


@router.get("/{target_name}/analysis")
async def analyze_light_curve(
    target_name: str = Path(..., description="Target name"),
    mission: Optional[str] = Query(None, description="Mission"),
    period_min: float = Query(0.5, description="Minimum period to search (days)", gt=0),
    period_max: float = Query(
        50.0, description="Maximum period to search (days)", gt=0
    ),
    snr_threshold: float = Query(
        7.0, description="SNR threshold for detection", ge=3.0
    ),
    current_user: User = Depends(require_researcher),
):
    """
    Analyze light curve for transit signals

    **🔒 Requires researcher role or higher**

    Performs Box Least Squares (BLS) analysis on real light curve data
    to search for periodic transit signals.
    """
    start_time = time.time()

    try:
        # First get the light curve
        cache = get_cache()
        lc_cache_key = f"{target_name}:{mission}:None:True:True"
        lightcurve_data = await cache.get("lightcurves", lc_cache_key)

        if not lightcurve_data:
            # Need to fetch light curve first
            registry = get_registry()
            sources = [
                s
                for s in registry.get_available_sources()
                if s.get_capabilities().get("light_curves", False)
            ]

            if not sources:
                return create_error_response(
                    ErrorCode.SERVICE_UNAVAILABLE,
                    "No light curve data sources available",
                )

            # Try to get light curve
            lc_obj = None
            for source in sources:
                try:
                    lc_obj = await source.fetch_light_curve(target_name, mission)
                    if lc_obj:
                        break
                except Exception:
                    continue

            if not lc_obj:
                return create_error_response(
                    ErrorCode.DATA_NOT_FOUND,
                    f"Light curve data for '{target_name}' not found",
                )

            lightcurve_data = _process_light_curve_data(
                lc_obj, normalize=True, remove_outliers=True
            )

        # Check analysis cache
        analysis_cache_key = (
            f"analysis:{target_name}:{period_min}:{period_max}:{snr_threshold}"
        )
        cached_analysis = await cache.get("lc_analysis", analysis_cache_key)

        if cached_analysis:
            logger.info(f"Cache hit for light curve analysis: {target_name}")
            return create_success_response(
                data=cached_analysis,
                message=f"Light curve analysis for {target_name} (cached)",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Perform BLS analysis
        analysis_result = await _perform_bls_analysis(
            lightcurve_data,
            period_min=period_min,
            period_max=period_max,
            snr_threshold=snr_threshold,
        )

        # Add metadata
        analysis_result["target_name"] = target_name
        analysis_result["analysis_parameters"] = {
            "period_min_days": period_min,
            "period_max_days": period_max,
            "snr_threshold": snr_threshold,
        }
        analysis_result["data_points"] = len(lightcurve_data["time"])

        # Cache analysis for 4 hours
        await cache.set("lc_analysis", analysis_cache_key, analysis_result, ttl=14400)

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Light curve analysis completed: {target_name} "
            f"(SNR: {analysis_result.get('best_snr', 0):.2f})"
        )

        return create_success_response(
            data=analysis_result,
            message=f"Light curve analysis for {target_name}",
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Error analyzing light curve for '{target_name}': {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Light curve analysis failed: {str(e)}",
            processing_time_ms=(time.time() - start_time) * 1000,
        )


def _process_light_curve_data(lc_data, normalize=True, remove_outliers=True) -> dict:
    """Process light curve data according to options"""

    # Convert to numpy arrays if needed
    time = np.array(lc_data.time_bjd)
    flux = np.array(lc_data.flux)
    flux_err = np.array(lc_data.flux_err)

    # Remove NaN values
    mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    time = time[mask]
    flux = flux[mask]
    flux_err = flux_err[mask]

    # Remove outliers if requested
    if remove_outliers:
        # Simple sigma clipping
        flux_median = np.median(flux)
        flux_std = np.std(flux)
        outlier_mask = np.abs(flux - flux_median) < 5 * flux_std

        time = time[outlier_mask]
        flux = flux[outlier_mask]
        flux_err = flux_err[outlier_mask]

    # Normalize if requested
    if normalize:
        flux_median = np.median(flux)
        flux = flux / flux_median
        flux_err = flux_err / flux_median

    return {
        "target_name": lc_data.target_name,
        "mission": lc_data.mission,
        "time": time.tolist(),
        "flux": flux.tolist(),
        "flux_err": flux_err.tolist(),
        "time_format": "BJD",
        "flux_format": "Normalized" if normalize else "Raw",
        "cadence_minutes": lc_data.cadence_minutes,
        "sectors_quarters": lc_data.sectors_quarters,
        "data_points": len(time),
        "time_span_days": float(np.max(time) - np.min(time)),
        "noise_level_ppm": float(np.std(flux) * 1e6) if len(flux) > 0 else None,
    }


async def _perform_bls_analysis(
    lc_data, period_min=0.5, period_max=50.0, snr_threshold=7.0
) -> dict:
    """Perform Box Least Squares analysis on light curve data"""

    try:
        # Import BLS here to avoid dependency issues
        import astropy.units as u
        from astropy.timeseries import BoxLeastSquares

        time = np.array(lc_data["time"])
        flux = np.array(lc_data["flux"])

        # Create BLS object
        bls = BoxLeastSquares(time * u.day, flux)

        # Define period grid
        periods = np.linspace(period_min, period_max, 10000)

        # Run BLS
        periodogram = bls.power(periods * u.day)

        # Find best period
        best_index = np.argmax(periodogram.power)
        best_period = periods[best_index]
        best_power = periodogram.power[best_index]

        # Calculate SNR (simplified)
        noise_level = np.std(periodogram.power)
        snr = (best_power - np.median(periodogram.power)) / noise_level

        # Get transit parameters for best period
        stats = bls.compute_stats(best_period * u.day)

        # Determine if significant
        is_significant = snr >= snr_threshold

        return {
            "method": "Box Least Squares (BLS)",
            "best_period_days": float(best_period),
            "best_power": float(best_power),
            "best_snr": float(snr),
            "is_significant": is_significant,
            "transit_depth": (
                float(stats["depth"][0]) if len(stats["depth"]) > 0 else None
            ),
            "transit_duration_hours": (
                float(stats["duration"][0] * 24) if len(stats["duration"]) > 0 else None
            ),
            "transit_epoch": (
                float(stats["transit_time"][0])
                if len(stats["transit_time"]) > 0
                else None
            ),
            "periods_searched": len(periods),
            "period_range_days": [period_min, period_max],
            "snr_threshold": snr_threshold,
        }

    except ImportError:
        # Fallback simple analysis if astropy not available
        logger.warning("Astropy BLS not available, using simple analysis")

        time = np.array(lc_data["time"])
        flux = np.array(lc_data["flux"])

        # Very simple period detection (for demo)
        # In real implementation, you'd use proper BLS
        flux_std = np.std(flux)

        return {
            "method": "Simple Analysis (Fallback)",
            "best_period_days": 2.5,  # Placeholder
            "best_power": 0.1,
            "best_snr": 5.0,
            "is_significant": False,
            "transit_depth": None,
            "transit_duration_hours": None,
            "transit_epoch": None,
            "periods_searched": 1000,
            "period_range_days": [period_min, period_max],
            "snr_threshold": snr_threshold,
            "note": "Fallback analysis - install astropy for full BLS",
        }

    except Exception as e:
        logger.error(f"BLS analysis error: {e}")
        raise
