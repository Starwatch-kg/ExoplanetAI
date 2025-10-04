"""
Comprehensive light curve preprocessing system
Комплексная система предобработки кривых блеска
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.stats import sigma_clip
from lightkurve import LightCurve
from scipy import signal
from scipy.signal import savgol_filter

from .quality_filter import QualityFilter
from .denoiser import WaveletDenoiser
from .normalizer import FluxNormalizer
from .outlier_detector import OutlierDetector
from data_sources.base import LightCurveData

logger = logging.getLogger(__name__)


class LightCurveProcessor:
    """
    Advanced light curve preprocessing pipeline implementing
    state-of-the-art detrending and denoising techniques
    """

    def __init__(self):
        self.quality_filter = QualityFilter()
        self.denoiser = WaveletDenoiser()
        self.normalizer = FluxNormalizer()
        self.outlier_detector = OutlierDetector()
        
        # Processing parameters
        self.default_params = {
            "remove_outliers": True,
            "sigma_clip_sigma": 5.0,
            "sigma_clip_maxiters": 3,
            "baseline_window_length": 101,
            "baseline_polyorder": 2,
            "wavelet_denoising": False,
            "wavelet_type": "db4",
            "wavelet_levels": 6,
            "normalize_method": "median",
            "quality_bitmask": "default",
            "gap_threshold_hours": 0.5,
            "min_points_per_segment": 100
        }

    async def process_lightcurve(
        self,
        lightcurve_data: LightCurveData,
        processing_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[LightCurveData, Dict[str, Any]]:
        """
        Process light curve through complete preprocessing pipeline
        
        Args:
            lightcurve_data: Input light curve data
            processing_params: Custom processing parameters
            
        Returns:
            Tuple of (processed_lightcurve, processing_report)
        """
        params = {**self.default_params, **(processing_params or {})}
        
        logger.info(f"Processing light curve for {lightcurve_data.target_name}")
        
        # Initialize processing report
        report = {
            "target_name": lightcurve_data.target_name,
            "mission": lightcurve_data.mission,
            "original_points": len(lightcurve_data.time_bjd),
            "processing_steps": [],
            "quality_metrics": {},
            "parameters_used": params
        }
        
        # Create working copy
        processed_lc = LightCurveData(
            target_name=lightcurve_data.target_name,
            time_bjd=lightcurve_data.time_bjd.copy(),
            flux=lightcurve_data.flux.copy(),
            flux_err=lightcurve_data.flux_err.copy(),
            mission=lightcurve_data.mission,
            instrument=lightcurve_data.instrument,
            cadence_minutes=lightcurve_data.cadence_minutes,
            sectors_quarters=lightcurve_data.sectors_quarters,
            data_quality_flags=lightcurve_data.data_quality_flags.copy() if lightcurve_data.data_quality_flags is not None else None,
            source=lightcurve_data.source,
            download_date=lightcurve_data.download_date
        )
        
        try:
            # Step 1: Quality filtering
            if processed_lc.data_quality_flags is not None:
                processed_lc, quality_report = await self.quality_filter.filter_by_quality(
                    processed_lc, params["quality_bitmask"]
                )
                report["processing_steps"].append({
                    "step": "quality_filtering",
                    "points_before": quality_report["points_before"],
                    "points_after": quality_report["points_after"],
                    "removed_points": quality_report["removed_points"],
                    "bad_quality_flags": quality_report["bad_quality_flags"]
                })
                logger.info(f"Quality filtering: {quality_report['removed_points']} points removed")
            
            # Step 2: Outlier detection and removal
            if params["remove_outliers"]:
                processed_lc, outlier_report = await self.outlier_detector.remove_outliers(
                    processed_lc,
                    sigma=params["sigma_clip_sigma"],
                    maxiters=params["sigma_clip_maxiters"]
                )
                report["processing_steps"].append({
                    "step": "outlier_removal",
                    "points_before": outlier_report["points_before"],
                    "points_after": outlier_report["points_after"],
                    "outliers_removed": outlier_report["outliers_removed"],
                    "outlier_percentage": outlier_report["outlier_percentage"]
                })
                logger.info(f"Outlier removal: {outlier_report['outliers_removed']} outliers removed")
            
            # Step 3: Gap detection and segmentation
            segments = await self._detect_gaps_and_segment(
                processed_lc, 
                gap_threshold_hours=params["gap_threshold_hours"],
                min_points=params["min_points_per_segment"]
            )
            
            if len(segments) > 1:
                report["processing_steps"].append({
                    "step": "gap_detection",
                    "segments_found": len(segments),
                    "segment_sizes": [len(seg["indices"]) for seg in segments]
                })
                logger.info(f"Found {len(segments)} data segments")
            
            # Step 4: Baseline removal (detrending) per segment
            processed_segments = []
            for i, segment in enumerate(segments):
                segment_lc = self._extract_segment(processed_lc, segment["indices"])
                
                # Apply Savitzky-Golay baseline removal
                detrended_lc = await self._remove_baseline_savgol(
                    segment_lc,
                    window_length=min(params["baseline_window_length"], len(segment_lc.flux) // 4 * 2 + 1),
                    polyorder=params["baseline_polyorder"]
                )
                
                processed_segments.append({
                    "lightcurve": detrended_lc,
                    "indices": segment["indices"],
                    "gap_before": segment["gap_before"],
                    "gap_after": segment["gap_after"]
                })
            
            # Combine segments back
            processed_lc = self._combine_segments(processed_segments)
            processed_lc.detrended = True
            
            report["processing_steps"].append({
                "step": "baseline_removal",
                "method": "savitzky_golay",
                "window_length": params["baseline_window_length"],
                "polyorder": params["baseline_polyorder"],
                "segments_processed": len(segments)
            })
            
            # Step 5: Wavelet denoising (optional)
            if params["wavelet_denoising"]:
                processed_lc, denoise_report = await self.denoiser.denoise_wavelet(
                    processed_lc,
                    wavelet=params["wavelet_type"],
                    levels=params["wavelet_levels"]
                )
                report["processing_steps"].append({
                    "step": "wavelet_denoising",
                    "wavelet_type": params["wavelet_type"],
                    "levels": params["wavelet_levels"],
                    "noise_reduction_db": denoise_report["noise_reduction_db"]
                })
                logger.info(f"Wavelet denoising: {denoise_report['noise_reduction_db']:.2f} dB noise reduction")
            
            # Step 6: Normalization
            processed_lc, norm_report = await self.normalizer.normalize_flux(
                processed_lc, method=params["normalize_method"]
            )
            processed_lc.normalized = True
            
            report["processing_steps"].append({
                "step": "normalization",
                "method": params["normalize_method"],
                "normalization_factor": norm_report["normalization_factor"],
                "flux_std_before": norm_report["flux_std_before"],
                "flux_std_after": norm_report["flux_std_after"]
            })
            
            # Step 7: Final quality assessment
            quality_metrics = await self._calculate_quality_metrics(processed_lc)
            report["quality_metrics"] = quality_metrics
            
            report["final_points"] = len(processed_lc.time_bjd)
            report["processing_efficiency"] = report["final_points"] / report["original_points"]
            report["status"] = "success"
            
            logger.info(f"Processing completed: {report['original_points']} → {report['final_points']} points ({report['processing_efficiency']:.2%} retained)")
            
            return processed_lc, report
            
        except Exception as e:
            logger.error(f"Light curve processing failed: {e}")
            report["status"] = "error"
            report["error_message"] = str(e)
            return lightcurve_data, report

    async def _detect_gaps_and_segment(
        self,
        lightcurve: LightCurveData,
        gap_threshold_hours: float = 0.5,
        min_points: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Detect gaps in time series and create segments
        
        Args:
            lightcurve: Light curve data
            gap_threshold_hours: Minimum gap size to split on (hours)
            min_points: Minimum points per segment
            
        Returns:
            List of segment information
        """
        time_diffs = np.diff(lightcurve.time_bjd)
        gap_threshold_days = gap_threshold_hours / 24.0
        
        # Find gap indices
        gap_indices = np.where(time_diffs > gap_threshold_days)[0]
        
        # Create segments
        segments = []
        start_idx = 0
        
        for gap_idx in gap_indices:
            end_idx = gap_idx + 1
            
            if end_idx - start_idx >= min_points:
                segments.append({
                    "indices": np.arange(start_idx, end_idx),
                    "start_time": lightcurve.time_bjd[start_idx],
                    "end_time": lightcurve.time_bjd[end_idx - 1],
                    "gap_before": time_diffs[gap_idx] if gap_idx < len(time_diffs) else 0,
                    "gap_after": 0
                })
            
            start_idx = end_idx
        
        # Add final segment
        if len(lightcurve.time_bjd) - start_idx >= min_points:
            segments.append({
                "indices": np.arange(start_idx, len(lightcurve.time_bjd)),
                "start_time": lightcurve.time_bjd[start_idx],
                "end_time": lightcurve.time_bjd[-1],
                "gap_before": 0,
                "gap_after": 0
            })
        
        # If no valid segments found, use entire light curve
        if not segments:
            segments.append({
                "indices": np.arange(len(lightcurve.time_bjd)),
                "start_time": lightcurve.time_bjd[0],
                "end_time": lightcurve.time_bjd[-1],
                "gap_before": 0,
                "gap_after": 0
            })
        
        return segments

    def _extract_segment(self, lightcurve: LightCurveData, indices: np.ndarray) -> LightCurveData:
        """Extract a segment from light curve data"""
        return LightCurveData(
            target_name=lightcurve.target_name,
            time_bjd=lightcurve.time_bjd[indices],
            flux=lightcurve.flux[indices],
            flux_err=lightcurve.flux_err[indices],
            mission=lightcurve.mission,
            instrument=lightcurve.instrument,
            cadence_minutes=lightcurve.cadence_minutes,
            sectors_quarters=lightcurve.sectors_quarters,
            data_quality_flags=lightcurve.data_quality_flags[indices] if lightcurve.data_quality_flags is not None else None,
            detrended=lightcurve.detrended,
            normalized=lightcurve.normalized,
            outliers_removed=lightcurve.outliers_removed,
            source=lightcurve.source,
            download_date=lightcurve.download_date
        )

    def _combine_segments(self, segments: List[Dict[str, Any]]) -> LightCurveData:
        """Combine processed segments back into single light curve"""
        if not segments:
            raise ValueError("No segments to combine")
        
        # Get reference from first segment
        ref_lc = segments[0]["lightcurve"]
        
        # Combine all data
        all_time = np.concatenate([seg["lightcurve"].time_bjd for seg in segments])
        all_flux = np.concatenate([seg["lightcurve"].flux for seg in segments])
        all_flux_err = np.concatenate([seg["lightcurve"].flux_err for seg in segments])
        
        all_quality = None
        if ref_lc.data_quality_flags is not None:
            all_quality = np.concatenate([
                seg["lightcurve"].data_quality_flags for seg in segments
                if seg["lightcurve"].data_quality_flags is not None
            ])
        
        # Sort by time
        sort_indices = np.argsort(all_time)
        
        return LightCurveData(
            target_name=ref_lc.target_name,
            time_bjd=all_time[sort_indices],
            flux=all_flux[sort_indices],
            flux_err=all_flux_err[sort_indices],
            mission=ref_lc.mission,
            instrument=ref_lc.instrument,
            cadence_minutes=ref_lc.cadence_minutes,
            sectors_quarters=ref_lc.sectors_quarters,
            data_quality_flags=all_quality[sort_indices] if all_quality is not None else None,
            detrended=True,
            normalized=ref_lc.normalized,
            outliers_removed=ref_lc.outliers_removed,
            source=ref_lc.source,
            download_date=ref_lc.download_date
        )

    async def _remove_baseline_savgol(
        self,
        lightcurve: LightCurveData,
        window_length: int = 101,
        polyorder: int = 2
    ) -> LightCurveData:
        """
        Remove baseline using Savitzky-Golay filter
        
        Args:
            lightcurve: Input light curve
            window_length: Window length for filter (must be odd)
            polyorder: Polynomial order
            
        Returns:
            Detrended light curve
        """
        # Ensure window length is odd and reasonable
        if window_length % 2 == 0:
            window_length += 1
        
        window_length = min(window_length, len(lightcurve.flux) // 4 * 2 + 1)
        window_length = max(window_length, polyorder + 2)
        
        if window_length >= len(lightcurve.flux):
            # If window is too large, use simple median detrending
            baseline = np.median(lightcurve.flux)
            detrended_flux = lightcurve.flux / baseline
        else:
            # Apply Savitzky-Golay smoothing to estimate baseline
            baseline = savgol_filter(lightcurve.flux, window_length, polyorder)
            
            # Avoid division by zero
            baseline = np.where(baseline <= 0, np.median(lightcurve.flux), baseline)
            detrended_flux = lightcurve.flux / baseline
        
        # Create detrended light curve
        detrended_lc = LightCurveData(
            target_name=lightcurve.target_name,
            time_bjd=lightcurve.time_bjd.copy(),
            flux=detrended_flux,
            flux_err=lightcurve.flux_err / np.median(baseline) if hasattr(baseline, '__len__') else lightcurve.flux_err / baseline,
            mission=lightcurve.mission,
            instrument=lightcurve.instrument,
            cadence_minutes=lightcurve.cadence_minutes,
            sectors_quarters=lightcurve.sectors_quarters,
            data_quality_flags=lightcurve.data_quality_flags.copy() if lightcurve.data_quality_flags is not None else None,
            detrended=True,
            normalized=lightcurve.normalized,
            outliers_removed=lightcurve.outliers_removed,
            source=lightcurve.source,
            download_date=lightcurve.download_date
        )
        
        return detrended_lc

    async def _calculate_quality_metrics(self, lightcurve: LightCurveData) -> Dict[str, float]:
        """Calculate quality metrics for processed light curve"""
        flux = lightcurve.flux
        time = lightcurve.time_bjd
        
        metrics = {}
        
        # Basic statistics
        metrics["flux_mean"] = float(np.mean(flux))
        metrics["flux_std"] = float(np.std(flux))
        metrics["flux_median"] = float(np.median(flux))
        metrics["flux_mad"] = float(np.median(np.abs(flux - np.median(flux))))
        
        # Variability metrics
        metrics["rms_variability"] = float(np.sqrt(np.mean((flux - np.mean(flux))**2)))
        metrics["peak_to_peak"] = float(np.max(flux) - np.min(flux))
        
        # Time series metrics
        time_diffs = np.diff(time)
        metrics["median_cadence_minutes"] = float(np.median(time_diffs) * 24 * 60)
        metrics["cadence_std_minutes"] = float(np.std(time_diffs) * 24 * 60)
        metrics["duty_cycle"] = float(len(time) * np.median(time_diffs) / (time[-1] - time[0]))
        
        # Data quality metrics
        metrics["data_points"] = len(flux)
        metrics["time_span_days"] = float(time[-1] - time[0])
        metrics["points_per_day"] = metrics["data_points"] / metrics["time_span_days"]
        
        # Noise characteristics
        if len(flux) > 10:
            # Estimate photon noise level
            flux_err_median = float(np.median(lightcurve.flux_err))
            metrics["median_flux_error"] = flux_err_median
            metrics["snr_estimate"] = metrics["flux_median"] / flux_err_median if flux_err_median > 0 else 0
            
            # Estimate correlated noise using Allan variance
            if len(flux) > 100:
                allan_var = self._calculate_allan_variance(flux)
                metrics["allan_variance"] = float(allan_var)
        
        return metrics

    def _calculate_allan_variance(self, flux: np.ndarray, max_tau: int = 50) -> float:
        """
        Calculate Allan variance to estimate correlated noise
        
        Args:
            flux: Flux array
            max_tau: Maximum tau for calculation
            
        Returns:
            Allan variance at tau=2
        """
        try:
            n = len(flux)
            max_tau = min(max_tau, n // 3)
            
            if max_tau < 2:
                return 0.0
            
            # Calculate Allan variance for tau=2 (most common)
            tau = 2
            if n >= 3 * tau:
                diff_sum = 0.0
                count = 0
                
                for i in range(n - 2 * tau):
                    avg1 = np.mean(flux[i:i + tau])
                    avg2 = np.mean(flux[i + tau:i + 2 * tau])
                    diff_sum += (avg2 - avg1) ** 2
                    count += 1
                
                if count > 0:
                    allan_var = diff_sum / (2 * count)
                    return allan_var
            
            return 0.0
            
        except Exception:
            return 0.0

    async def preprocess_for_transit_search(
        self,
        lightcurve_data: LightCurveData,
        transit_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[LightCurveData, Dict[str, Any]]:
        """
        Specialized preprocessing for transit search
        
        Args:
            lightcurve_data: Input light curve
            transit_params: Transit-specific parameters
            
        Returns:
            Tuple of (processed_lightcurve, processing_report)
        """
        # Transit-optimized parameters
        transit_processing_params = {
            "remove_outliers": True,
            "sigma_clip_sigma": 4.0,  # More conservative for transits
            "sigma_clip_maxiters": 2,
            "baseline_window_length": 201,  # Longer window to preserve transits
            "baseline_polyorder": 3,
            "wavelet_denoising": True,  # Enable for better transit detection
            "wavelet_type": "db6",
            "wavelet_levels": 4,
            "normalize_method": "robust",
            "quality_bitmask": "hard",
            "gap_threshold_hours": 2.0,  # Larger gaps for transit search
            "min_points_per_segment": 200
        }
        
        # Override with user parameters
        if transit_params:
            transit_processing_params.update(transit_params)
        
        # Process with transit-optimized parameters
        processed_lc, report = await self.process_lightcurve(
            lightcurve_data, transit_processing_params
        )
        
        # Add transit-specific quality checks
        if report["status"] == "success":
            transit_metrics = await self._calculate_transit_metrics(processed_lc)
            report["transit_metrics"] = transit_metrics
        
        return processed_lc, report

    async def _calculate_transit_metrics(self, lightcurve: LightCurveData) -> Dict[str, float]:
        """Calculate metrics relevant for transit detection"""
        flux = lightcurve.flux
        
        metrics = {}
        
        # Transit detection metrics
        metrics["flux_precision_ppm"] = float(np.std(flux) * 1e6)
        metrics["median_absolute_deviation_ppm"] = float(np.median(np.abs(flux - np.median(flux))) * 1e6)
        
        # Estimate minimum detectable transit depth (3-sigma)
        metrics["min_detectable_depth_ppm"] = float(3 * np.std(flux) * 1e6)
        
        # Data coverage metrics
        time_span_hours = (lightcurve.time_bjd[-1] - lightcurve.time_bjd[0]) * 24
        metrics["observing_time_hours"] = float(time_span_hours)
        metrics["data_density_per_hour"] = len(flux) / time_span_hours
        
        return metrics
