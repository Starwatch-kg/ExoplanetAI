"""
Advanced denoising system for light curve data
Система шумоподавления для данных кривых блеска
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pywt
from scipy import signal
from scipy.ndimage import median_filter

from data_sources.base import LightCurveData

logger = logging.getLogger(__name__)


class WaveletDenoiser:
    """
    Advanced wavelet-based denoising system for light curve data
    """

    def __init__(self):
        # Available wavelets and their characteristics
        self.wavelets = {
            "db4": {"family": "Daubechies", "vanishing_moments": 4, "support": 7},
            "db6": {"family": "Daubechies", "vanishing_moments": 6, "support": 11},
            "db8": {"family": "Daubechies", "vanishing_moments": 8, "support": 15},
            "haar": {"family": "Haar", "vanishing_moments": 1, "support": 1},
            "bior2.2": {"family": "Biorthogonal", "vanishing_moments": 2, "support": 5},
            "bior4.4": {"family": "Biorthogonal", "vanishing_moments": 4, "support": 9},
            "coif2": {"family": "Coiflets", "vanishing_moments": 2, "support": 5},
            "coif4": {"family": "Coiflets", "vanishing_moments": 4, "support": 11},
            "sym4": {"family": "Symlets", "vanishing_moments": 4, "support": 7},
            "sym8": {"family": "Symlets", "vanishing_moments": 8, "support": 15}
        }
        
        # Thresholding methods
        self.threshold_methods = {
            "soft": "Soft thresholding - gradual transition",
            "hard": "Hard thresholding - sharp cutoff"
        }

    async def denoise_wavelet(
        self,
        lightcurve: LightCurveData,
        wavelet: str = "db6",
        levels: int = 6,
        threshold_method: str = "soft",
        sigma_estimate: Optional[float] = None
    ) -> Tuple[LightCurveData, Dict[str, Any]]:
        """
        Apply wavelet denoising to light curve data
        
        Args:
            lightcurve: Input light curve data
            wavelet: Wavelet type to use
            levels: Number of decomposition levels
            threshold_method: Thresholding method ("soft" or "hard")
            sigma_estimate: Noise level estimate (auto-estimated if None)
            
        Returns:
            Tuple of (denoised_lightcurve, denoising_report)
        """
        logger.info(f"Applying wavelet denoising with {wavelet} wavelet, {levels} levels")
        
        flux = lightcurve.flux.copy()
        original_flux = flux.copy()
        
        # Validate wavelet
        if wavelet not in self.wavelets:
            logger.warning(f"Unknown wavelet '{wavelet}', using 'db6'")
            wavelet = "db6"
        
        try:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(flux, wavelet, level=levels)
            
            # Estimate noise level if not provided
            if sigma_estimate is None:
                # Use median absolute deviation of finest detail coefficients
                sigma_estimate = np.median(np.abs(coeffs[-1])) / 0.6745
            
            # Calculate threshold using universal threshold
            n = len(flux)
            threshold = sigma_estimate * np.sqrt(2 * np.log(n))
            
            # Apply thresholding to detail coefficients
            coeffs_thresh = coeffs.copy()
            for i in range(1, len(coeffs)):  # Skip approximation coefficients
                if threshold_method == "soft":
                    coeffs_thresh[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
                else:
                    coeffs_thresh[i] = pywt.threshold(coeffs[i], threshold, mode='hard')
            
            # Reconstruct signal
            denoised_flux = pywt.waverec(coeffs_thresh, wavelet)
            
            # Handle length mismatch due to boundary conditions
            if len(denoised_flux) != len(flux):
                if len(denoised_flux) > len(flux):
                    denoised_flux = denoised_flux[:len(flux)]
                else:
                    # Pad with original values
                    pad_length = len(flux) - len(denoised_flux)
                    denoised_flux = np.pad(denoised_flux, (0, pad_length), mode='edge')
            
            # Calculate denoising metrics
            noise_removed = original_flux - denoised_flux
            noise_power_original = np.var(original_flux)
            noise_power_denoised = np.var(denoised_flux)
            
            # Signal-to-noise ratio improvement
            snr_improvement_db = 10 * np.log10(noise_power_original / noise_power_denoised) if noise_power_denoised > 0 else 0
            
            # Create denoised light curve
            denoised_lc = LightCurveData(
                target_name=lightcurve.target_name,
                time_bjd=lightcurve.time_bjd.copy(),
                flux=denoised_flux,
                flux_err=lightcurve.flux_err.copy(),  # Keep original error estimates
                mission=lightcurve.mission,
                instrument=lightcurve.instrument,
                cadence_minutes=lightcurve.cadence_minutes,
                sectors_quarters=lightcurve.sectors_quarters,
                data_quality_flags=lightcurve.data_quality_flags.copy() if lightcurve.data_quality_flags is not None else None,
                detrended=lightcurve.detrended,
                normalized=lightcurve.normalized,
                outliers_removed=lightcurve.outliers_removed,
                source=lightcurve.source,
                download_date=lightcurve.download_date
            )
            
            # Generate report
            report = {
                "wavelet_used": wavelet,
                "decomposition_levels": levels,
                "threshold_method": threshold_method,
                "noise_estimate": float(sigma_estimate),
                "threshold_value": float(threshold),
                "noise_reduction_db": float(snr_improvement_db),
                "flux_std_before": float(np.std(original_flux)),
                "flux_std_after": float(np.std(denoised_flux)),
                "coefficients_modified": len(coeffs) - 1,  # Exclude approximation
                "status": "success"
            }
            
            logger.info(f"Wavelet denoising completed: {snr_improvement_db:.2f} dB noise reduction")
            
            return denoised_lc, report
            
        except Exception as e:
            logger.error(f"Wavelet denoising failed: {e}")
            return lightcurve, {
                "status": "error",
                "error_message": str(e),
                "wavelet_used": wavelet,
                "decomposition_levels": levels
            }

    async def denoise_median_filter(
        self,
        lightcurve: LightCurveData,
        kernel_size: int = 5
    ) -> Tuple[LightCurveData, Dict[str, Any]]:
        """
        Apply median filtering for impulse noise removal
        
        Args:
            lightcurve: Input light curve data
            kernel_size: Size of median filter kernel
            
        Returns:
            Tuple of (filtered_lightcurve, filtering_report)
        """
        logger.info(f"Applying median filter with kernel size {kernel_size}")
        
        try:
            original_flux = lightcurve.flux.copy()
            
            # Apply median filter
            filtered_flux = median_filter(original_flux, size=kernel_size)
            
            # Calculate filtering metrics
            noise_removed = original_flux - filtered_flux
            rms_noise_removed = np.sqrt(np.mean(noise_removed**2))
            
            # Create filtered light curve
            filtered_lc = LightCurveData(
                target_name=lightcurve.target_name,
                time_bjd=lightcurve.time_bjd.copy(),
                flux=filtered_flux,
                flux_err=lightcurve.flux_err.copy(),
                mission=lightcurve.mission,
                instrument=lightcurve.instrument,
                cadence_minutes=lightcurve.cadence_minutes,
                sectors_quarters=lightcurve.sectors_quarters,
                data_quality_flags=lightcurve.data_quality_flags.copy() if lightcurve.data_quality_flags is not None else None,
                detrended=lightcurve.detrended,
                normalized=lightcurve.normalized,
                outliers_removed=lightcurve.outliers_removed,
                source=lightcurve.source,
                download_date=lightcurve.download_date
            )
            
            report = {
                "filter_type": "median",
                "kernel_size": kernel_size,
                "rms_noise_removed": float(rms_noise_removed),
                "flux_std_before": float(np.std(original_flux)),
                "flux_std_after": float(np.std(filtered_flux)),
                "status": "success"
            }
            
            logger.info(f"Median filtering completed: RMS noise removed = {rms_noise_removed:.6f}")
            
            return filtered_lc, report
            
        except Exception as e:
            logger.error(f"Median filtering failed: {e}")
            return lightcurve, {
                "status": "error",
                "error_message": str(e),
                "filter_type": "median",
                "kernel_size": kernel_size
            }

    async def denoise_gaussian_filter(
        self,
        lightcurve: LightCurveData,
        sigma: float = 1.0
    ) -> Tuple[LightCurveData, Dict[str, Any]]:
        """
        Apply Gaussian smoothing filter
        
        Args:
            lightcurve: Input light curve data
            sigma: Standard deviation of Gaussian kernel
            
        Returns:
            Tuple of (smoothed_lightcurve, smoothing_report)
        """
        logger.info(f"Applying Gaussian filter with sigma = {sigma}")
        
        try:
            from scipy.ndimage import gaussian_filter1d
            
            original_flux = lightcurve.flux.copy()
            
            # Apply Gaussian filter
            smoothed_flux = gaussian_filter1d(original_flux, sigma=sigma)
            
            # Calculate smoothing metrics
            smoothing_factor = np.std(original_flux) / np.std(smoothed_flux) if np.std(smoothed_flux) > 0 else 1.0
            
            # Create smoothed light curve
            smoothed_lc = LightCurveData(
                target_name=lightcurve.target_name,
                time_bjd=lightcurve.time_bjd.copy(),
                flux=smoothed_flux,
                flux_err=lightcurve.flux_err.copy(),
                mission=lightcurve.mission,
                instrument=lightcurve.instrument,
                cadence_minutes=lightcurve.cadence_minutes,
                sectors_quarters=lightcurve.sectors_quarters,
                data_quality_flags=lightcurve.data_quality_flags.copy() if lightcurve.data_quality_flags is not None else None,
                detrended=lightcurve.detrended,
                normalized=lightcurve.normalized,
                outliers_removed=lightcurve.outliers_removed,
                source=lightcurve.source,
                download_date=lightcurve.download_date
            )
            
            report = {
                "filter_type": "gaussian",
                "sigma": float(sigma),
                "smoothing_factor": float(smoothing_factor),
                "flux_std_before": float(np.std(original_flux)),
                "flux_std_after": float(np.std(smoothed_flux)),
                "status": "success"
            }
            
            logger.info(f"Gaussian filtering completed: smoothing factor = {smoothing_factor:.2f}")
            
            return smoothed_lc, report
            
        except Exception as e:
            logger.error(f"Gaussian filtering failed: {e}")
            return lightcurve, {
                "status": "error",
                "error_message": str(e),
                "filter_type": "gaussian",
                "sigma": sigma
            }

    async def adaptive_denoise(
        self,
        lightcurve: LightCurveData,
        noise_threshold: float = 0.001
    ) -> Tuple[LightCurveData, Dict[str, Any]]:
        """
        Adaptive denoising that selects optimal parameters based on data characteristics
        
        Args:
            lightcurve: Input light curve data
            noise_threshold: Threshold for noise level decision
            
        Returns:
            Tuple of (denoised_lightcurve, denoising_report)
        """
        logger.info("Applying adaptive denoising")
        
        try:
            flux = lightcurve.flux
            
            # Analyze data characteristics
            flux_std = np.std(flux)
            flux_range = np.max(flux) - np.min(flux)
            
            # Estimate noise level using high-frequency components
            # Apply high-pass filter to isolate noise
            b, a = signal.butter(4, 0.1, btype='high', fs=1.0)
            high_freq = signal.filtfilt(b, a, flux - np.mean(flux))
            noise_estimate = np.std(high_freq)
            
            # Select denoising strategy based on noise characteristics
            if noise_estimate > noise_threshold:
                # High noise - use wavelet denoising
                if len(flux) > 1000:
                    # Long time series - use more levels
                    denoised_lc, report = await self.denoise_wavelet(
                        lightcurve, wavelet="db6", levels=6, threshold_method="soft"
                    )
                    strategy = "wavelet_aggressive"
                else:
                    # Short time series - fewer levels
                    denoised_lc, report = await self.denoise_wavelet(
                        lightcurve, wavelet="db4", levels=4, threshold_method="soft"
                    )
                    strategy = "wavelet_conservative"
            else:
                # Low noise - use gentle median filter
                kernel_size = min(5, len(flux) // 20)
                if kernel_size < 3:
                    kernel_size = 3
                
                denoised_lc, report = await self.denoise_median_filter(
                    lightcurve, kernel_size=kernel_size
                )
                strategy = "median_gentle"
            
            # Add adaptive strategy info to report
            report["adaptive_strategy"] = strategy
            report["noise_estimate"] = float(noise_estimate)
            report["noise_threshold"] = float(noise_threshold)
            report["data_characteristics"] = {
                "flux_std": float(flux_std),
                "flux_range": float(flux_range),
                "data_points": len(flux)
            }
            
            logger.info(f"Adaptive denoising completed using strategy: {strategy}")
            
            return denoised_lc, report
            
        except Exception as e:
            logger.error(f"Adaptive denoising failed: {e}")
            return lightcurve, {
                "status": "error",
                "error_message": str(e),
                "adaptive_strategy": "failed"
            }

    def get_wavelet_info(self) -> Dict[str, Any]:
        """Get information about available wavelets"""
        return {
            "available_wavelets": self.wavelets,
            "threshold_methods": self.threshold_methods,
            "recommendations": {
                "general_purpose": "db6",
                "sharp_features": "haar",
                "smooth_signals": "coif4",
                "asymmetric_features": "bior4.4"
            }
        }

    async def analyze_noise_characteristics(
        self,
        lightcurve: LightCurveData
    ) -> Dict[str, Any]:
        """
        Analyze noise characteristics in light curve data
        
        Args:
            lightcurve: Light curve data to analyze
            
        Returns:
            Noise analysis report
        """
        flux = lightcurve.flux
        
        try:
            # Basic noise statistics
            flux_std = np.std(flux)
            flux_mad = np.median(np.abs(flux - np.median(flux)))
            
            # Estimate white noise level using difference method
            diff_noise = np.std(np.diff(flux)) / np.sqrt(2)
            
            # Frequency domain analysis
            if len(flux) > 64:
                freqs, psd = signal.welch(flux - np.mean(flux), nperseg=min(256, len(flux)//4))
                
                # Estimate noise floor from high frequencies
                high_freq_mask = freqs > 0.4  # High frequency region
                if np.any(high_freq_mask):
                    noise_floor = np.median(psd[high_freq_mask])
                else:
                    noise_floor = np.median(psd)
                
                # Check for colored noise
                low_freq_power = np.mean(psd[freqs < 0.1])
                high_freq_power = np.mean(psd[freqs > 0.3])
                color_ratio = low_freq_power / high_freq_power if high_freq_power > 0 else 1.0
            else:
                noise_floor = flux_std**2
                color_ratio = 1.0
            
            # Wavelet-based noise analysis
            coeffs = pywt.wavedec(flux, 'db4', level=3)
            detail_stds = [np.std(c) for c in coeffs[1:]]  # Detail coefficients only
            
            analysis = {
                "flux_statistics": {
                    "std": float(flux_std),
                    "mad": float(flux_mad),
                    "range": float(np.max(flux) - np.min(flux))
                },
                "noise_estimates": {
                    "difference_method": float(diff_noise),
                    "noise_floor_psd": float(noise_floor),
                    "wavelet_detail_std": [float(std) for std in detail_stds]
                },
                "noise_characteristics": {
                    "color_ratio": float(color_ratio),
                    "is_white_noise": color_ratio < 2.0,
                    "dominant_noise_type": "white" if color_ratio < 2.0 else "colored"
                },
                "denoising_recommendations": []
            }
            
            # Generate recommendations
            if flux_std > 0.01:
                analysis["denoising_recommendations"].append("High noise level - consider wavelet denoising")
            
            if color_ratio > 3.0:
                analysis["denoising_recommendations"].append("Colored noise detected - use adaptive filtering")
            
            if diff_noise / flux_std > 0.7:
                analysis["denoising_recommendations"].append("High frequency noise - median filter may help")
            
            if not analysis["denoising_recommendations"]:
                analysis["denoising_recommendations"].append("Low noise level - minimal filtering needed")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Noise analysis failed: {e}")
            return {
                "status": "error",
                "error_message": str(e)
            }
