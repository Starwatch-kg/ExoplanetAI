"""
Flux normalization system for light curve data
Система нормализации потока для данных кривых блеска
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from data_sources.base import LightCurveData

logger = logging.getLogger(__name__)


class FluxNormalizer:
    """
    Advanced flux normalization system with multiple methods
    """

    def __init__(self):
        self.normalization_methods = {
            "median": "Normalize by median flux",
            "mean": "Normalize by mean flux", 
            "robust": "Robust normalization using percentiles",
            "unity": "Scale to unit variance",
            "minmax": "Min-max normalization to [0,1]",
            "zscore": "Z-score normalization (mean=0, std=1)",
            "quantile": "Quantile-based normalization"
        }

    async def normalize_flux(
        self,
        lightcurve: LightCurveData,
        method: str = "median"
    ) -> Tuple[LightCurveData, Dict[str, Any]]:
        """
        Normalize light curve flux using specified method
        
        Args:
            lightcurve: Input light curve data
            method: Normalization method
            
        Returns:
            Tuple of (normalized_lightcurve, normalization_report)
        """
        logger.info(f"Normalizing flux using method: {method}")
        
        if method not in self.normalization_methods:
            logger.warning(f"Unknown normalization method '{method}', using 'median'")
            method = "median"
        
        try:
            flux = lightcurve.flux.copy()
            flux_err = lightcurve.flux_err.copy()
            original_flux = flux.copy()
            
            # Apply normalization method
            if method == "median":
                norm_factor = np.median(flux)
                normalized_flux = flux / norm_factor
                normalized_flux_err = flux_err / norm_factor
                
            elif method == "mean":
                norm_factor = np.mean(flux)
                normalized_flux = flux / norm_factor
                normalized_flux_err = flux_err / norm_factor
                
            elif method == "robust":
                # Use 5th and 95th percentiles for robust normalization
                p5, p95 = np.percentile(flux, [5, 95])
                norm_factor = p95 - p5
                offset = p5
                normalized_flux = (flux - offset) / norm_factor
                normalized_flux_err = flux_err / norm_factor
                
            elif method == "unity":
                # Scale to unit variance, preserve mean
                flux_mean = np.mean(flux)
                flux_std = np.std(flux)
                normalized_flux = (flux - flux_mean) / flux_std + 1.0
                normalized_flux_err = flux_err / flux_std
                norm_factor = flux_std
                
            elif method == "minmax":
                # Scale to [0, 1] range
                flux_min = np.min(flux)
                flux_max = np.max(flux)
                norm_factor = flux_max - flux_min
                normalized_flux = (flux - flux_min) / norm_factor
                normalized_flux_err = flux_err / norm_factor
                
            elif method == "zscore":
                # Z-score normalization (mean=0, std=1)
                flux_mean = np.mean(flux)
                flux_std = np.std(flux)
                normalized_flux = (flux - flux_mean) / flux_std
                normalized_flux_err = flux_err / flux_std
                norm_factor = flux_std
                
            elif method == "quantile":
                # Quantile-based normalization using median and MAD
                flux_median = np.median(flux)
                flux_mad = np.median(np.abs(flux - flux_median))
                normalized_flux = (flux - flux_median) / (1.4826 * flux_mad) + 1.0
                normalized_flux_err = flux_err / (1.4826 * flux_mad)
                norm_factor = 1.4826 * flux_mad
            
            else:
                # Fallback to median
                norm_factor = np.median(flux)
                normalized_flux = flux / norm_factor
                normalized_flux_err = flux_err / norm_factor
            
            # Create normalized light curve
            normalized_lc = LightCurveData(
                target_name=lightcurve.target_name,
                time_bjd=lightcurve.time_bjd.copy(),
                flux=normalized_flux,
                flux_err=normalized_flux_err,
                mission=lightcurve.mission,
                instrument=lightcurve.instrument,
                cadence_minutes=lightcurve.cadence_minutes,
                sectors_quarters=lightcurve.sectors_quarters,
                data_quality_flags=lightcurve.data_quality_flags.copy() if lightcurve.data_quality_flags is not None else None,
                detrended=lightcurve.detrended,
                normalized=True,
                outliers_removed=lightcurve.outliers_removed,
                source=lightcurve.source,
                download_date=lightcurve.download_date
            )
            
            # Generate normalization report
            report = {
                "method": method,
                "normalization_factor": float(norm_factor) if hasattr(norm_factor, '__float__') else norm_factor,
                "flux_stats_before": {
                    "mean": float(np.mean(original_flux)),
                    "median": float(np.median(original_flux)),
                    "std": float(np.std(original_flux)),
                    "min": float(np.min(original_flux)),
                    "max": float(np.max(original_flux))
                },
                "flux_stats_after": {
                    "mean": float(np.mean(normalized_flux)),
                    "median": float(np.median(normalized_flux)),
                    "std": float(np.std(normalized_flux)),
                    "min": float(np.min(normalized_flux)),
                    "max": float(np.max(normalized_flux))
                },
                "flux_std_before": float(np.std(original_flux)),
                "flux_std_after": float(np.std(normalized_flux)),
                "status": "success"
            }
            
            logger.info(f"Flux normalization completed using {method} method")
            
            return normalized_lc, report
            
        except Exception as e:
            logger.error(f"Flux normalization failed: {e}")
            return lightcurve, {
                "status": "error",
                "error_message": str(e),
                "method": method
            }

    async def auto_normalize(
        self,
        lightcurve: LightCurveData
    ) -> Tuple[LightCurveData, Dict[str, Any]]:
        """
        Automatically select optimal normalization method based on data characteristics
        
        Args:
            lightcurve: Input light curve data
            
        Returns:
            Tuple of (normalized_lightcurve, normalization_report)
        """
        logger.info("Applying automatic normalization")
        
        try:
            flux = lightcurve.flux
            
            # Analyze flux characteristics
            flux_mean = np.mean(flux)
            flux_median = np.median(flux)
            flux_std = np.std(flux)
            flux_mad = np.median(np.abs(flux - flux_median))
            
            # Check for outliers
            outlier_fraction = np.sum(np.abs(flux - flux_median) > 3 * flux_mad) / len(flux)
            
            # Check distribution shape
            skewness = stats.skew(flux)
            kurtosis = stats.kurtosis(flux)
            
            # Select normalization method based on characteristics
            if outlier_fraction > 0.05:  # More than 5% outliers
                if abs(skewness) > 1.0:  # Highly skewed
                    method = "quantile"
                    reason = "High outlier fraction with skewed distribution"
                else:
                    method = "robust"
                    reason = "High outlier fraction"
            elif abs(skewness) > 0.5:  # Moderately skewed
                method = "median"
                reason = "Skewed distribution"
            elif flux_std / flux_mean > 0.1:  # High variability
                method = "robust"
                reason = "High flux variability"
            else:
                method = "median"
                reason = "Well-behaved distribution"
            
            # Apply selected normalization
            normalized_lc, report = await self.normalize_flux(lightcurve, method)
            
            # Add auto-selection info to report
            report["auto_selection"] = {
                "selected_method": method,
                "reason": reason,
                "data_characteristics": {
                    "outlier_fraction": float(outlier_fraction),
                    "skewness": float(skewness),
                    "kurtosis": float(kurtosis),
                    "coefficient_of_variation": float(flux_std / flux_mean)
                }
            }
            
            logger.info(f"Auto-normalization selected method: {method} ({reason})")
            
            return normalized_lc, report
            
        except Exception as e:
            logger.error(f"Auto-normalization failed: {e}")
            return lightcurve, {
                "status": "error",
                "error_message": str(e),
                "auto_selection": {"selected_method": "failed"}
            }

    async def denormalize_flux(
        self,
        normalized_lightcurve: LightCurveData,
        normalization_report: Dict[str, Any]
    ) -> Tuple[LightCurveData, Dict[str, Any]]:
        """
        Reverse normalization to restore original flux scale
        
        Args:
            normalized_lightcurve: Normalized light curve data
            normalization_report: Report from original normalization
            
        Returns:
            Tuple of (denormalized_lightcurve, denormalization_report)
        """
        logger.info("Reversing flux normalization")
        
        if normalization_report.get("status") != "success":
            return normalized_lightcurve, {
                "status": "error",
                "error_message": "Invalid normalization report"
            }
        
        try:
            method = normalization_report["method"]
            norm_factor = normalization_report["normalization_factor"]
            
            flux = normalized_lightcurve.flux.copy()
            flux_err = normalized_lightcurve.flux_err.copy()
            
            # Reverse normalization based on method
            if method == "median" or method == "mean":
                denorm_flux = flux * norm_factor
                denorm_flux_err = flux_err * norm_factor
                
            elif method == "robust":
                # Need additional parameters for robust denormalization
                # For now, use simple scaling
                denorm_flux = flux * norm_factor
                denorm_flux_err = flux_err * norm_factor
                
            elif method == "unity":
                # Reverse unit variance scaling
                original_mean = normalization_report["flux_stats_before"]["mean"]
                denorm_flux = (flux - 1.0) * norm_factor + original_mean
                denorm_flux_err = flux_err * norm_factor
                
            elif method == "minmax":
                # Reverse min-max scaling
                original_min = normalization_report["flux_stats_before"]["min"]
                denorm_flux = flux * norm_factor + original_min
                denorm_flux_err = flux_err * norm_factor
                
            elif method == "zscore":
                # Reverse z-score normalization
                original_mean = normalization_report["flux_stats_before"]["mean"]
                denorm_flux = flux * norm_factor + original_mean
                denorm_flux_err = flux_err * norm_factor
                
            elif method == "quantile":
                # Reverse quantile normalization
                original_median = normalization_report["flux_stats_before"]["median"]
                denorm_flux = (flux - 1.0) * norm_factor + original_median
                denorm_flux_err = flux_err * norm_factor
                
            else:
                # Default to simple scaling
                denorm_flux = flux * norm_factor
                denorm_flux_err = flux_err * norm_factor
            
            # Create denormalized light curve
            denormalized_lc = LightCurveData(
                target_name=normalized_lightcurve.target_name,
                time_bjd=normalized_lightcurve.time_bjd.copy(),
                flux=denorm_flux,
                flux_err=denorm_flux_err,
                mission=normalized_lightcurve.mission,
                instrument=normalized_lightcurve.instrument,
                cadence_minutes=normalized_lightcurve.cadence_minutes,
                sectors_quarters=normalized_lightcurve.sectors_quarters,
                data_quality_flags=normalized_lightcurve.data_quality_flags.copy() if normalized_lightcurve.data_quality_flags is not None else None,
                detrended=normalized_lightcurve.detrended,
                normalized=False,
                outliers_removed=normalized_lightcurve.outliers_removed,
                source=normalized_lightcurve.source,
                download_date=normalized_lightcurve.download_date
            )
            
            report = {
                "method_reversed": method,
                "normalization_factor_used": norm_factor,
                "flux_stats_recovered": {
                    "mean": float(np.mean(denorm_flux)),
                    "median": float(np.median(denorm_flux)),
                    "std": float(np.std(denorm_flux))
                },
                "status": "success"
            }
            
            logger.info(f"Flux denormalization completed for method: {method}")
            
            return denormalized_lc, report
            
        except Exception as e:
            logger.error(f"Flux denormalization failed: {e}")
            return normalized_lightcurve, {
                "status": "error",
                "error_message": str(e),
                "method_reversed": method
            }

    def get_normalization_info(self) -> Dict[str, Any]:
        """Get information about available normalization methods"""
        return {
            "available_methods": self.normalization_methods,
            "recommendations": {
                "general_purpose": "median",
                "high_outliers": "robust", 
                "skewed_distribution": "quantile",
                "machine_learning": "zscore",
                "visualization": "minmax"
            },
            "method_details": {
                "median": {
                    "description": "Divides by median flux value",
                    "preserves": "relative variations",
                    "robust_to_outliers": True
                },
                "mean": {
                    "description": "Divides by mean flux value",
                    "preserves": "relative variations",
                    "robust_to_outliers": False
                },
                "robust": {
                    "description": "Uses 5th-95th percentile range",
                    "preserves": "central distribution",
                    "robust_to_outliers": True
                },
                "unity": {
                    "description": "Scales to unit variance around mean=1",
                    "preserves": "variance structure",
                    "robust_to_outliers": False
                },
                "minmax": {
                    "description": "Scales to [0,1] range",
                    "preserves": "relative ordering",
                    "robust_to_outliers": False
                },
                "zscore": {
                    "description": "Centers at 0 with unit variance",
                    "preserves": "distribution shape",
                    "robust_to_outliers": False
                },
                "quantile": {
                    "description": "Uses median and MAD for scaling",
                    "preserves": "distribution shape",
                    "robust_to_outliers": True
                }
            }
        }

    async def compare_normalization_methods(
        self,
        lightcurve: LightCurveData,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare different normalization methods on the same data
        
        Args:
            lightcurve: Input light curve data
            methods: List of methods to compare (default: all methods)
            
        Returns:
            Comparison results
        """
        if methods is None:
            methods = list(self.normalization_methods.keys())
        
        logger.info(f"Comparing normalization methods: {methods}")
        
        comparison = {
            "original_stats": {
                "mean": float(np.mean(lightcurve.flux)),
                "median": float(np.median(lightcurve.flux)),
                "std": float(np.std(lightcurve.flux)),
                "min": float(np.min(lightcurve.flux)),
                "max": float(np.max(lightcurve.flux))
            },
            "method_results": {},
            "recommendations": []
        }
        
        for method in methods:
            if method in self.normalization_methods:
                try:
                    normalized_lc, report = await self.normalize_flux(lightcurve, method)
                    
                    if report["status"] == "success":
                        comparison["method_results"][method] = {
                            "normalization_factor": report["normalization_factor"],
                            "final_stats": report["flux_stats_after"],
                            "preservation_score": self._calculate_preservation_score(
                                lightcurve.flux, normalized_lc.flux
                            )
                        }
                    else:
                        comparison["method_results"][method] = {
                            "status": "failed",
                            "error": report.get("error_message", "Unknown error")
                        }
                        
                except Exception as e:
                    comparison["method_results"][method] = {
                        "status": "failed",
                        "error": str(e)
                    }
        
        # Generate recommendations based on results
        successful_methods = [m for m, r in comparison["method_results"].items() 
                            if r.get("status") != "failed"]
        
        if successful_methods:
            # Find method with best preservation score
            best_method = max(successful_methods, 
                            key=lambda m: comparison["method_results"][m].get("preservation_score", 0))
            
            comparison["recommendations"].append(f"Best overall: {best_method}")
            
            # Find most robust method
            robust_methods = ["median", "robust", "quantile"]
            available_robust = [m for m in robust_methods if m in successful_methods]
            if available_robust:
                comparison["recommendations"].append(f"Most robust: {available_robust[0]}")
        
        return comparison

    def _calculate_preservation_score(self, original_flux: np.ndarray, normalized_flux: np.ndarray) -> float:
        """Calculate how well normalization preserves relative structure"""
        try:
            # Calculate correlation between relative changes
            orig_diff = np.diff(original_flux)
            norm_diff = np.diff(normalized_flux)
            
            if len(orig_diff) > 1 and np.std(orig_diff) > 0 and np.std(norm_diff) > 0:
                correlation = np.corrcoef(orig_diff, norm_diff)[0, 1]
                return float(correlation) if not np.isnan(correlation) else 0.0
            else:
                return 0.0
                
        except Exception:
            return 0.0
