"""
Outlier detection and removal system for light curve data
Система обнаружения и удаления выбросов для данных кривых блеска
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.stats import sigma_clip
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from data_sources.base import LightCurveData

logger = logging.getLogger(__name__)


class OutlierDetector:
    """
    Advanced outlier detection system using multiple statistical methods
    """

    def __init__(self):
        self.detection_methods = {
            "sigma_clip": "Statistical sigma clipping",
            "mad": "Median Absolute Deviation",
            "iqr": "Interquartile Range",
            "isolation_forest": "Machine Learning Isolation Forest",
            "z_score": "Z-score based detection",
            "modified_z_score": "Modified Z-score using MAD",
            "grubbs": "Grubbs test for outliers",
            "ensemble": "Ensemble of multiple methods"
        }

    async def remove_outliers(
        self,
        lightcurve: LightCurveData,
        method: str = "sigma_clip",
        sigma: float = 5.0,
        maxiters: int = 3
    ) -> Tuple[LightCurveData, Dict[str, Any]]:
        """
        Remove outliers from light curve data
        
        Args:
            lightcurve: Input light curve data
            method: Outlier detection method
            sigma: Sigma threshold for statistical methods
            maxiters: Maximum iterations for iterative methods
            
        Returns:
            Tuple of (cleaned_lightcurve, outlier_report)
        """
        logger.info(f"Removing outliers using method: {method}")
        
        if method not in self.detection_methods:
            logger.warning(f"Unknown outlier method '{method}', using 'sigma_clip'")
            method = "sigma_clip"
        
        try:
            flux = lightcurve.flux
            points_before = len(flux)
            
            # Detect outliers using specified method
            if method == "sigma_clip":
                outlier_mask = await self._sigma_clip_outliers(flux, sigma, maxiters)
            elif method == "mad":
                outlier_mask = await self._mad_outliers(flux, sigma)
            elif method == "iqr":
                outlier_mask = await self._iqr_outliers(flux)
            elif method == "isolation_forest":
                outlier_mask = await self._isolation_forest_outliers(lightcurve)
            elif method == "z_score":
                outlier_mask = await self._z_score_outliers(flux, sigma)
            elif method == "modified_z_score":
                outlier_mask = await self._modified_z_score_outliers(flux, sigma)
            elif method == "grubbs":
                outlier_mask = await self._grubbs_outliers(flux, sigma)
            elif method == "ensemble":
                outlier_mask = await self._ensemble_outliers(lightcurve, sigma)
            else:
                # Default to sigma clipping
                outlier_mask = await self._sigma_clip_outliers(flux, sigma, maxiters)
            
            # Create cleaned light curve
            good_points_mask = ~outlier_mask
            
            cleaned_lc = LightCurveData(
                target_name=lightcurve.target_name,
                time_bjd=lightcurve.time_bjd[good_points_mask],
                flux=lightcurve.flux[good_points_mask],
                flux_err=lightcurve.flux_err[good_points_mask],
                mission=lightcurve.mission,
                instrument=lightcurve.instrument,
                cadence_minutes=lightcurve.cadence_minutes,
                sectors_quarters=lightcurve.sectors_quarters,
                data_quality_flags=lightcurve.data_quality_flags[good_points_mask] if lightcurve.data_quality_flags is not None else None,
                detrended=lightcurve.detrended,
                normalized=lightcurve.normalized,
                outliers_removed=True,
                source=lightcurve.source,
                download_date=lightcurve.download_date
            )
            
            points_after = len(cleaned_lc.flux)
            outliers_removed = points_before - points_after
            
            # Generate report
            report = {
                "method": method,
                "points_before": points_before,
                "points_after": points_after,
                "outliers_removed": outliers_removed,
                "outlier_percentage": (outliers_removed / points_before * 100) if points_before > 0 else 0,
                "parameters": {
                    "sigma": sigma,
                    "maxiters": maxiters
                },
                "outlier_statistics": await self._calculate_outlier_stats(
                    lightcurve.flux, outlier_mask
                ),
                "status": "success"
            }
            
            logger.info(f"Outlier removal completed: {outliers_removed} outliers removed ({report['outlier_percentage']:.1f}%)")
            
            return cleaned_lc, report
            
        except Exception as e:
            logger.error(f"Outlier removal failed: {e}")
            return lightcurve, {
                "status": "error",
                "error_message": str(e),
                "method": method
            }

    async def _sigma_clip_outliers(
        self, 
        flux: np.ndarray, 
        sigma: float, 
        maxiters: int
    ) -> np.ndarray:
        """Detect outliers using sigma clipping"""
        try:
            # Use astropy's sigma_clip which is more robust
            clipped_flux = sigma_clip(flux, sigma=sigma, maxiters=maxiters, masked=True)
            return clipped_flux.mask
        except Exception:
            # Fallback to manual implementation
            mask = np.zeros(len(flux), dtype=bool)
            working_flux = flux.copy()
            
            for _ in range(maxiters):
                mean_flux = np.mean(working_flux[~mask])
                std_flux = np.std(working_flux[~mask])
                
                new_outliers = np.abs(flux - mean_flux) > sigma * std_flux
                if np.array_equal(mask, new_outliers):
                    break
                mask = new_outliers
            
            return mask

    async def _mad_outliers(self, flux: np.ndarray, sigma: float) -> np.ndarray:
        """Detect outliers using Median Absolute Deviation"""
        median_flux = np.median(flux)
        mad = np.median(np.abs(flux - median_flux))
        
        # Convert sigma to MAD threshold (approximately)
        mad_threshold = sigma * 1.4826 * mad
        
        outlier_mask = np.abs(flux - median_flux) > mad_threshold
        return outlier_mask

    async def _iqr_outliers(self, flux: np.ndarray) -> np.ndarray:
        """Detect outliers using Interquartile Range"""
        q1 = np.percentile(flux, 25)
        q3 = np.percentile(flux, 75)
        iqr = q3 - q1
        
        # Standard IQR outlier detection
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (flux < lower_bound) | (flux > upper_bound)
        return outlier_mask

    async def _isolation_forest_outliers(self, lightcurve: LightCurveData) -> np.ndarray:
        """Detect outliers using Isolation Forest ML algorithm"""
        try:
            # Prepare features: flux, flux_err, and time derivatives
            flux = lightcurve.flux
            flux_err = lightcurve.flux_err
            time = lightcurve.time_bjd
            
            # Calculate derivatives
            flux_diff = np.gradient(flux)
            flux_diff2 = np.gradient(flux_diff)
            
            # Create feature matrix
            features = np.column_stack([
                flux,
                flux_err,
                flux_diff,
                flux_diff2
            ])
            
            # Handle any NaN or infinite values
            finite_mask = np.isfinite(features).all(axis=1)
            if not np.any(finite_mask):
                return np.zeros(len(flux), dtype=bool)
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features[finite_mask])
            
            # Apply Isolation Forest
            iso_forest = IsolationForest(
                contamination=0.1,  # Expect ~10% outliers
                random_state=42,
                n_estimators=100
            )
            
            outlier_labels = iso_forest.fit_predict(features_scaled)
            
            # Convert to boolean mask
            outlier_mask = np.zeros(len(flux), dtype=bool)
            outlier_mask[finite_mask] = (outlier_labels == -1)
            
            return outlier_mask
            
        except Exception as e:
            logger.warning(f"Isolation Forest failed: {e}, falling back to sigma clipping")
            return await self._sigma_clip_outliers(flux, 5.0, 3)

    async def _z_score_outliers(self, flux: np.ndarray, sigma: float) -> np.ndarray:
        """Detect outliers using Z-score"""
        mean_flux = np.mean(flux)
        std_flux = np.std(flux)
        
        if std_flux == 0:
            return np.zeros(len(flux), dtype=bool)
        
        z_scores = np.abs((flux - mean_flux) / std_flux)
        outlier_mask = z_scores > sigma
        
        return outlier_mask

    async def _modified_z_score_outliers(self, flux: np.ndarray, sigma: float) -> np.ndarray:
        """Detect outliers using Modified Z-score with MAD"""
        median_flux = np.median(flux)
        mad = np.median(np.abs(flux - median_flux))
        
        if mad == 0:
            return np.zeros(len(flux), dtype=bool)
        
        modified_z_scores = 0.6745 * (flux - median_flux) / mad
        outlier_mask = np.abs(modified_z_scores) > sigma
        
        return outlier_mask

    async def _grubbs_outliers(self, flux: np.ndarray, sigma: float) -> np.ndarray:
        """Detect outliers using Grubbs test"""
        try:
            n = len(flux)
            if n < 3:
                return np.zeros(n, dtype=bool)
            
            # Calculate Grubbs statistic for each point
            mean_flux = np.mean(flux)
            std_flux = np.std(flux, ddof=1)
            
            if std_flux == 0:
                return np.zeros(n, dtype=bool)
            
            grubbs_stats = np.abs((flux - mean_flux) / std_flux)
            
            # Critical value for Grubbs test (approximation)
            # For exact values, would need t-distribution
            alpha = 0.05  # 95% confidence
            t_critical = stats.t.ppf(1 - alpha / (2 * n), n - 2)
            grubbs_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_critical**2 / (n - 2 + t_critical**2))
            
            # Convert sigma to approximate Grubbs threshold
            grubbs_threshold = sigma / 2.0  # Rough conversion
            
            outlier_mask = grubbs_stats > max(grubbs_critical, grubbs_threshold)
            
            return outlier_mask
            
        except Exception:
            # Fallback to modified z-score
            return await self._modified_z_score_outliers(flux, sigma)

    async def _ensemble_outliers(self, lightcurve: LightCurveData, sigma: float) -> np.ndarray:
        """Detect outliers using ensemble of methods"""
        flux = lightcurve.flux
        
        # Apply multiple methods
        methods_results = []
        
        # Statistical methods
        methods_results.append(await self._sigma_clip_outliers(flux, sigma, 3))
        methods_results.append(await self._mad_outliers(flux, sigma))
        methods_results.append(await self._modified_z_score_outliers(flux, sigma))
        methods_results.append(await self._iqr_outliers(flux))
        
        # ML method (if it works)
        try:
            ml_result = await self._isolation_forest_outliers(lightcurve)
            methods_results.append(ml_result)
        except Exception:
            pass
        
        # Combine results using voting
        if len(methods_results) == 0:
            return np.zeros(len(flux), dtype=bool)
        
        # Stack all results
        all_results = np.stack(methods_results, axis=0)
        
        # Majority voting: point is outlier if majority of methods agree
        vote_threshold = len(methods_results) // 2 + 1
        ensemble_mask = np.sum(all_results, axis=0) >= vote_threshold
        
        return ensemble_mask

    async def _calculate_outlier_stats(
        self, 
        original_flux: np.ndarray, 
        outlier_mask: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate statistics about detected outliers"""
        if not np.any(outlier_mask):
            return {
                "outlier_count": 0,
                "outlier_flux_range": [0, 0],
                "outlier_severity": 0.0
            }
        
        outlier_flux = original_flux[outlier_mask]
        normal_flux = original_flux[~outlier_mask]
        
        if len(normal_flux) == 0:
            return {
                "outlier_count": len(outlier_flux),
                "outlier_flux_range": [float(np.min(outlier_flux)), float(np.max(outlier_flux))],
                "outlier_severity": 0.0
            }
        
        # Calculate severity as distance from normal distribution
        normal_median = np.median(normal_flux)
        normal_mad = np.median(np.abs(normal_flux - normal_median))
        
        if normal_mad > 0:
            outlier_distances = np.abs(outlier_flux - normal_median) / normal_mad
            avg_severity = np.mean(outlier_distances)
        else:
            avg_severity = 0.0
        
        return {
            "outlier_count": int(np.sum(outlier_mask)),
            "outlier_flux_range": [float(np.min(outlier_flux)), float(np.max(outlier_flux))],
            "outlier_severity": float(avg_severity),
            "normal_flux_stats": {
                "median": float(normal_median),
                "mad": float(normal_mad),
                "std": float(np.std(normal_flux))
            }
        }

    async def analyze_outliers(
        self,
        lightcurve: LightCurveData,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze outliers using multiple methods without removing them
        
        Args:
            lightcurve: Light curve data to analyze
            methods: List of methods to use (default: all methods)
            
        Returns:
            Outlier analysis report
        """
        if methods is None:
            methods = ["sigma_clip", "mad", "iqr", "modified_z_score"]
        
        logger.info(f"Analyzing outliers using methods: {methods}")
        
        flux = lightcurve.flux
        analysis = {
            "total_points": len(flux),
            "method_results": {},
            "consensus": {},
            "recommendations": []
        }
        
        outlier_masks = {}
        
        # Apply each method
        for method in methods:
            if method in self.detection_methods:
                try:
                    if method == "sigma_clip":
                        mask = await self._sigma_clip_outliers(flux, 5.0, 3)
                    elif method == "mad":
                        mask = await self._mad_outliers(flux, 5.0)
                    elif method == "iqr":
                        mask = await self._iqr_outliers(flux)
                    elif method == "isolation_forest":
                        mask = await self._isolation_forest_outliers(lightcurve)
                    elif method == "z_score":
                        mask = await self._z_score_outliers(flux, 5.0)
                    elif method == "modified_z_score":
                        mask = await self._modified_z_score_outliers(flux, 5.0)
                    elif method == "grubbs":
                        mask = await self._grubbs_outliers(flux, 5.0)
                    else:
                        continue
                    
                    outlier_masks[method] = mask
                    outlier_count = np.sum(mask)
                    
                    analysis["method_results"][method] = {
                        "outliers_detected": int(outlier_count),
                        "outlier_percentage": float(outlier_count / len(flux) * 100),
                        "outlier_indices": np.where(mask)[0].tolist()
                    }
                    
                except Exception as e:
                    analysis["method_results"][method] = {
                        "status": "failed",
                        "error": str(e)
                    }
        
        # Calculate consensus
        if outlier_masks:
            # Find points identified as outliers by multiple methods
            all_masks = np.stack(list(outlier_masks.values()), axis=0)
            consensus_counts = np.sum(all_masks, axis=0)
            
            # Points identified by majority of methods
            majority_threshold = len(outlier_masks) // 2 + 1
            consensus_outliers = consensus_counts >= majority_threshold
            
            # Points identified by all methods
            unanimous_outliers = consensus_counts == len(outlier_masks)
            
            analysis["consensus"] = {
                "majority_outliers": int(np.sum(consensus_outliers)),
                "unanimous_outliers": int(np.sum(unanimous_outliers)),
                "consensus_percentage": float(np.sum(consensus_outliers) / len(flux) * 100),
                "agreement_scores": consensus_counts.tolist()
            }
            
            # Generate recommendations
            consensus_pct = analysis["consensus"]["consensus_percentage"]
            
            if consensus_pct > 10:
                analysis["recommendations"].append("High outlier rate - consider data quality issues")
            elif consensus_pct > 5:
                analysis["recommendations"].append("Moderate outlier rate - outlier removal recommended")
            elif consensus_pct > 1:
                analysis["recommendations"].append("Low outlier rate - conservative removal suggested")
            else:
                analysis["recommendations"].append("Very few outliers detected - minimal cleaning needed")
            
            # Recommend best method
            method_scores = {}
            for method, result in analysis["method_results"].items():
                if "outliers_detected" in result:
                    # Score based on consensus agreement
                    method_mask = outlier_masks[method]
                    agreement = np.sum(method_mask & consensus_outliers) / max(1, np.sum(method_mask))
                    method_scores[method] = agreement
            
            if method_scores:
                best_method = max(method_scores, key=method_scores.get)
                analysis["recommendations"].append(f"Best performing method: {best_method}")
        
        return analysis

    def get_outlier_detection_info(self) -> Dict[str, Any]:
        """Get information about available outlier detection methods"""
        return {
            "available_methods": self.detection_methods,
            "method_characteristics": {
                "sigma_clip": {
                    "type": "statistical",
                    "robust": True,
                    "iterative": True,
                    "parameters": ["sigma", "maxiters"]
                },
                "mad": {
                    "type": "statistical",
                    "robust": True,
                    "iterative": False,
                    "parameters": ["sigma"]
                },
                "iqr": {
                    "type": "statistical",
                    "robust": True,
                    "iterative": False,
                    "parameters": []
                },
                "isolation_forest": {
                    "type": "machine_learning",
                    "robust": True,
                    "iterative": False,
                    "parameters": ["contamination"]
                },
                "z_score": {
                    "type": "statistical",
                    "robust": False,
                    "iterative": False,
                    "parameters": ["sigma"]
                },
                "modified_z_score": {
                    "type": "statistical",
                    "robust": True,
                    "iterative": False,
                    "parameters": ["sigma"]
                },
                "grubbs": {
                    "type": "statistical",
                    "robust": False,
                    "iterative": False,
                    "parameters": ["sigma"]
                },
                "ensemble": {
                    "type": "ensemble",
                    "robust": True,
                    "iterative": False,
                    "parameters": ["sigma"]
                }
            },
            "recommendations": {
                "general_purpose": "sigma_clip",
                "robust": "mad",
                "conservative": "iqr",
                "aggressive": "ensemble",
                "machine_learning": "isolation_forest"
            }
        }
