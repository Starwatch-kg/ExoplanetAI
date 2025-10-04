"""
Quality filtering system for light curve data
Система фильтрации качества для данных кривых блеска
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data_sources.base import LightCurveData

logger = logging.getLogger(__name__)


class QualityFilter:
    """
    Quality filtering system for removing bad data points
    based on instrument-specific quality flags
    """

    def __init__(self):
        # TESS quality flag definitions
        self.tess_quality_flags = {
            1: "Attitude tweak",
            2: "Safe mode",
            4: "Coarse point",
            8: "Earth point",
            16: "Argabrightening",
            32: "Reaction wheel desaturation",
            64: "Manual exclude",
            128: "Discontinuity",
            256: "Impulsive outlier",
            512: "Argabrightening",
            1024: "Cosmic ray",
            2048: "Straylight",
            4096: "Straylight",
            8192: "Straylight",
            16384: "Collateral cosmic ray"
        }
        
        # Kepler quality flag definitions
        self.kepler_quality_flags = {
            1: "Attitude tweak",
            2: "Safe mode",
            4: "Coarse point",
            8: "Earth point",
            16: "Zero crossing",
            32: "Desaturation event",
            64: "Argabrightening",
            128: "Cosmic ray",
            256: "Manual exclude",
            512: "Sudden pixel sensitivity dropout",
            1024: "Impulsive outlier",
            2048: "Argabrightening",
            4096: "Cosmic ray",
            8192: "Detector anomaly",
            16384: "No fine point",
            32768: "No data",
            65536: "Rolling band",
            131072: "Rolling band",
            262144: "Possible thruster firing",
            524288: "Thruster firing"
        }
        
        # Quality bitmask presets
        self.quality_presets = {
            "none": 0,  # No filtering
            "default": self._get_default_bitmask(),
            "hard": self._get_hard_bitmask(),
            "hardest": self._get_hardest_bitmask()
        }

    def _get_default_bitmask(self) -> int:
        """Default quality bitmask (moderate filtering)"""
        # Remove major issues but keep most data
        return (2 + 4 + 8 + 32 + 64 + 128 + 256 + 1024 + 2048 + 4096 + 8192)

    def _get_hard_bitmask(self) -> int:
        """Hard quality bitmask (aggressive filtering)"""
        # Remove all known issues
        return sum(self.tess_quality_flags.keys())

    def _get_hardest_bitmask(self) -> int:
        """Hardest quality bitmask (very aggressive filtering)"""
        # Remove any non-zero quality flag
        return -1  # All bits set

    async def filter_by_quality(
        self,
        lightcurve: LightCurveData,
        quality_bitmask: str = "default"
    ) -> Tuple[LightCurveData, Dict[str, Any]]:
        """
        Filter light curve data based on quality flags
        
        Args:
            lightcurve: Input light curve data
            quality_bitmask: Quality filtering level ("none", "default", "hard", "hardest")
            
        Returns:
            Tuple of (filtered_lightcurve, filtering_report)
        """
        logger.info(f"Applying quality filtering with bitmask: {quality_bitmask}")
        
        if lightcurve.data_quality_flags is None:
            logger.warning("No quality flags available, skipping quality filtering")
            return lightcurve, {
                "points_before": len(lightcurve.time_bjd),
                "points_after": len(lightcurve.time_bjd),
                "removed_points": 0,
                "bad_quality_flags": [],
                "filtering_applied": False
            }
        
        # Get bitmask value
        if isinstance(quality_bitmask, str):
            if quality_bitmask in self.quality_presets:
                bitmask = self.quality_presets[quality_bitmask]
            else:
                logger.warning(f"Unknown quality preset '{quality_bitmask}', using 'default'")
                bitmask = self.quality_presets["default"]
        else:
            bitmask = int(quality_bitmask)
        
        points_before = len(lightcurve.time_bjd)
        
        if bitmask == 0:
            # No filtering requested
            return lightcurve, {
                "points_before": points_before,
                "points_after": points_before,
                "removed_points": 0,
                "bad_quality_flags": [],
                "filtering_applied": False
            }
        
        # Apply quality filtering
        if bitmask == -1:
            # Remove all non-zero quality flags
            good_quality_mask = lightcurve.data_quality_flags == 0
        else:
            # Remove points matching any bit in the bitmask
            good_quality_mask = (lightcurve.data_quality_flags & bitmask) == 0
        
        # Identify bad quality flags
        bad_flags = lightcurve.data_quality_flags[~good_quality_mask]
        unique_bad_flags = np.unique(bad_flags)
        
        # Create filtered light curve
        filtered_lc = LightCurveData(
            target_name=lightcurve.target_name,
            time_bjd=lightcurve.time_bjd[good_quality_mask],
            flux=lightcurve.flux[good_quality_mask],
            flux_err=lightcurve.flux_err[good_quality_mask],
            mission=lightcurve.mission,
            instrument=lightcurve.instrument,
            cadence_minutes=lightcurve.cadence_minutes,
            sectors_quarters=lightcurve.sectors_quarters,
            data_quality_flags=lightcurve.data_quality_flags[good_quality_mask],
            detrended=lightcurve.detrended,
            normalized=lightcurve.normalized,
            outliers_removed=lightcurve.outliers_removed,
            source=lightcurve.source,
            download_date=lightcurve.download_date
        )
        
        points_after = len(filtered_lc.time_bjd)
        removed_points = points_before - points_after
        
        # Create report
        report = {
            "points_before": points_before,
            "points_after": points_after,
            "removed_points": removed_points,
            "removal_percentage": (removed_points / points_before * 100) if points_before > 0 else 0,
            "bad_quality_flags": unique_bad_flags.tolist(),
            "bitmask_used": bitmask,
            "filtering_applied": True
        }
        
        # Add flag descriptions if available
        flag_descriptions = []
        mission_flags = self.tess_quality_flags if lightcurve.mission.upper() == "TESS" else self.kepler_quality_flags
        
        for flag in unique_bad_flags:
            if flag in mission_flags:
                flag_descriptions.append({
                    "flag": int(flag),
                    "description": mission_flags[flag]
                })
        
        report["flag_descriptions"] = flag_descriptions
        
        logger.info(f"Quality filtering completed: {removed_points} points removed ({report['removal_percentage']:.1f}%)")
        
        return filtered_lc, report

    async def analyze_quality_flags(
        self,
        lightcurve: LightCurveData
    ) -> Dict[str, Any]:
        """
        Analyze quality flags in light curve data
        
        Args:
            lightcurve: Light curve data to analyze
            
        Returns:
            Quality flag analysis report
        """
        if lightcurve.data_quality_flags is None:
            return {
                "has_quality_flags": False,
                "message": "No quality flags available"
            }
        
        quality_flags = lightcurve.data_quality_flags
        unique_flags, flag_counts = np.unique(quality_flags, return_counts=True)
        
        # Get mission-specific flag definitions
        mission_flags = self.tess_quality_flags if lightcurve.mission.upper() == "TESS" else self.kepler_quality_flags
        
        # Analyze flag distribution
        flag_analysis = []
        for flag, count in zip(unique_flags, flag_counts):
            percentage = (count / len(quality_flags)) * 100
            
            analysis_entry = {
                "flag": int(flag),
                "count": int(count),
                "percentage": float(percentage),
                "description": mission_flags.get(flag, "Unknown flag")
            }
            
            # Analyze individual bits for composite flags
            if flag > 0:
                active_bits = []
                for bit_value, description in mission_flags.items():
                    if flag & bit_value:
                        active_bits.append({
                            "bit": bit_value,
                            "description": description
                        })
                analysis_entry["active_bits"] = active_bits
            
            flag_analysis.append(analysis_entry)
        
        # Calculate quality statistics
        good_quality_points = np.sum(quality_flags == 0)
        bad_quality_points = len(quality_flags) - good_quality_points
        
        report = {
            "has_quality_flags": True,
            "mission": lightcurve.mission,
            "total_points": len(quality_flags),
            "good_quality_points": int(good_quality_points),
            "bad_quality_points": int(bad_quality_points),
            "good_quality_percentage": float(good_quality_points / len(quality_flags) * 100),
            "unique_flags": len(unique_flags),
            "flag_analysis": flag_analysis,
            "most_common_issues": []
        }
        
        # Identify most common issues (excluding good quality)
        bad_flags = [(flag, count, percentage) for flag, count, percentage in 
                    zip(unique_flags, flag_counts, flag_counts/len(quality_flags)*100) 
                    if flag != 0]
        
        bad_flags.sort(key=lambda x: x[1], reverse=True)  # Sort by count
        
        for flag, count, percentage in bad_flags[:5]:  # Top 5 issues
            report["most_common_issues"].append({
                "flag": int(flag),
                "count": int(count),
                "percentage": float(percentage),
                "description": mission_flags.get(flag, "Unknown flag")
            })
        
        return report

    async def recommend_quality_filter(
        self,
        lightcurve: LightCurveData,
        target_retention: float = 0.8
    ) -> Dict[str, Any]:
        """
        Recommend optimal quality filtering based on data analysis
        
        Args:
            lightcurve: Light curve data to analyze
            target_retention: Target fraction of data to retain (0.0-1.0)
            
        Returns:
            Filtering recommendations
        """
        if lightcurve.data_quality_flags is None:
            return {
                "recommendation": "none",
                "reason": "No quality flags available"
            }
        
        # Analyze current quality flags
        analysis = await self.analyze_quality_flags(lightcurve)
        
        # Test different filtering levels
        filter_tests = {}
        for preset_name, bitmask in self.quality_presets.items():
            if preset_name == "none":
                continue
                
            if bitmask == -1:
                # Hardest filter - only keep zero flags
                retained_points = analysis["good_quality_points"]
            else:
                # Count points that would pass the filter
                good_mask = (lightcurve.data_quality_flags & bitmask) == 0
                retained_points = np.sum(good_mask)
            
            retention_rate = retained_points / analysis["total_points"]
            
            filter_tests[preset_name] = {
                "retained_points": retained_points,
                "retention_rate": retention_rate,
                "removed_points": analysis["total_points"] - retained_points
            }
        
        # Find best filter level
        best_filter = "none"
        best_score = float('inf')
        
        for preset_name, test_result in filter_tests.items():
            retention_rate = test_result["retention_rate"]
            
            # Score based on how close to target retention and data quality
            retention_penalty = abs(retention_rate - target_retention)
            quality_bonus = analysis["good_quality_percentage"] / 100.0
            
            score = retention_penalty - 0.1 * quality_bonus
            
            if score < best_score and retention_rate >= target_retention * 0.9:
                best_score = score
                best_filter = preset_name
        
        # Generate recommendation
        recommendation = {
            "recommended_filter": best_filter,
            "expected_retention": filter_tests.get(best_filter, {}).get("retention_rate", 1.0),
            "data_quality_summary": {
                "good_quality_percentage": analysis["good_quality_percentage"],
                "most_common_issues": analysis["most_common_issues"][:3]
            },
            "filter_comparison": filter_tests,
            "reasoning": self._generate_filtering_reasoning(
                analysis, filter_tests, best_filter, target_retention
            )
        }
        
        return recommendation

    def _generate_filtering_reasoning(
        self,
        analysis: Dict[str, Any],
        filter_tests: Dict[str, Any],
        recommended_filter: str,
        target_retention: float
    ) -> str:
        """Generate human-readable reasoning for filter recommendation"""
        
        good_quality_pct = analysis["good_quality_percentage"]
        
        if good_quality_pct > 95:
            quality_desc = "excellent"
        elif good_quality_pct > 85:
            quality_desc = "good"
        elif good_quality_pct > 70:
            quality_desc = "moderate"
        else:
            quality_desc = "poor"
        
        reasoning = f"Data quality is {quality_desc} ({good_quality_pct:.1f}% good quality points). "
        
        if recommended_filter == "none":
            reasoning += "No filtering recommended as data quality is already high."
        else:
            expected_retention = filter_tests[recommended_filter]["retention_rate"] * 100
            reasoning += f"Recommended '{recommended_filter}' filtering will retain {expected_retention:.1f}% of data "
            reasoning += f"while removing problematic points. "
            
            if analysis["most_common_issues"]:
                top_issue = analysis["most_common_issues"][0]
                reasoning += f"Main quality issue: {top_issue['description']} "
                reasoning += f"({top_issue['percentage']:.1f}% of data)."
        
        return reasoning

    def get_quality_flag_info(self, mission: str) -> Dict[int, str]:
        """
        Get quality flag definitions for a specific mission
        
        Args:
            mission: Mission name (TESS, Kepler, K2)
            
        Returns:
            Dictionary mapping flag values to descriptions
        """
        mission_upper = mission.upper()
        
        if mission_upper == "TESS":
            return self.tess_quality_flags.copy()
        elif mission_upper in ["KEPLER", "K2"]:
            return self.kepler_quality_flags.copy()
        else:
            logger.warning(f"Unknown mission '{mission}', returning TESS flags")
            return self.tess_quality_flags.copy()

    def create_custom_bitmask(self, flags_to_remove: List[int]) -> int:
        """
        Create custom quality bitmask from list of flag values
        
        Args:
            flags_to_remove: List of quality flag values to filter out
            
        Returns:
            Bitmask value
        """
        bitmask = 0
        for flag in flags_to_remove:
            bitmask |= flag
        
        return bitmask
