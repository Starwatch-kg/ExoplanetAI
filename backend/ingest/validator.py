"""
Data validation system for ExoplanetAI
Система валидации данных для ExoplanetAI
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_sources.base import LightCurveData, PlanetInfo

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validation system for astronomical data
    """

    def __init__(self):
        # Validation thresholds and ranges
        self.valid_ranges = {
            "period_days": (0.1, 10000),  # Orbital period in days
            "radius_earth": (0.1, 100),   # Planet radius in Earth radii
            "radius_jupiter": (0.01, 10), # Planet radius in Jupiter radii
            "mass_earth": (0.01, 10000),  # Planet mass in Earth masses
            "mass_jupiter": (0.001, 100), # Planet mass in Jupiter masses
            "temperature_k": (50, 5000),  # Temperature in Kelvin
            "ra_deg": (0, 360),           # Right ascension in degrees
            "dec_deg": (-90, 90),         # Declination in degrees
            "distance_pc": (0.1, 10000),  # Distance in parsecs
            "stellar_mass": (0.1, 10),    # Stellar mass in solar masses
            "stellar_radius": (0.1, 100), # Stellar radius in solar radii
            "transit_depth_ppm": (1, 100000), # Transit depth in ppm
            "transit_duration_hours": (0.1, 48), # Transit duration in hours
            "magnitude": (5, 25),         # Stellar magnitude
            "flux_relative": (0.5, 1.5),  # Relative flux range
            "flux_error_max": 0.1         # Maximum flux error (10%)
        }
        
        # Required columns for different table types
        self.required_columns = {
            "koi": ["kepid", "koi_period", "koi_disposition"],
            "toi": ["tid", "toi_period", "toi_disposition"], 
            "k2": ["epic_name", "k2c_period", "k2c_disposition"]
        }
        
        # Valid dispositions
        self.valid_dispositions = {
            "koi": ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"],
            "toi": ["CP", "PC", "FP", "FA", "APC", "KP"],
            "k2": ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]
        }

    async def validate_koi_table(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate KOI (Kepler Objects of Interest) table
        
        Args:
            df: KOI DataFrame to validate
            
        Returns:
            Dict with validation results
        """
        logger.info(f"Validating KOI table with {len(df)} records")
        
        errors = []
        warnings = []
        stats = {}
        
        # Check required columns
        missing_cols = [col for col in self.required_columns["koi"] if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data types and ranges
        if "kepid" in df.columns:
            invalid_kepid = df[~df["kepid"].astype(str).str.match(r'^\d+$')].index
            if len(invalid_kepid) > 0:
                warnings.append(f"Invalid KEPID format in {len(invalid_kepid)} records")
        
        # Validate orbital periods
        if "koi_period" in df.columns:
            period_col = df["koi_period"]
            invalid_periods = period_col[
                (period_col <= self.valid_ranges["period_days"][0]) |
                (period_col >= self.valid_ranges["period_days"][1]) |
                (period_col.isna())
            ]
            if len(invalid_periods) > 0:
                warnings.append(f"Invalid orbital periods in {len(invalid_periods)} records")
            
            stats["period_range"] = {
                "min": float(period_col.min()) if not period_col.empty else None,
                "max": float(period_col.max()) if not period_col.empty else None,
                "median": float(period_col.median()) if not period_col.empty else None
            }
        
        # Validate planet radii
        if "koi_prad" in df.columns:
            radius_col = df["koi_prad"]
            invalid_radii = radius_col[
                (radius_col <= self.valid_ranges["radius_earth"][0]) |
                (radius_col >= self.valid_ranges["radius_earth"][1])
            ]
            if len(invalid_radii) > 0:
                warnings.append(f"Invalid planet radii in {len(invalid_radii)} records")
        
        # Validate coordinates
        coord_errors = await self._validate_coordinates(df, "ra", "dec")
        if coord_errors:
            warnings.extend(coord_errors)
        
        # Validate dispositions
        if "koi_disposition" in df.columns:
            invalid_disp = df[~df["koi_disposition"].isin(self.valid_dispositions["koi"])]
            if len(invalid_disp) > 0:
                warnings.append(f"Invalid dispositions in {len(invalid_disp)} records")
            
            stats["dispositions"] = df["koi_disposition"].value_counts().to_dict()
        
        # Check for duplicates
        if "kepid" in df.columns and "koi_name" in df.columns:
            duplicates = df.duplicated(subset=["kepid", "koi_name"])
            if duplicates.any():
                warnings.append(f"Found {duplicates.sum()} duplicate records")
        
        # Data quality statistics
        stats.update({
            "total_records": len(df),
            "null_values": df.isnull().sum().to_dict(),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        })
        
        result = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "stats": stats,
            "table_type": "koi",
            "validation_timestamp": pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"KOI validation completed: {len(errors)} errors, {len(warnings)} warnings")
        return result

    async def validate_toi_table(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate TOI (TESS Objects of Interest) table
        
        Args:
            df: TOI DataFrame to validate
            
        Returns:
            Dict with validation results
        """
        logger.info(f"Validating TOI table with {len(df)} records")
        
        errors = []
        warnings = []
        stats = {}
        
        # Check required columns (TOI tables have different column names)
        toi_required = ["tid", "toi", "pl_name"] if "tid" in df.columns else ["toi", "tic_id"]
        missing_cols = [col for col in toi_required if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Validate TIC IDs
        if "tid" in df.columns:
            invalid_tic = df[~df["tid"].astype(str).str.match(r'^\d+$')].index
            if len(invalid_tic) > 0:
                warnings.append(f"Invalid TIC ID format in {len(invalid_tic)} records")
        
        # Validate periods
        period_cols = [col for col in df.columns if "period" in col.lower()]
        if period_cols:
            period_col = df[period_cols[0]]
            invalid_periods = period_col[
                (period_col <= self.valid_ranges["period_days"][0]) |
                (period_col >= self.valid_ranges["period_days"][1]) |
                (period_col.isna())
            ]
            if len(invalid_periods) > 0:
                warnings.append(f"Invalid orbital periods in {len(invalid_periods)} records")
        
        # Validate coordinates
        coord_errors = await self._validate_coordinates(df, "ra", "dec")
        if coord_errors:
            warnings.extend(coord_errors)
        
        # Validate dispositions
        disp_cols = [col for col in df.columns if "disposition" in col.lower()]
        if disp_cols:
            disp_col = df[disp_cols[0]]
            invalid_disp = disp_col[~disp_col.isin(self.valid_dispositions["toi"])]
            if len(invalid_disp) > 0:
                warnings.append(f"Invalid dispositions in {len(invalid_disp)} records")
            
            stats["dispositions"] = disp_col.value_counts().to_dict()
        
        # Check for duplicates
        if "toi" in df.columns:
            duplicates = df.duplicated(subset=["toi"])
            if duplicates.any():
                warnings.append(f"Found {duplicates.sum()} duplicate TOI records")
        
        stats.update({
            "total_records": len(df),
            "null_values": df.isnull().sum().to_dict(),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        })
        
        result = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "stats": stats,
            "table_type": "toi",
            "validation_timestamp": pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"TOI validation completed: {len(errors)} errors, {len(warnings)} warnings")
        return result

    async def validate_k2_table(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate K2 candidates table
        
        Args:
            df: K2 DataFrame to validate
            
        Returns:
            Dict with validation results
        """
        logger.info(f"Validating K2 table with {len(df)} records")
        
        errors = []
        warnings = []
        stats = {}
        
        # Check required columns
        k2_required = ["epic_name"] if "epic_name" in df.columns else ["epic_id", "k2_name"]
        missing_cols = [col for col in k2_required if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Validate EPIC IDs
        epic_cols = [col for col in df.columns if "epic" in col.lower()]
        if epic_cols:
            epic_col = df[epic_cols[0]]
            invalid_epic = epic_col[~epic_col.astype(str).str.match(r'^\d+$')].index
            if len(invalid_epic) > 0:
                warnings.append(f"Invalid EPIC ID format in {len(invalid_epic)} records")
        
        # Validate periods
        period_cols = [col for col in df.columns if "period" in col.lower()]
        if period_cols:
            period_col = df[period_cols[0]]
            invalid_periods = period_col[
                (period_col <= self.valid_ranges["period_days"][0]) |
                (period_col >= self.valid_ranges["period_days"][1]) |
                (period_col.isna())
            ]
            if len(invalid_periods) > 0:
                warnings.append(f"Invalid orbital periods in {len(invalid_periods)} records")
        
        # Validate coordinates
        coord_errors = await self._validate_coordinates(df, "ra", "dec")
        if coord_errors:
            warnings.extend(coord_errors)
        
        stats.update({
            "total_records": len(df),
            "null_values": df.isnull().sum().to_dict(),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        })
        
        result = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "stats": stats,
            "table_type": "k2",
            "validation_timestamp": pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"K2 validation completed: {len(errors)} errors, {len(warnings)} warnings")
        return result

    async def validate_lightcurve(self, lightcurve: LightCurveData) -> Dict[str, Any]:
        """
        Validate light curve data
        
        Args:
            lightcurve: LightCurveData object to validate
            
        Returns:
            Dict with validation results
        """
        logger.info(f"Validating light curve for {lightcurve.target_name}")
        
        errors = []
        warnings = []
        stats = {}
        
        # Check data consistency
        if len(lightcurve.time_bjd) != len(lightcurve.flux):
            errors.append("Time and flux arrays have different lengths")
        
        if len(lightcurve.flux) != len(lightcurve.flux_err):
            errors.append("Flux and flux_err arrays have different lengths")
        
        # Check for valid time values
        if np.any(lightcurve.time_bjd <= 0):
            errors.append("Invalid time values (BJD <= 0)")
        
        # Check time ordering
        if not np.all(np.diff(lightcurve.time_bjd) >= 0):
            warnings.append("Time series is not monotonically increasing")
        
        # Check flux values
        flux_range = self.valid_ranges["flux_relative"]
        invalid_flux = np.sum(
            (lightcurve.flux < flux_range[0]) | 
            (lightcurve.flux > flux_range[1]) |
            np.isnan(lightcurve.flux)
        )
        if invalid_flux > 0:
            warnings.append(f"Invalid flux values in {invalid_flux} data points")
        
        # Check flux errors
        max_flux_err = self.valid_ranges["flux_error_max"]
        high_error_points = np.sum(lightcurve.flux_err > max_flux_err)
        if high_error_points > len(lightcurve.flux) * 0.1:  # More than 10% high error
            warnings.append(f"High flux errors in {high_error_points} data points (>{max_flux_err*100}%)")
        
        # Check for gaps in time series
        time_diffs = np.diff(lightcurve.time_bjd)
        median_cadence = np.median(time_diffs)
        large_gaps = np.sum(time_diffs > 5 * median_cadence)
        if large_gaps > 0:
            warnings.append(f"Found {large_gaps} large gaps in time series")
        
        # Data quality statistics
        stats.update({
            "data_points": len(lightcurve.time_bjd),
            "time_span_days": float(lightcurve.time_bjd.max() - lightcurve.time_bjd.min()),
            "median_cadence_days": float(median_cadence),
            "flux_stats": {
                "mean": float(np.mean(lightcurve.flux)),
                "std": float(np.std(lightcurve.flux)),
                "min": float(np.min(lightcurve.flux)),
                "max": float(np.max(lightcurve.flux))
            },
            "flux_error_stats": {
                "mean": float(np.mean(lightcurve.flux_err)),
                "median": float(np.median(lightcurve.flux_err)),
                "max": float(np.max(lightcurve.flux_err))
            },
            "nan_count": {
                "time": int(np.sum(np.isnan(lightcurve.time_bjd))),
                "flux": int(np.sum(np.isnan(lightcurve.flux))),
                "flux_err": int(np.sum(np.isnan(lightcurve.flux_err)))
            }
        })
        
        # Quality flags analysis
        if lightcurve.data_quality_flags is not None:
            unique_flags = np.unique(lightcurve.data_quality_flags)
            bad_quality_points = np.sum(lightcurve.data_quality_flags > 0)
            stats["quality_flags"] = {
                "unique_flags": unique_flags.tolist(),
                "bad_quality_points": int(bad_quality_points),
                "bad_quality_percentage": float(bad_quality_points / len(lightcurve.data_quality_flags) * 100)
            }
            
            if bad_quality_points > len(lightcurve.data_quality_flags) * 0.2:  # More than 20% bad
                warnings.append(f"High percentage of bad quality data points: {bad_quality_points/len(lightcurve.data_quality_flags)*100:.1f}%")
        
        result = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "stats": stats,
            "target_name": lightcurve.target_name,
            "mission": lightcurve.mission,
            "validation_timestamp": pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"Light curve validation completed: {len(errors)} errors, {len(warnings)} warnings")
        return result

    async def _validate_coordinates(
        self, 
        df: pd.DataFrame, 
        ra_col: str, 
        dec_col: str
    ) -> List[str]:
        """
        Validate celestial coordinates
        
        Args:
            df: DataFrame containing coordinates
            ra_col: Right ascension column name
            dec_col: Declination column name
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        if ra_col not in df.columns or dec_col not in df.columns:
            return warnings
        
        ra_values = df[ra_col]
        dec_values = df[dec_col]
        
        # Check RA range (0-360 degrees)
        invalid_ra = ra_values[
            (ra_values < 0) | (ra_values > 360) | ra_values.isna()
        ]
        if len(invalid_ra) > 0:
            warnings.append(f"Invalid RA values in {len(invalid_ra)} records")
        
        # Check Dec range (-90 to +90 degrees)
        invalid_dec = dec_values[
            (dec_values < -90) | (dec_values > 90) | dec_values.isna()
        ]
        if len(invalid_dec) > 0:
            warnings.append(f"Invalid Dec values in {len(invalid_dec)} records")
        
        # Check for reasonable coordinate precision
        try:
            # Create SkyCoord objects for valid coordinates
            valid_mask = (~ra_values.isna()) & (~dec_values.isna())
            if valid_mask.any():
                coords = SkyCoord(
                    ra=ra_values[valid_mask].values * u.degree,
                    dec=dec_values[valid_mask].values * u.degree
                )
                
                # Check for coordinates at (0,0) which might be invalid
                zero_coords = ((ra_values == 0) & (dec_values == 0)).sum()
                if zero_coords > 0:
                    warnings.append(f"Found {zero_coords} coordinates at (0,0) - possibly invalid")
                    
        except Exception as e:
            warnings.append(f"Error validating coordinates: {str(e)}")
        
        return warnings

    def validate_checksum(self, data: bytes, expected_checksum: str) -> bool:
        """
        Validate data integrity using checksum
        
        Args:
            data: Raw data bytes
            expected_checksum: Expected SHA256 checksum
            
        Returns:
            True if checksum matches
        """
        actual_checksum = hashlib.sha256(data).hexdigest()
        return actual_checksum == expected_checksum

    def detect_duplicates(
        self, 
        df: pd.DataFrame, 
        key_columns: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect and separate duplicate records
        
        Args:
            df: DataFrame to check
            key_columns: Columns to use for duplicate detection
            
        Returns:
            Tuple of (unique_records, duplicate_records)
        """
        # Find duplicates
        duplicates_mask = df.duplicated(subset=key_columns, keep=False)
        
        unique_records = df[~duplicates_mask].copy()
        duplicate_records = df[duplicates_mask].copy()
        
        logger.info(f"Found {len(duplicate_records)} duplicate records out of {len(df)} total")
        
        return unique_records, duplicate_records

    async def generate_validation_report(
        self, 
        validation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report
        
        Args:
            validation_results: List of validation results from different tables
            
        Returns:
            Comprehensive validation report
        """
        report = {
            "report_timestamp": pd.Timestamp.now().isoformat(),
            "total_validations": len(validation_results),
            "summary": {
                "passed": 0,
                "failed": 0,
                "total_errors": 0,
                "total_warnings": 0
            },
            "by_table_type": {},
            "common_issues": [],
            "recommendations": []
        }
        
        # Analyze results
        all_errors = []
        all_warnings = []
        
        for result in validation_results:
            table_type = result.get("table_type", "unknown")
            
            if result["valid"]:
                report["summary"]["passed"] += 1
            else:
                report["summary"]["failed"] += 1
            
            report["summary"]["total_errors"] += len(result["errors"])
            report["summary"]["total_warnings"] += len(result["warnings"])
            
            all_errors.extend(result["errors"])
            all_warnings.extend(result["warnings"])
            
            # Per table type summary
            if table_type not in report["by_table_type"]:
                report["by_table_type"][table_type] = {
                    "count": 0,
                    "errors": 0,
                    "warnings": 0,
                    "records": 0
                }
            
            report["by_table_type"][table_type]["count"] += 1
            report["by_table_type"][table_type]["errors"] += len(result["errors"])
            report["by_table_type"][table_type]["warnings"] += len(result["warnings"])
            
            if "stats" in result and "total_records" in result["stats"]:
                report["by_table_type"][table_type]["records"] += result["stats"]["total_records"]
        
        # Identify common issues
        error_counts = {}
        warning_counts = {}
        
        for error in all_errors:
            error_counts[error] = error_counts.get(error, 0) + 1
        
        for warning in all_warnings:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
        
        # Most common issues
        report["common_issues"] = {
            "errors": sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "warnings": sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        # Generate recommendations
        if report["summary"]["total_errors"] > 0:
            report["recommendations"].append("Address critical errors before proceeding with data ingestion")
        
        if report["summary"]["total_warnings"] > 10:
            report["recommendations"].append("Review data quality warnings and consider data cleaning")
        
        if any("duplicate" in str(issue).lower() for issue in all_warnings):
            report["recommendations"].append("Implement duplicate detection and removal process")
        
        if any("coordinate" in str(issue).lower() for issue in all_warnings):
            report["recommendations"].append("Verify coordinate system and validate celestial coordinates")
        
        logger.info(f"Generated validation report: {report['summary']['passed']}/{len(validation_results)} validations passed")
        
        return report
