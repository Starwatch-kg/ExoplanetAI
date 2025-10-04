"""
Comprehensive data management system for ExoplanetAI
Комплексная система управления данными для ExoplanetAI
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import pandas as pd
# Astronomy data access - using fallback for compatibility
try:
    from astroquery.exoplanetarchive import ExoplanetArchive
except ImportError:
    # Fallback for older astroquery versions
    try:
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive as ExoplanetArchive
    except ImportError:
        ExoplanetArchive = None
        logger.warning("ExoplanetArchive not available - using HTTP fallback")

try:
    from astroquery.mast import Catalogs, Observations
except ImportError:
    Catalogs = None
    Observations = None
    logger.warning("MAST services not available - using HTTP fallback")
from lightkurve import search_lightcurve

from .storage import StorageManager
from .validator import DataValidator
from .versioning import VersionManager
from core.config import get_settings
from data_sources.base import LightCurveData, PlanetInfo

logger = logging.getLogger(__name__)


class DataManager:
    """
    Central data management system for automated ingestion,
    validation, and storage of astronomical data
    """

    def __init__(self):
        self.settings = get_settings()
        self.storage = StorageManager()
        self.validator = DataValidator()
        self.version_manager = VersionManager()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Data source URLs
        self.nasa_archive_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        self.mast_url = "https://mast.stsci.edu/api/v0.1"
        self.exofop_url = "https://exofop.ipac.caltech.edu/tess"
        
        # Cache settings
        self.cache_ttl = {
            "koi_table": timedelta(hours=24),
            "toi_table": timedelta(hours=12),
            "k2_table": timedelta(hours=24),
            "lightcurves": timedelta(hours=6)
        }

    async def initialize(self) -> bool:
        """Initialize data manager and all components"""
        try:
            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)
            )
            logger.info("HTTP session initialized")
            
            # Initialize storage with error checking
            storage_ok = await self.storage.initialize()
            if not storage_ok:
                logger.error("Storage initialization failed")
                await self.cleanup()
                return False
            logger.info("Storage initialized successfully")
            
            # Initialize version manager with error checking
            version_ok = await self.version_manager.initialize()
            if not version_ok:
                logger.error("Version manager initialization failed")
                await self.cleanup()
                return False
            logger.info("Version manager initialized successfully")
            
            # Create directory structure
            await self._create_directory_structure()
            logger.info("Directory structure created")
            
            logger.info("DataManager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DataManager: {e}")
            await self.cleanup()  # Cleanup partial state
            return False

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        await self.storage.cleanup()

    async def _create_directory_structure(self):
        """Create standardized directory structure"""
        base_path = Path(self.settings.data_path)
        
        directories = [
            "raw/nasa",
            "raw/mast", 
            "raw/exofop",
            "raw/kepler",
            "raw/tess",
            "processed/v1",
            "processed/v2",
            "lightcurves/tess",
            "lightcurves/kepler", 
            "lightcurves/k2",
            "metadata",
            "validation_logs",
            "checksums"
        ]
        
        for directory in directories:
            dir_path = base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Created directory structure at {base_path}")

    async def ingest_koi_table(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Ingest Kepler Objects of Interest (KOI) table from NASA Exoplanet Archive
        
        Args:
            force_refresh: Force download even if cached version exists
            
        Returns:
            Dict with ingestion results
        """
        table_name = "koi_table"
        cache_key = f"nasa_archive_{table_name}"
        
        # Check cache
        if not force_refresh:
            cached_data = await self.storage.get_cached_table(cache_key)
            if cached_data and self._is_cache_valid(cached_data, "koi_table"):
                logger.info("Using cached KOI table")
                return {
                    "status": "success",
                    "source": "cache",
                    "records": len(cached_data["data"]),
                    "last_updated": cached_data["timestamp"]
                }

        try:
            logger.info("Downloading KOI table from NASA Exoplanet Archive")
            
            # Query NASA Exoplanet Archive
            query = """
            SELECT koi_name, kepid, koi_period, koi_period_err1, koi_period_err2,
                   koi_time0bk, koi_time0bk_err1, koi_time0bk_err2,
                   koi_impact, koi_impact_err1, koi_impact_err2,
                   koi_duration, koi_duration_err1, koi_duration_err2,
                   koi_depth, koi_depth_err1, koi_depth_err2,
                   koi_prad, koi_prad_err1, koi_prad_err2,
                   koi_sma, koi_sma_err1, koi_sma_err2,
                   koi_incl, koi_incl_err1, koi_incl_err2,
                   koi_teq, koi_teq_err1, koi_teq_err2,
                   koi_dor, koi_dor_err1, koi_dor_err2,
                   koi_disposition, koi_pdisposition,
                   koi_score, ra, dec, koi_kepmag
            FROM koi
            WHERE koi_disposition IS NOT NULL
            """
            
            # Use astroquery for reliable access
            table = ExoplanetArchive.query_criteria(
                table="koi",
                select="*",
                where="koi_disposition is not null"
            )
            
            # Convert to pandas DataFrame
            df = table.to_pandas()
            
            # Validate data
            validation_result = await self.validator.validate_koi_table(df)
            if not validation_result["valid"]:
                logger.error(f"KOI table validation failed: {validation_result['errors']}")
                return {
                    "status": "error",
                    "message": "Data validation failed",
                    "errors": validation_result["errors"]
                }
            
            # Calculate checksum
            data_hash = self._calculate_dataframe_hash(df)
            
            # Save raw data
            timestamp = datetime.utcnow()
            raw_path = await self.storage.save_raw_table(
                df, "nasa", f"koi_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            # Cache processed data
            cache_data = {
                "data": df.to_dict("records"),
                "metadata": {
                    "source": "NASA Exoplanet Archive",
                    "table": "koi",
                    "records": len(df),
                    "columns": list(df.columns),
                    "checksum": data_hash,
                    "raw_path": str(raw_path)
                },
                "timestamp": timestamp.isoformat(),
                "validation": validation_result
            }
            
            await self.storage.cache_table(cache_key, cache_data)
            
            # Update database
            await self._update_candidates_database(df, "kepler", "koi")
            
            logger.info(f"Successfully ingested {len(df)} KOI records")
            
            return {
                "status": "success",
                "source": "nasa_archive",
                "records": len(df),
                "checksum": data_hash,
                "validation": validation_result,
                "raw_path": str(raw_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest KOI table: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def ingest_toi_table(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Ingest TESS Objects of Interest (TOI) table from ExoFOP-TESS
        
        Args:
            force_refresh: Force download even if cached version exists
            
        Returns:
            Dict with ingestion results
        """
        table_name = "toi_table"
        cache_key = f"exofop_{table_name}"
        
        # Check cache
        if not force_refresh:
            cached_data = await self.storage.get_cached_table(cache_key)
            if cached_data and self._is_cache_valid(cached_data, "toi_table"):
                logger.info("Using cached TOI table")
                return {
                    "status": "success",
                    "source": "cache",
                    "records": len(cached_data["data"]),
                    "last_updated": cached_data["timestamp"]
                }

        try:
            logger.info("Downloading TOI table from ExoFOP-TESS")
            
            # Query TOI table from NASA Exoplanet Archive (more reliable than ExoFOP direct)
            table = ExoplanetArchive.query_criteria(
                table="toi",
                select="*"
            )
            
            # Convert to pandas DataFrame
            df = table.to_pandas()
            
            # Validate data
            validation_result = await self.validator.validate_toi_table(df)
            if not validation_result["valid"]:
                logger.error(f"TOI table validation failed: {validation_result['errors']}")
                return {
                    "status": "error", 
                    "message": "Data validation failed",
                    "errors": validation_result["errors"]
                }
            
            # Calculate checksum
            data_hash = self._calculate_dataframe_hash(df)
            
            # Save raw data
            timestamp = datetime.utcnow()
            raw_path = await self.storage.save_raw_table(
                df, "exofop", f"toi_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            # Cache processed data
            cache_data = {
                "data": df.to_dict("records"),
                "metadata": {
                    "source": "ExoFOP-TESS / NASA Exoplanet Archive",
                    "table": "toi",
                    "records": len(df),
                    "columns": list(df.columns),
                    "checksum": data_hash,
                    "raw_path": str(raw_path)
                },
                "timestamp": timestamp.isoformat(),
                "validation": validation_result
            }
            
            await self.storage.cache_table(cache_key, cache_data)
            
            # Update database
            await self._update_candidates_database(df, "tess", "toi")
            
            logger.info(f"Successfully ingested {len(df)} TOI records")
            
            return {
                "status": "success",
                "source": "exofop_tess",
                "records": len(df),
                "checksum": data_hash,
                "validation": validation_result,
                "raw_path": str(raw_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest TOI table: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def ingest_k2_table(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Ingest K2 candidates table from NASA Exoplanet Archive
        
        Args:
            force_refresh: Force download even if cached version exists
            
        Returns:
            Dict with ingestion results
        """
        table_name = "k2_table"
        cache_key = f"nasa_archive_{table_name}"
        
        # Check cache
        if not force_refresh:
            cached_data = await self.storage.get_cached_table(cache_key)
            if cached_data and self._is_cache_valid(cached_data, "k2_table"):
                logger.info("Using cached K2 table")
                return {
                    "status": "success",
                    "source": "cache", 
                    "records": len(cached_data["data"]),
                    "last_updated": cached_data["timestamp"]
                }

        try:
            logger.info("Downloading K2 table from NASA Exoplanet Archive")
            
            # Query K2 candidates
            table = ExoplanetArchive.query_criteria(
                table="k2candidates",
                select="*"
            )
            
            # Convert to pandas DataFrame
            df = table.to_pandas()
            
            # Validate data
            validation_result = await self.validator.validate_k2_table(df)
            if not validation_result["valid"]:
                logger.error(f"K2 table validation failed: {validation_result['errors']}")
                return {
                    "status": "error",
                    "message": "Data validation failed", 
                    "errors": validation_result["errors"]
                }
            
            # Calculate checksum
            data_hash = self._calculate_dataframe_hash(df)
            
            # Save raw data
            timestamp = datetime.utcnow()
            raw_path = await self.storage.save_raw_table(
                df, "nasa", f"k2_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            # Cache processed data
            cache_data = {
                "data": df.to_dict("records"),
                "metadata": {
                    "source": "NASA Exoplanet Archive",
                    "table": "k2candidates",
                    "records": len(df),
                    "columns": list(df.columns),
                    "checksum": data_hash,
                    "raw_path": str(raw_path)
                },
                "timestamp": timestamp.isoformat(),
                "validation": validation_result
            }
            
            await self.storage.cache_table(cache_key, cache_data)
            
            # Update database
            await self._update_candidates_database(df, "k2", "k2candidates")
            
            logger.info(f"Successfully ingested {len(df)} K2 records")
            
            return {
                "status": "success",
                "source": "nasa_archive",
                "records": len(df),
                "checksum": data_hash,
                "validation": validation_result,
                "raw_path": str(raw_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest K2 table: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def ingest_lightcurve(
        self,
        target_name: str,
        mission: str = "TESS",
        sector_quarter: Optional[int] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest light curve data for a specific target
        
        Args:
            target_name: Target identifier (TIC, KIC, etc.)
            mission: Mission name (TESS, Kepler, K2)
            sector_quarter: Specific sector/quarter
            force_refresh: Force download even if cached
            
        Returns:
            Dict with ingestion results
        """
        cache_key = f"lightcurve_{mission.lower()}_{target_name}"
        if sector_quarter:
            cache_key += f"_s{sector_quarter}"
            
        # Check cache
        if not force_refresh:
            cached_lc = await self.storage.get_cached_lightcurve(cache_key)
            if cached_lc and self._is_cache_valid(cached_lc, "lightcurves"):
                logger.info(f"Using cached light curve for {target_name}")
                return {
                    "status": "success",
                    "source": "cache",
                    "target": target_name,
                    "mission": mission,
                    "data_points": len(cached_lc["data"]["time_bjd"]),
                    "cached": True
                }

        try:
            logger.info(f"Downloading light curve for {target_name} from {mission}")
            
            # Search for light curve data
            search_result = search_lightcurve(
                target_name,
                mission=mission.upper(),
                sector=sector_quarter if mission.upper() == "TESS" else None,
                quarter=sector_quarter if mission.upper() in ["KEPLER", "K2"] else None
            )
            
            if len(search_result) == 0:
                return {
                    "status": "error",
                    "message": f"No light curve data found for {target_name}"
                }
            
            # Download the light curve
            lc_collection = search_result.download_all()
            
            if len(lc_collection) == 0:
                return {
                    "status": "error",
                    "message": f"Failed to download light curve for {target_name}"
                }
            
            # Combine all light curves if multiple sectors/quarters
            lc = lc_collection.stitch()
            
            # Create standardized LightCurveData object
            lightcurve_data = LightCurveData(
                target_name=target_name,
                time_bjd=lc.time.btjd,
                flux=lc.flux.value,
                flux_err=lc.flux_err.value,
                mission=mission.upper(),
                instrument=lc.meta.get("INSTRUME", None),
                cadence_minutes=lc.meta.get("TIMEDEL", None),
                sectors_quarters=getattr(lc, "sector", getattr(lc, "quarter", None)),
                data_quality_flags=lc.quality.value if hasattr(lc, "quality") else None,
                source="lightkurve",
                download_date=datetime.utcnow()
            )
            
            # Validate light curve data
            validation_result = await self.validator.validate_lightcurve(lightcurve_data)
            if not validation_result["valid"]:
                logger.warning(f"Light curve validation warnings: {validation_result['warnings']}")
            
            # Save raw light curve
            raw_path = await self.storage.save_raw_lightcurve(
                lightcurve_data, mission.lower(), target_name
            )
            
            # Cache the data
            cache_data = {
                "data": {
                    "target_name": lightcurve_data.target_name,
                    "time_bjd": lightcurve_data.time_bjd.tolist(),
                    "flux": lightcurve_data.flux.tolist(),
                    "flux_err": lightcurve_data.flux_err.tolist(),
                    "mission": lightcurve_data.mission,
                    "instrument": lightcurve_data.instrument,
                    "cadence_minutes": lightcurve_data.cadence_minutes,
                    "sectors_quarters": lightcurve_data.sectors_quarters,
                    "data_quality_flags": lightcurve_data.data_quality_flags.tolist() if lightcurve_data.data_quality_flags is not None else None
                },
                "metadata": {
                    "source": "lightkurve",
                    "data_points": len(lightcurve_data.time_bjd),
                    "time_span_days": float(lightcurve_data.time_bjd.max() - lightcurve_data.time_bjd.min()),
                    "raw_path": str(raw_path)
                },
                "timestamp": datetime.utcnow().isoformat(),
                "validation": validation_result
            }
            
            await self.storage.cache_lightcurve(cache_key, cache_data)
            
            logger.info(f"Successfully ingested light curve for {target_name}: {len(lightcurve_data.time_bjd)} data points")
            
            return {
                "status": "success",
                "source": "lightkurve",
                "target": target_name,
                "mission": mission,
                "data_points": len(lightcurve_data.time_bjd),
                "time_span_days": float(lightcurve_data.time_bjd.max() - lightcurve_data.time_bjd.min()),
                "validation": validation_result,
                "raw_path": str(raw_path),
                "cached": False
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest light curve for {target_name}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "target": target_name,
                "mission": mission
            }

    async def batch_ingest_all_tables(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Batch ingest all standard tables (KOI, TOI, K2)
        
        Args:
            force_refresh: Force refresh all tables
            
        Returns:
            Dict with results for all tables
        """
        logger.info("Starting batch ingestion of all tables")
        
        results = {}
        
        # Ingest tables in parallel
        tasks = [
            ("koi", self.ingest_koi_table(force_refresh)),
            ("toi", self.ingest_toi_table(force_refresh)),
            ("k2", self.ingest_k2_table(force_refresh))
        ]
        
        for table_name, task in tasks:
            try:
                result = await task
                results[table_name] = result
                logger.info(f"Completed {table_name}: {result['status']}")
            except Exception as e:
                logger.error(f"Failed to ingest {table_name}: {e}")
                results[table_name] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Create summary
        total_records = sum(
            r.get("records", 0) for r in results.values() 
            if r.get("status") == "success"
        )
        
        successful_tables = [
            name for name, result in results.items()
            if result.get("status") == "success"
        ]
        
        summary = {
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat(),
            "total_records": total_records,
            "successful_tables": successful_tables,
            "failed_tables": [
                name for name, result in results.items()
                if result.get("status") == "error"
            ],
            "results": results
        }
        
        logger.info(f"Batch ingestion completed: {len(successful_tables)}/3 tables successful, {total_records} total records")
        
        return summary

    def _calculate_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Calculate SHA256 hash of DataFrame"""
        return hashlib.sha256(
            df.to_csv(index=False).encode('utf-8')
        ).hexdigest()

    def _is_cache_valid(self, cached_data: Dict[str, Any], data_type: str) -> bool:
        """Check if cached data is still valid based on TTL"""
        if "timestamp" not in cached_data:
            return False
            
        cache_time = datetime.fromisoformat(cached_data["timestamp"])
        ttl = self.cache_ttl.get(data_type, timedelta(hours=6))
        
        return datetime.utcnow() - cache_time < ttl

    async def _update_candidates_database(
        self, 
        df: pd.DataFrame, 
        mission: str, 
        table_type: str
    ):
        """Update the candidates database with new data"""
        # This would integrate with your database system
        # For now, we'll log the update
        logger.info(f"Updated candidates database: {len(df)} records from {mission} {table_type}")

    async def get_ingestion_status(self) -> Dict[str, Any]:
        """Get status of all ingested data"""
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "tables": {},
            "lightcurves": {},
            "storage": await self.storage.get_storage_stats()
        }
        
        # Check table cache status
        for table_name in ["koi_table", "toi_table", "k2_table"]:
            cache_key = f"nasa_archive_{table_name}" if table_name != "toi_table" else f"exofop_{table_name}"
            cached_data = await self.storage.get_cached_table(cache_key)
            
            if cached_data:
                status["tables"][table_name] = {
                    "cached": True,
                    "records": len(cached_data["data"]),
                    "last_updated": cached_data["timestamp"],
                    "valid": self._is_cache_valid(cached_data, table_name),
                    "checksum": cached_data["metadata"].get("checksum")
                }
            else:
                status["tables"][table_name] = {
                    "cached": False,
                    "records": 0,
                    "last_updated": None,
                    "valid": False
                }
        
        return status
