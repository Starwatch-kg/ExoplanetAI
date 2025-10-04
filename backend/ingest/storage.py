"""
Storage management system for ExoplanetAI data
Система управления хранением данных для ExoplanetAI
"""

import asyncio
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles
import numpy as np
import pandas as pd
import redis.asyncio as redis
from astropy.io import fits

from core.config import get_settings
from data_sources.base import LightCurveData

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Manages all data storage operations including:
    - Raw data storage (CSV, FITS)
    - Processed data storage
    - Light curve storage
    - Metadata management
    - Redis caching
    """

    def __init__(self):
        self.settings = get_settings()
        self.base_path = Path(self.settings.data_path)
        self.redis_client: Optional[redis.Redis] = None
        
        # Storage paths
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.lightcurves_path = self.base_path / "lightcurves"
        self.metadata_path = self.base_path / "metadata"
        self.checksums_path = self.base_path / "checksums"

    async def initialize(self) -> bool:
        """Initialize storage system"""
        try:
            # Create directory structure
            for path in [self.raw_path, self.processed_path, self.lightcurves_path, 
                        self.metadata_path, self.checksums_path]:
                path.mkdir(parents=True, exist_ok=True)
            
            # Initialize Redis connection
            if hasattr(self.settings, 'redis_url'):
                self.redis_client = redis.from_url(
                    self.settings.redis_url,
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("Redis connection established")
            else:
                logger.warning("Redis not configured, using file-based caching")
            
            logger.info(f"Storage system initialized at {self.base_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize storage system: {e}")
            return False

    async def cleanup(self):
        """Cleanup storage resources"""
        if self.redis_client:
            await self.redis_client.close()

    # Raw Data Storage
    async def save_raw_table(
        self, 
        df: pd.DataFrame, 
        source: str, 
        filename: str
    ) -> Path:
        """
        Save raw table data to CSV
        
        Args:
            df: DataFrame to save
            source: Data source (nasa, mast, exofop)
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        source_path = self.raw_path / source
        source_path.mkdir(exist_ok=True)
        
        file_path = source_path / filename
        
        # Save CSV with metadata
        df.to_csv(file_path, index=False)
        
        # Save metadata
        metadata = {
            "filename": filename,
            "source": source,
            "records": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "created_at": datetime.utcnow().isoformat(),
            "file_size_bytes": file_path.stat().st_size
        }
        
        metadata_file = self.metadata_path / f"{file_path.stem}_metadata.json"
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(metadata, indent=2))
        
        logger.info(f"Saved raw table: {file_path} ({len(df)} records)")
        return file_path

    async def save_raw_lightcurve(
        self, 
        lightcurve: LightCurveData, 
        mission: str, 
        target_name: str
    ) -> Path:
        """
        Save raw light curve data to FITS format
        
        Args:
            lightcurve: LightCurveData object
            mission: Mission name (tess, kepler, k2)
            target_name: Target identifier
            
        Returns:
            Path to saved file
        """
        mission_path = self.lightcurves_path / mission.lower()
        mission_path.mkdir(exist_ok=True)
        
        # Create safe filename
        safe_target = "".join(c for c in target_name if c.isalnum() or c in ".-_")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_target}_{timestamp}.fits"
        file_path = mission_path / filename
        
        # Create FITS file
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['TARGET'] = lightcurve.target_name
        primary_hdu.header['MISSION'] = lightcurve.mission
        primary_hdu.header['INSTRUME'] = lightcurve.instrument or 'UNKNOWN'
        primary_hdu.header['CADENCE'] = lightcurve.cadence_minutes or 0.0
        primary_hdu.header['CREATED'] = datetime.utcnow().isoformat()
        
        # Create data table
        cols = [
            fits.Column(name='TIME_BJD', format='D', array=lightcurve.time_bjd),
            fits.Column(name='FLUX', format='D', array=lightcurve.flux),
            fits.Column(name='FLUX_ERR', format='D', array=lightcurve.flux_err)
        ]
        
        if lightcurve.data_quality_flags is not None:
            cols.append(fits.Column(name='QUALITY', format='J', array=lightcurve.data_quality_flags))
        
        table_hdu = fits.BinTableHDU.from_columns(cols)
        table_hdu.header['EXTNAME'] = 'LIGHTCURVE'
        
        # Write FITS file
        hdul = fits.HDUList([primary_hdu, table_hdu])
        hdul.writeto(file_path, overwrite=True)
        
        # Save metadata
        metadata = {
            "filename": filename,
            "target_name": lightcurve.target_name,
            "mission": lightcurve.mission,
            "instrument": lightcurve.instrument,
            "data_points": len(lightcurve.time_bjd),
            "time_span_days": float(lightcurve.time_bjd.max() - lightcurve.time_bjd.min()),
            "cadence_minutes": lightcurve.cadence_minutes,
            "sectors_quarters": lightcurve.sectors_quarters,
            "detrended": lightcurve.detrended,
            "normalized": lightcurve.normalized,
            "outliers_removed": lightcurve.outliers_removed,
            "created_at": datetime.utcnow().isoformat(),
            "file_size_bytes": file_path.stat().st_size
        }
        
        metadata_file = self.metadata_path / f"{file_path.stem}_lightcurve_metadata.json"
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(metadata, indent=2))
        
        logger.info(f"Saved raw light curve: {file_path} ({len(lightcurve.time_bjd)} points)")
        return file_path

    # Processed Data Storage
    async def save_processed_table(
        self, 
        df: pd.DataFrame, 
        version: str, 
        table_name: str,
        processing_info: Dict[str, Any]
    ) -> Path:
        """
        Save processed table data
        
        Args:
            df: Processed DataFrame
            version: Processing version (v1, v2, etc.)
            table_name: Table identifier
            processing_info: Information about processing steps
            
        Returns:
            Path to saved file
        """
        version_path = self.processed_path / version
        version_path.mkdir(exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{table_name}_processed_{timestamp}.csv"
        file_path = version_path / filename
        
        # Save processed data
        df.to_csv(file_path, index=False)
        
        # Save processing metadata
        metadata = {
            "filename": filename,
            "table_name": table_name,
            "version": version,
            "records": len(df),
            "columns": list(df.columns),
            "processing_info": processing_info,
            "created_at": datetime.utcnow().isoformat(),
            "file_size_bytes": file_path.stat().st_size
        }
        
        metadata_file = self.metadata_path / f"{file_path.stem}_processed_metadata.json"
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(metadata, indent=2))
        
        logger.info(f"Saved processed table: {file_path} ({len(df)} records)")
        return file_path

    async def save_processed_lightcurve(
        self,
        lightcurve: LightCurveData,
        processing_steps: List[str],
        version: str = "v1"
    ) -> Path:
        """
        Save processed light curve data
        
        Args:
            lightcurve: Processed LightCurveData
            processing_steps: List of processing steps applied
            version: Processing version
            
        Returns:
            Path to saved file
        """
        version_path = self.processed_path / version / "lightcurves"
        version_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        safe_target = "".join(c for c in lightcurve.target_name if c.isalnum() or c in ".-_")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_target}_processed_{timestamp}.fits"
        file_path = version_path / filename
        
        # Create FITS file with processing info
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['TARGET'] = lightcurve.target_name
        primary_hdu.header['MISSION'] = lightcurve.mission
        primary_hdu.header['VERSION'] = version
        primary_hdu.header['DETREND'] = lightcurve.detrended
        primary_hdu.header['NORMALIZ'] = lightcurve.normalized
        primary_hdu.header['OUTLIERS'] = lightcurve.outliers_removed
        primary_hdu.header['CREATED'] = datetime.utcnow().isoformat()
        
        # Add processing steps to header
        for i, step in enumerate(processing_steps[:20]):  # Limit to 20 steps
            primary_hdu.header[f'STEP{i:02d}'] = step
        
        # Create data table
        cols = [
            fits.Column(name='TIME_BJD', format='D', array=lightcurve.time_bjd),
            fits.Column(name='FLUX', format='D', array=lightcurve.flux),
            fits.Column(name='FLUX_ERR', format='D', array=lightcurve.flux_err)
        ]
        
        if lightcurve.data_quality_flags is not None:
            cols.append(fits.Column(name='QUALITY', format='J', array=lightcurve.data_quality_flags))
        
        table_hdu = fits.BinTableHDU.from_columns(cols)
        table_hdu.header['EXTNAME'] = 'PROCESSED_LIGHTCURVE'
        
        # Write FITS file
        hdul = fits.HDUList([primary_hdu, table_hdu])
        hdul.writeto(file_path, overwrite=True)
        
        logger.info(f"Saved processed light curve: {file_path}")
        return file_path

    # Caching System
    async def cache_table(self, cache_key: str, data: Dict[str, Any], ttl: int = 21600):
        """
        Cache table data (Redis or file-based)
        
        Args:
            cache_key: Unique cache key
            data: Data to cache
            ttl: Time to live in seconds (default 6 hours)
        """
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"table:{cache_key}",
                    ttl,
                    json.dumps(data, default=str)
                )
                logger.debug(f"Cached table data to Redis: {cache_key}")
                return
            except Exception as e:
                logger.warning(f"Redis cache failed, using file cache: {e}")
        
        # Fallback to file-based cache
        cache_path = self.base_path / "cache" / "tables"
        cache_path.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_path / f"{cache_key}.json"
        async with aiofiles.open(cache_file, 'w') as f:
            await f.write(json.dumps(data, indent=2, default=str))
        
        logger.debug(f"Cached table data to file: {cache_file}")

    async def get_cached_table(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached table data
        
        Args:
            cache_key: Cache key to retrieve
            
        Returns:
            Cached data or None if not found
        """
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(f"table:{cache_key}")
                if cached_data:
                    return json.loads(cached_data)
            except Exception as e:
                logger.warning(f"Redis cache retrieval failed: {e}")
        
        # Fallback to file-based cache
        cache_file = self.base_path / "cache" / "tables" / f"{cache_key}.json"
        if cache_file.exists():
            try:
                async with aiofiles.open(cache_file, 'r') as f:
                    content = await f.read()
                    return json.loads(content)
            except Exception as e:
                logger.warning(f"File cache retrieval failed: {e}")
        
        return None

    async def cache_lightcurve(self, cache_key: str, data: Dict[str, Any], ttl: int = 21600):
        """
        Cache light curve data
        
        Args:
            cache_key: Unique cache key
            data: Light curve data to cache
            ttl: Time to live in seconds (default 6 hours)
        """
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"lightcurve:{cache_key}",
                    ttl,
                    json.dumps(data, default=str)
                )
                logger.debug(f"Cached light curve to Redis: {cache_key}")
                return
            except Exception as e:
                logger.warning(f"Redis cache failed, using file cache: {e}")
        
        # Fallback to file-based cache
        cache_path = self.base_path / "cache" / "lightcurves"
        cache_path.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_path / f"{cache_key}.json"
        async with aiofiles.open(cache_file, 'w') as f:
            await f.write(json.dumps(data, indent=2, default=str))
        
        logger.debug(f"Cached light curve to file: {cache_file}")

    async def get_cached_lightcurve(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached light curve data
        
        Args:
            cache_key: Cache key to retrieve
            
        Returns:
            Cached light curve data or None if not found
        """
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(f"lightcurve:{cache_key}")
                if cached_data:
                    return json.loads(cached_data)
            except Exception as e:
                logger.warning(f"Redis cache retrieval failed: {e}")
        
        # Fallback to file-based cache
        cache_file = self.base_path / "cache" / "lightcurves" / f"{cache_key}.json"
        if cache_file.exists():
            try:
                async with aiofiles.open(cache_file, 'r') as f:
                    content = await f.read()
                    return json.loads(content)
            except Exception as e:
                logger.warning(f"File cache retrieval failed: {e}")
        
        return None

    # Storage Statistics and Management
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "base_path": str(self.base_path),
            "total_size_mb": 0,
            "directories": {},
            "file_counts": {},
            "cache_stats": {}
        }
        
        # Calculate directory sizes and file counts
        for directory in [self.raw_path, self.processed_path, self.lightcurves_path, self.metadata_path]:
            if directory.exists():
                size_bytes = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
                file_count = len([f for f in directory.rglob('*') if f.is_file()])
                
                dir_name = directory.name
                stats["directories"][dir_name] = {
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                    "file_count": file_count
                }
                stats["total_size_mb"] += stats["directories"][dir_name]["size_mb"]
        
        # Cache statistics
        if self.redis_client:
            try:
                info = await self.redis_client.info()
                stats["cache_stats"]["redis"] = {
                    "connected": True,
                    "used_memory_mb": round(info.get("used_memory", 0) / (1024 * 1024), 2),
                    "keys": info.get("db0", {}).get("keys", 0) if "db0" in info else 0
                }
            except Exception as e:
                stats["cache_stats"]["redis"] = {"connected": False, "error": str(e)}
        else:
            stats["cache_stats"]["redis"] = {"connected": False, "reason": "not_configured"}
        
        # File-based cache stats
        cache_path = self.base_path / "cache"
        if cache_path.exists():
            cache_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
            cache_files = len([f for f in cache_path.rglob('*') if f.is_file()])
            stats["cache_stats"]["file_cache"] = {
                "size_mb": round(cache_size / (1024 * 1024), 2),
                "file_count": cache_files
            }
        
        return stats

    async def cleanup_old_cache(self, max_age_days: int = 7):
        """
        Clean up old cache files
        
        Args:
            max_age_days: Maximum age of cache files to keep
        """
        cutoff_time = datetime.utcnow().timestamp() - (max_age_days * 24 * 3600)
        cleaned_files = 0
        
        cache_path = self.base_path / "cache"
        if cache_path.exists():
            for cache_file in cache_path.rglob('*.json'):
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    cleaned_files += 1
        
        logger.info(f"Cleaned up {cleaned_files} old cache files")
        return cleaned_files

    async def create_backup(self, backup_path: Optional[Path] = None) -> Path:
        """
        Create backup of all data
        
        Args:
            backup_path: Custom backup path (optional)
            
        Returns:
            Path to backup directory
        """
        if backup_path is None:
            backup_path = self.base_path.parent / "backups" / f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all data directories
        for source_dir in [self.raw_path, self.processed_path, self.lightcurves_path, self.metadata_path]:
            if source_dir.exists():
                dest_dir = backup_path / source_dir.name
                shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
        
        # Create backup manifest
        manifest = {
            "created_at": datetime.utcnow().isoformat(),
            "source_path": str(self.base_path),
            "backup_path": str(backup_path),
            "directories_backed_up": [d.name for d in [self.raw_path, self.processed_path, self.lightcurves_path, self.metadata_path] if d.exists()]
        }
        
        manifest_file = backup_path / "backup_manifest.json"
        async with aiofiles.open(manifest_file, 'w') as f:
            await f.write(json.dumps(manifest, indent=2))
        
        logger.info(f"Created backup at {backup_path}")
        return backup_path
