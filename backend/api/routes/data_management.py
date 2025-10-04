"""
API routes for data management and preprocessing
API маршруты для управления данными и предобработки
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from auth.dependencies import get_current_user, require_role
from auth.models import User, UserRole
from ingest.data_manager import DataManager
from ingest.storage import StorageManager
from ingest.validator import DataValidator
from ingest.versioning import VersionManager
from preprocessing.lightcurve_processor import LightCurveProcessor
from core.rate_limiting import get_rate_limiter
from core.validation import (
    DataIngestionRequest,
    LightCurveIngestionRequest,
    PreprocessingRequest,
    VersionCreateRequest,
    DataValidationRequest,
    SecurityValidator
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/data", tags=["data_management"])


async def check_rate_limit_middleware(
    request: Request,
    current_user: Optional[User] = None
):
    """Middleware to check rate limits for API endpoints"""
    rate_limiter = await get_rate_limiter()
    
    # Get user identifier
    if current_user:
        identifier = f"user:{current_user.username}"
        user_role = current_user.role
    else:
        # Use IP address for unauthenticated requests
        client_ip = request.client.host if request.client else "unknown"
        identifier = f"ip:{client_ip}"
        user_role = UserRole.GUEST
    
    # Check rate limit
    endpoint = str(request.url.path)
    is_allowed, limit_info = await rate_limiter.check_rate_limit(
        identifier, endpoint, user_role
    )
    
    if not is_allowed:
        logger.warning(
            f"Rate limit exceeded for {identifier} on {endpoint}: {limit_info['violations']}"
        )
        
        # Return detailed rate limit information
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "violations": limit_info["violations"],
                "remaining": limit_info["remaining"],
                "reset_times": limit_info["reset_times"]
            },
            headers={
                "Retry-After": str(int(min(
                    violation["reset_time"] - __import__("time").time()
                    for violation in limit_info["violations"]
                )))
            }
        )
    
    return None  # Allow request to proceed


# Enterprise-grade dependency injection with proper lifecycle management
import asyncio
from functools import lru_cache
from typing import Dict, Any

# Global managers for lifecycle management
_global_managers: Dict[str, Any] = {}
_initialization_locks: Dict[str, asyncio.Lock] = {}

async def get_initialized_data_manager() -> DataManager:
    """Get properly initialized DataManager singleton"""
    if 'data_manager' not in _global_managers:
        if 'data_manager' not in _initialization_locks:
            _initialization_locks['data_manager'] = asyncio.Lock()
        
        async with _initialization_locks['data_manager']:
            if 'data_manager' not in _global_managers:
                manager = DataManager()
                success = await manager.initialize()
                if not success:
                    raise RuntimeError("DataManager initialization failed")
                _global_managers['data_manager'] = manager
                logger.info("✅ DataManager initialized and cached")
    
    return _global_managers['data_manager']

async def get_initialized_storage_manager() -> StorageManager:
    """Get properly initialized StorageManager singleton"""
    if 'storage_manager' not in _global_managers:
        if 'storage_manager' not in _initialization_locks:
            _initialization_locks['storage_manager'] = asyncio.Lock()
        
        async with _initialization_locks['storage_manager']:
            if 'storage_manager' not in _global_managers:
                manager = StorageManager()
                success = await manager.initialize()
                if not success:
                    raise RuntimeError("StorageManager initialization failed")
                _global_managers['storage_manager'] = manager
                logger.info("✅ StorageManager initialized and cached")
    
    return _global_managers['storage_manager']

async def get_initialized_validator() -> DataValidator:
    """Get properly initialized DataValidator singleton"""
    if 'validator' not in _global_managers:
        if 'validator' not in _initialization_locks:
            _initialization_locks['validator'] = asyncio.Lock()
        
        async with _initialization_locks['validator']:
            if 'validator' not in _global_managers:
                manager = DataValidator()
                # DataValidator doesn't need async initialization, but we keep pattern consistent
                _global_managers['validator'] = manager
                logger.info("✅ DataValidator initialized and cached")
    
    return _global_managers['validator']

async def get_initialized_version_manager() -> VersionManager:
    """Get properly initialized VersionManager singleton"""
    if 'version_manager' not in _global_managers:
        if 'version_manager' not in _initialization_locks:
            _initialization_locks['version_manager'] = asyncio.Lock()
        
        async with _initialization_locks['version_manager']:
            if 'version_manager' not in _global_managers:
                manager = VersionManager()
                success = await manager.initialize()
                if not success:
                    raise RuntimeError("VersionManager initialization failed")
                _global_managers['version_manager'] = manager
                logger.info("✅ VersionManager initialized and cached")
    
    return _global_managers['version_manager']

async def get_initialized_lightcurve_processor() -> LightCurveProcessor:
    """Get properly initialized LightCurveProcessor singleton"""
    if 'lightcurve_processor' not in _global_managers:
        if 'lightcurve_processor' not in _initialization_locks:
            _initialization_locks['lightcurve_processor'] = asyncio.Lock()
        
        async with _initialization_locks['lightcurve_processor']:
            if 'lightcurve_processor' not in _global_managers:
                processor = LightCurveProcessor()
                # LightCurveProcessor doesn't need async initialization
                _global_managers['lightcurve_processor'] = processor
                logger.info("✅ LightCurveProcessor initialized and cached")
    
    return _global_managers['lightcurve_processor']

# DEPRECATED FUNCTIONS REMOVED - Use get_initialized_* versions instead
# These functions have been removed to prevent usage of deprecated DI pattern


@router.on_event("startup")
async def startup_data_management():
    """Initialize all data management components with proper error handling"""
    logger.info("🚀 Starting data management components initialization...")
    
    try:
        # Pre-initialize all managers to ensure they're ready
        managers_to_init = [
            ("DataManager", get_initialized_data_manager),
            ("StorageManager", get_initialized_storage_manager), 
            ("DataValidator", get_initialized_validator),
            ("VersionManager", get_initialized_version_manager),
            ("LightCurveProcessor", get_initialized_lightcurve_processor)
        ]
        
        for name, init_func in managers_to_init:
            try:
                await init_func()
                logger.info(f"✅ {name} startup completed")
            except Exception as e:
                logger.error(f"❌ {name} startup failed: {e}")
                # Continue with other managers, but log the failure
                
        logger.info("🎉 Data management startup completed successfully")
        
    except Exception as e:
        logger.error(f"💥 Critical error during data management startup: {e}")
        # Attempt cleanup of partially initialized state
        await shutdown_data_management()
        raise RuntimeError(f"Data management startup failed: {e}")


@router.on_event("shutdown")
async def shutdown_data_management():
    """Cleanup all data management components with proper error handling"""
    logger.info("🛑 Starting data management components shutdown...")
    
    cleanup_errors = []
    
    # Cleanup all initialized managers
    for name, manager in _global_managers.items():
        try:
            if hasattr(manager, 'cleanup') and callable(getattr(manager, 'cleanup')):
                await manager.cleanup()
                logger.info(f"✅ {name} cleanup completed")
            else:
                logger.info(f"ℹ️ {name} doesn't require cleanup")
        except Exception as e:
            error_msg = f"❌ {name} cleanup failed: {e}"
            logger.error(error_msg)
            cleanup_errors.append(error_msg)
    
    # Clear the global state
    _global_managers.clear()
    _initialization_locks.clear()
    
    if cleanup_errors:
        logger.warning(f"⚠️ Shutdown completed with {len(cleanup_errors)} errors")
        for error in cleanup_errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("🎉 Data management shutdown completed successfully")


async def cleanup_managers():
    """Utility function to cleanup managers (for use in main lifespan)"""
    await shutdown_data_management()


# Data Ingestion Endpoints with Enterprise Security
@router.post("/ingest/table")
async def ingest_table(
    request: DataIngestionRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    current_user: User = Depends(get_current_user),
    data_manager: DataManager = Depends(get_initialized_data_manager)
):
    """
    Ingest astronomical data table (KOI, TOI, K2)
    
    Requires: User role or higher
    """
    require_role(current_user, UserRole.USER)
    
    # Check rate limits
    rate_limit_response = await check_rate_limit_middleware(http_request, current_user)
    if rate_limit_response:
        return rate_limit_response
    
    logger.info(f"User {current_user.username} requested {request.table_type} table ingestion")
    
    try:
        if request.table_type == "koi":
            result = await data_manager.ingest_koi_table(request.force_refresh)
        elif request.table_type == "toi":
            result = await data_manager.ingest_toi_table(request.force_refresh)
        elif request.table_type == "k2":
            result = await data_manager.ingest_k2_table(request.force_refresh)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown table type: {request.table_type}"
            )
        
        return {
            "status": "success",
            "message": f"{request.table_type.upper()} table ingestion completed",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Table ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/lightcurve")
async def ingest_lightcurve(
    request: LightCurveIngestionRequest,
    current_user: User = Depends(get_current_user),
    data_manager: DataManager = Depends(get_initialized_data_manager)
):
    """
    Ingest light curve data for specific target
    
    Requires: User role or higher
    """
    require_role(current_user, UserRole.USER)
    
    logger.info(f"User {current_user.username} requested lightcurve ingestion for {request.target_name}")
    
    try:
        result = await data_manager.ingest_lightcurve(
            target_name=request.target_name,
            mission=request.mission,
            sector_quarter=request.sector_quarter,
            force_refresh=request.force_refresh
        )
        
        return {
            "status": "success",
            "message": f"Light curve ingestion completed for {request.target_name}",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Light curve ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/batch")
async def batch_ingest_all(
    force_refresh: bool = Query(False, description="Force refresh all tables"),
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user),
    data_manager: DataManager = Depends(get_initialized_data_manager)
):
    """
    Batch ingest all standard tables (KOI, TOI, K2)
    
    Requires: Researcher role or higher
    """
    require_role(current_user, UserRole.RESEARCHER)
    
    logger.info(f"User {current_user.username} requested batch ingestion")
    
    try:
        # Run in background for long-running operation
        if background_tasks:
            background_tasks.add_task(
                data_manager.batch_ingest_all_tables,
                force_refresh
            )
            
            return {
                "status": "accepted",
                "message": "Batch ingestion started in background"
            }
        else:
            result = await data_manager.batch_ingest_all_tables(force_refresh)
            return {
                "status": "success",
                "message": "Batch ingestion completed",
                "result": result
            }
        
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Data Validation Endpoints
@router.post("/validate")
async def validate_data(
    request: DataValidationRequest,
    current_user: User = Depends(get_current_user),
    data_manager: DataManager = Depends(get_initialized_data_manager)
):
    """
    Validate ingested data
    
    Requires: User role or higher
    """
    require_role(current_user, UserRole.USER)
    
    logger.info(f"User {current_user.username} requested data validation for {request.data_type}")
    
    try:
        # Get cached data for validation
        if request.data_type in ["koi", "toi", "k2"]:
            cache_key = f"nasa_archive_{request.data_type}_table" if request.data_type != "toi" else "exofop_toi_table"
            cached_data = await storage_manager.get_cached_table(cache_key)
            
            if not cached_data:
                raise HTTPException(
                    status_code=404,
                    detail=f"No cached {request.data_type} data found. Please ingest data first."
                )
            
            # Convert to DataFrame for validation
            import pandas as pd
            df = pd.DataFrame(cached_data["data"])
            
            # Validate based on data type
            if request.data_type == "koi":
                result = await validator.validate_koi_table(df)
            elif request.data_type == "toi":
                result = await validator.validate_toi_table(df)
            elif request.data_type == "k2":
                result = await validator.validate_k2_table(df)
            
        elif request.data_type == "lightcurve":
            if not request.target_name:
                raise HTTPException(
                    status_code=400,
                    detail="target_name required for lightcurve validation"
                )
            
            # Get cached lightcurve data
            cache_key = f"lightcurve_tess_{request.target_name}"
            cached_lc = await storage_manager.get_cached_lightcurve(cache_key)
            
            if not cached_lc:
                raise HTTPException(
                    status_code=404,
                    detail=f"No cached lightcurve data found for {request.target_name}"
                )
            
            # Convert to LightCurveData object and validate
            from data_sources.base import LightCurveData
            import numpy as np
            
            lc_data = LightCurveData(
                target_name=cached_lc["data"]["target_name"],
                time_bjd=np.array(cached_lc["data"]["time_bjd"]),
                flux=np.array(cached_lc["data"]["flux"]),
                flux_err=np.array(cached_lc["data"]["flux_err"]),
                mission=cached_lc["data"]["mission"],
                instrument=cached_lc["data"]["instrument"],
                cadence_minutes=cached_lc["data"]["cadence_minutes"],
                sectors_quarters=cached_lc["data"]["sectors_quarters"],
                data_quality_flags=np.array(cached_lc["data"]["data_quality_flags"]) if cached_lc["data"]["data_quality_flags"] else None
            )
            
            result = await validator.validate_lightcurve(lc_data)
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown data type: {request.data_type}"
            )
        
        return {
            "status": "success",
            "message": f"Data validation completed for {request.data_type}",
            "validation_result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Preprocessing Endpoints
@router.post("/preprocess/lightcurve")
async def preprocess_lightcurve(
    request: PreprocessingRequest,
    current_user: User = Depends(get_current_user),
    data_manager: DataManager = Depends(get_initialized_data_manager)
):
    """
    Preprocess light curve data
    
    Requires: User role or higher
    """
    require_role(current_user, UserRole.USER)
    
    logger.info(f"User {current_user.username} requested lightcurve preprocessing for {request.target_name}")
    
    try:
        # Get cached lightcurve data
        cache_key = f"lightcurve_{request.mission.lower()}_{request.target_name}"
        cached_lc = await storage_manager.get_cached_lightcurve(cache_key)
        
        if not cached_lc:
            raise HTTPException(
                status_code=404,
                detail=f"No lightcurve data found for {request.target_name}. Please ingest data first."
            )
        
        # Convert to LightCurveData object
        from data_sources.base import LightCurveData
        import numpy as np
        
        lc_data = LightCurveData(
            target_name=cached_lc["data"]["target_name"],
            time_bjd=np.array(cached_lc["data"]["time_bjd"]),
            flux=np.array(cached_lc["data"]["flux"]),
            flux_err=np.array(cached_lc["data"]["flux_err"]),
            mission=cached_lc["data"]["mission"],
            instrument=cached_lc["data"]["instrument"],
            cadence_minutes=cached_lc["data"]["cadence_minutes"],
            sectors_quarters=cached_lc["data"]["sectors_quarters"],
            data_quality_flags=np.array(cached_lc["data"]["data_quality_flags"]) if cached_lc["data"]["data_quality_flags"] else None
        )
        
        # Process lightcurve
        processed_lc, processing_report = await lightcurve_processor.process_lightcurve(
            lc_data, request.processing_params
        )
        
        # Cache processed lightcurve
        processed_cache_key = f"processed_{cache_key}"
        processed_cache_data = {
            "data": {
                "target_name": processed_lc.target_name,
                "time_bjd": processed_lc.time_bjd.tolist(),
                "flux": processed_lc.flux.tolist(),
                "flux_err": processed_lc.flux_err.tolist(),
                "mission": processed_lc.mission,
                "instrument": processed_lc.instrument,
                "cadence_minutes": processed_lc.cadence_minutes,
                "sectors_quarters": processed_lc.sectors_quarters,
                "data_quality_flags": processed_lc.data_quality_flags.tolist() if processed_lc.data_quality_flags is not None else None,
                "detrended": processed_lc.detrended,
                "normalized": processed_lc.normalized,
                "outliers_removed": processed_lc.outliers_removed
            },
            "processing_report": processing_report,
            "timestamp": cached_lc["timestamp"]
        }
        
        await storage_manager.cache_lightcurve(processed_cache_key, processed_cache_data)
        
        return {
            "status": "success",
            "message": f"Light curve preprocessing completed for {request.target_name}",
            "processing_report": processing_report,
            "processed_data_points": len(processed_lc.time_bjd)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Light curve preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Data Versioning Endpoints
@router.post("/version/create")
async def create_data_version(
    request: VersionCreateRequest,
    current_user: User = Depends(get_current_user),
    data_manager: DataManager = Depends(get_initialized_data_manager)
):
    """
    Create new data version
    
    Requires: Researcher role or higher
    """
    require_role(current_user, UserRole.RESEARCHER)
    
    logger.info(f"User {current_user.username} requested version creation: {request.version_name}")
    
    try:
        # Find files matching patterns
        from pathlib import Path
        
        base_path = Path(storage_manager.base_path)
        files_to_version = []
        
        for pattern in request.file_patterns:
            files_to_version.extend(base_path.rglob(pattern))
        
        # Create version
        result = await version_manager.create_version(
            version_name=request.version_name,
            description=request.description,
            files_to_version=files_to_version,
            metadata={
                "created_by": current_user.username,
                "file_patterns": request.file_patterns
            }
        )
        
        return {
            "status": "success",
            "message": f"Data version {request.version_name} created successfully",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Version creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/version/list")
async def list_data_versions(
    current_user: User = Depends(get_current_user),
    data_manager: DataManager = Depends(get_initialized_data_manager)
):
    """
    List all data versions
    
    Requires: User role or higher
    """
    require_role(current_user, UserRole.USER)
    
    try:
        versions = await version_manager.list_versions()
        
        return {
            "status": "success",
            "versions": versions,
            "total_versions": len(versions)
        }
        
    except Exception as e:
        logger.error(f"Version listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/version/{version_name}")
async def get_version_info(
    version_name: str,
    current_user: User = Depends(get_current_user),
    data_manager: DataManager = Depends(get_initialized_data_manager)
):
    """
    Get detailed information about specific version
    
    Requires: User role or higher
    """
    require_role(current_user, UserRole.USER)
    
    try:
        version_info = await version_manager.get_version_info(version_name)
        
        if not version_info:
            raise HTTPException(
                status_code=404,
                detail=f"Version {version_name} not found"
            )
        
        return {
            "status": "success",
            "version_info": version_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Version info retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Storage Management Endpoints
@router.get("/storage/stats")
async def get_storage_stats(
    current_user: User = Depends(get_current_user),
    data_manager: DataManager = Depends(get_initialized_data_manager)
):
    """
    Get storage statistics
    
    Requires: User role or higher
    """
    require_role(current_user, UserRole.USER)
    
    try:
        stats = await storage_manager.get_storage_stats()
        
        return {
            "status": "success",
            "storage_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Storage stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ingestion/status")
async def get_ingestion_status(
    current_user: User = Depends(get_current_user),
    data_manager: DataManager = Depends(get_initialized_data_manager)
):
    """
    Get status of all ingested data
    
    Requires: User role or higher
    """
    require_role(current_user, UserRole.USER)
    
    try:
        status = await data_manager.get_ingestion_status()
        
        return {
            "status": "success",
            "ingestion_status": status
        }
        
    except Exception as e:
        logger.error(f"Ingestion status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Admin Endpoints
@router.post("/admin/cleanup")
async def cleanup_old_cache(
    max_age_days: int = Query(7, description="Maximum age of cache files to keep"),
    current_user: User = Depends(get_current_user),
    data_manager: DataManager = Depends(get_initialized_data_manager)
):
    """
    Clean up old cache files
    
    Requires: Admin role
    """
    require_role(current_user, UserRole.ADMIN)
    
    logger.info(f"Admin {current_user.username} requested cache cleanup")
    
    try:
        cleaned_files = await storage_manager.cleanup_old_cache(max_age_days)
        
        return {
            "status": "success",
            "message": f"Cache cleanup completed",
            "cleaned_files": cleaned_files
        }
        
    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/backup")
async def create_data_backup(
    current_user: User = Depends(get_current_user),
    data_manager: DataManager = Depends(get_initialized_data_manager)
):
    """
    Create backup of all data
    
    Requires: Admin role
    """
    require_role(current_user, UserRole.ADMIN)
    
    logger.info(f"Admin {current_user.username} requested data backup")
    
    try:
        backup_path = await storage_manager.create_backup()
        
        return {
            "status": "success",
            "message": "Data backup created successfully",
            "backup_path": str(backup_path)
        }
        
    except Exception as e:
        logger.error(f"Data backup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
