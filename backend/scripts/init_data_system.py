#!/usr/bin/env python3
"""
Data Management System Initialization Script
Скрипт инициализации системы управления данными

This script sets up the complete data management infrastructure for ExoplanetAI,
including directory structure, database tables, cache setup, and initial data ingestion.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.data_manager import DataManager
from ingest.storage import StorageManager
from ingest.validator import DataValidator
from ingest.versioning import VersionManager
from preprocessing.lightcurve_processor import LightCurveProcessor
from core.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_system_init.log')
    ]
)
logger = logging.getLogger(__name__)


class DataSystemInitializer:
    """Initialize complete data management system"""
    
    def __init__(self):
        self.settings = get_settings()
        self.data_manager = None
        self.storage_manager = None
        self.validator = None
        self.version_manager = None
        self.processor = None
        
    async def initialize_all(self) -> Dict[str, Any]:
        """Initialize all components of the data management system"""
        logger.info("Starting ExoplanetAI Data Management System initialization")
        
        results = {
            "status": "success",
            "components": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Step 1: Initialize core components
            logger.info("Step 1: Initializing core components")
            await self._initialize_components(results)
            
            # Step 2: Create directory structure
            logger.info("Step 2: Creating directory structure")
            await self._create_directories(results)
            
            # Step 3: Initialize database tables
            logger.info("Step 3: Initializing database")
            await self._initialize_database(results)
            
            # Step 4: Setup caching system
            logger.info("Step 4: Setting up cache system")
            await self._setup_cache(results)
            
            # Step 5: Initialize version control
            logger.info("Step 5: Initializing version control")
            await self._initialize_versioning(results)
            
            # Step 6: Validate system health
            logger.info("Step 6: Validating system health")
            await self._validate_system_health(results)
            
            # Step 7: Optional initial data ingestion
            logger.info("Step 7: Initial data ingestion (optional)")
            await self._initial_data_ingestion(results)
            
            logger.info("Data management system initialization completed successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            results["status"] = "failed"
            results["errors"].append(str(e))
            
        return results
    
    async def _initialize_components(self, results: Dict[str, Any]):
        """Initialize all data management components"""
        try:
            # Initialize DataManager
            self.data_manager = DataManager()
            success = await self.data_manager.initialize()
            results["components"]["data_manager"] = {
                "status": "success" if success else "failed",
                "initialized": success
            }
            
            # Initialize StorageManager
            self.storage_manager = StorageManager()
            success = await self.storage_manager.initialize()
            results["components"]["storage_manager"] = {
                "status": "success" if success else "failed",
                "initialized": success
            }
            
            # Initialize DataValidator
            self.validator = DataValidator()
            results["components"]["validator"] = {
                "status": "success",
                "initialized": True
            }
            
            # Initialize VersionManager
            self.version_manager = VersionManager()
            success = await self.version_manager.initialize()
            results["components"]["version_manager"] = {
                "status": "success" if success else "failed",
                "initialized": success
            }
            
            # Initialize LightCurveProcessor
            self.processor = LightCurveProcessor()
            results["components"]["lightcurve_processor"] = {
                "status": "success",
                "initialized": True
            }
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            results["errors"].append(f"Component initialization: {e}")
            raise
    
    async def _create_directories(self, results: Dict[str, Any]):
        """Create required directory structure"""
        try:
            base_path = Path(self.settings.data.data_path)
            
            directories = [
                "raw/nasa",
                "raw/mast", 
                "raw/exofop",
                "raw/kepler",
                "raw/tess",
                "processed/v1",
                "processed/v2",
                "processed/lightcurves",
                "lightcurves/tess",
                "lightcurves/kepler", 
                "lightcurves/k2",
                "metadata",
                "metadata/versions",
                "versions",
                "cache/tables",
                "cache/lightcurves",
                "checksums",
                "validation_logs",
                "backups",
                "logs"
            ]
            
            created_dirs = []
            for directory in directories:
                dir_path = base_path / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(dir_path))
            
            results["components"]["directories"] = {
                "status": "success",
                "created_count": len(created_dirs),
                "base_path": str(base_path),
                "directories": created_dirs
            }
            
            logger.info(f"Created {len(created_dirs)} directories")
            
        except Exception as e:
            logger.error(f"Directory creation failed: {e}")
            results["errors"].append(f"Directory creation: {e}")
            raise
    
    async def _initialize_database(self, results: Dict[str, Any]):
        """Initialize database tables for metadata storage"""
        try:
            # This would integrate with your existing database system
            # For now, we'll create file-based metadata storage
            
            metadata_path = Path(self.settings.data.data_path) / "metadata"
            
            # Create initial metadata files
            initial_files = {
                "system_info.json": {
                    "version": "2.0.0",
                    "initialized_at": "2024-01-01T00:00:00Z",
                    "components": ["data_manager", "storage", "validator", "versioning", "processor"]
                },
                "data_sources.json": {
                    "nasa_archive": {"status": "configured", "last_check": None},
                    "mast": {"status": "configured", "last_check": None},
                    "exofop": {"status": "configured", "last_check": None},
                    "lightkurve": {"status": "configured", "last_check": None}
                },
                "ingestion_log.json": {
                    "last_koi_ingestion": None,
                    "last_toi_ingestion": None,
                    "last_k2_ingestion": None,
                    "total_ingestions": 0
                }
            }
            
            import json
            for filename, content in initial_files.items():
                file_path = metadata_path / filename
                with open(file_path, 'w') as f:
                    json.dump(content, f, indent=2)
            
            results["components"]["database"] = {
                "status": "success",
                "type": "file_based",
                "metadata_files": len(initial_files)
            }
            
            logger.info("Database initialization completed")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            results["errors"].append(f"Database initialization: {e}")
            raise
    
    async def _setup_cache(self, results: Dict[str, Any]):
        """Setup caching system (Redis + file-based fallback)"""
        try:
            cache_status = {
                "redis": {"available": False, "error": None},
                "file_cache": {"available": True, "path": None}
            }
            
            # Test Redis connection
            try:
                if hasattr(self.settings, 'redis_url'):
                    import redis.asyncio as redis
                    redis_client = redis.from_url(self.settings.redis_url)
                    await redis_client.ping()
                    cache_status["redis"]["available"] = True
                    await redis_client.close()
                    logger.info("Redis cache available")
                else:
                    cache_status["redis"]["error"] = "Redis URL not configured"
            except Exception as e:
                cache_status["redis"]["error"] = str(e)
                logger.warning(f"Redis not available: {e}")
            
            # Setup file-based cache
            cache_path = Path(self.settings.data.data_path) / "cache"
            cache_status["file_cache"]["path"] = str(cache_path)
            
            results["components"]["cache"] = {
                "status": "success",
                "redis_available": cache_status["redis"]["available"],
                "file_cache_available": cache_status["file_cache"]["available"],
                "details": cache_status
            }
            
            logger.info("Cache system setup completed")
            
        except Exception as e:
            logger.error(f"Cache setup failed: {e}")
            results["errors"].append(f"Cache setup: {e}")
            raise
    
    async def _initialize_versioning(self, results: Dict[str, Any]):
        """Initialize Git-based version control system"""
        try:
            # Version manager should already be initialized
            if not self.version_manager:
                raise Exception("Version manager not initialized")
            
            # Create initial version if none exists
            versions = await self.version_manager.list_versions()
            
            if not versions:
                # Create initial empty version
                initial_files = []
                result = await self.version_manager.create_version(
                    version_name="v1.0.0",
                    description="Initial data management system version",
                    files_to_version=initial_files,
                    metadata={
                        "system_initialization": True,
                        "created_by": "init_script"
                    }
                )
                
                results["components"]["versioning"] = {
                    "status": "success",
                    "initial_version_created": True,
                    "version_info": result
                }
            else:
                results["components"]["versioning"] = {
                    "status": "success",
                    "existing_versions": len(versions),
                    "latest_version": versions[0]["version"] if versions else None
                }
            
            logger.info("Version control system initialized")
            
        except Exception as e:
            logger.error(f"Versioning initialization failed: {e}")
            results["errors"].append(f"Versioning initialization: {e}")
            raise
    
    async def _validate_system_health(self, results: Dict[str, Any]):
        """Validate that all system components are healthy"""
        try:
            health_checks = {}
            
            # Check data manager health
            if self.data_manager:
                status = await self.data_manager.get_ingestion_status()
                health_checks["data_manager"] = {
                    "status": "healthy",
                    "details": status
                }
            
            # Check storage health
            if self.storage_manager:
                stats = await self.storage_manager.get_storage_stats()
                health_checks["storage"] = {
                    "status": "healthy",
                    "details": stats
                }
            
            # Check version manager health
            if self.version_manager:
                version_stats = await self.version_manager.get_version_stats()
                health_checks["versioning"] = {
                    "status": "healthy",
                    "details": version_stats
                }
            
            # Overall health assessment
            all_healthy = all(check["status"] == "healthy" for check in health_checks.values())
            
            results["components"]["health_check"] = {
                "status": "success" if all_healthy else "warning",
                "overall_health": "healthy" if all_healthy else "degraded",
                "component_checks": health_checks
            }
            
            logger.info(f"System health validation completed - Overall: {'healthy' if all_healthy else 'degraded'}")
            
        except Exception as e:
            logger.error(f"Health validation failed: {e}")
            results["errors"].append(f"Health validation: {e}")
            raise
    
    async def _initial_data_ingestion(self, results: Dict[str, Any]):
        """Perform initial data ingestion (optional)"""
        try:
            # Check if initial ingestion is requested
            perform_ingestion = os.getenv("INIT_DATA_INGESTION", "false").lower() == "true"
            
            if not perform_ingestion:
                results["components"]["initial_ingestion"] = {
                    "status": "skipped",
                    "reason": "Not requested (set INIT_DATA_INGESTION=true to enable)"
                }
                logger.info("Initial data ingestion skipped")
                return
            
            logger.info("Starting initial data ingestion")
            
            # Ingest a small sample of each table type
            ingestion_results = {}
            
            try:
                # Ingest KOI table (limited)
                koi_result = await self.data_manager.ingest_koi_table(force_refresh=True)
                ingestion_results["koi"] = koi_result
                logger.info(f"KOI ingestion: {koi_result.get('status', 'unknown')}")
            except Exception as e:
                ingestion_results["koi"] = {"status": "error", "error": str(e)}
                logger.warning(f"KOI ingestion failed: {e}")
            
            try:
                # Ingest TOI table (limited)
                toi_result = await self.data_manager.ingest_toi_table(force_refresh=True)
                ingestion_results["toi"] = toi_result
                logger.info(f"TOI ingestion: {toi_result.get('status', 'unknown')}")
            except Exception as e:
                ingestion_results["toi"] = {"status": "error", "error": str(e)}
                logger.warning(f"TOI ingestion failed: {e}")
            
            # Try to ingest a sample light curve
            try:
                sample_target = "TIC 441420236"  # TOI-715
                lc_result = await self.data_manager.ingest_lightcurve(
                    target_name=sample_target,
                    mission="TESS",
                    force_refresh=True
                )
                ingestion_results["sample_lightcurve"] = lc_result
                logger.info(f"Sample light curve ingestion: {lc_result.get('status', 'unknown')}")
            except Exception as e:
                ingestion_results["sample_lightcurve"] = {"status": "error", "error": str(e)}
                logger.warning(f"Sample light curve ingestion failed: {e}")
            
            # Count successful ingestions
            successful_ingestions = sum(1 for result in ingestion_results.values() 
                                      if result.get("status") == "success")
            
            results["components"]["initial_ingestion"] = {
                "status": "completed",
                "successful_ingestions": successful_ingestions,
                "total_attempted": len(ingestion_results),
                "results": ingestion_results
            }
            
            logger.info(f"Initial data ingestion completed: {successful_ingestions}/{len(ingestion_results)} successful")
            
        except Exception as e:
            logger.error(f"Initial data ingestion failed: {e}")
            results["errors"].append(f"Initial data ingestion: {e}")
            # Don't raise - this is optional
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.data_manager:
                await self.data_manager.cleanup()
            if self.storage_manager:
                await self.storage_manager.cleanup()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


async def main():
    """Main initialization function"""
    print("ExoplanetAI Data Management System Initialization")
    print("=" * 50)
    
    initializer = DataSystemInitializer()
    
    try:
        # Run initialization
        results = await initializer.initialize_all()
        
        # Print results
        print(f"\nInitialization Status: {results['status'].upper()}")
        print(f"Components initialized: {len(results['components'])}")
        
        if results['errors']:
            print(f"Errors encountered: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  - {error}")
        
        if results['warnings']:
            print(f"Warnings: {len(results['warnings'])}")
            for warning in results['warnings']:
                print(f"  - {warning}")
        
        # Component status summary
        print("\nComponent Status:")
        for component, info in results['components'].items():
            status = info.get('status', 'unknown')
            print(f"  {component}: {status.upper()}")
        
        # Save detailed results
        import json
        results_file = "data_system_init_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {results_file}")
        
        if results['status'] == 'success':
            print("\n✅ Data management system initialization completed successfully!")
            print("\nNext steps:")
            print("1. Start the ExoplanetAI backend server")
            print("2. Use the API endpoints to ingest and process data")
            print("3. Monitor system health through the admin interface")
        else:
            print("\n❌ Initialization completed with errors. Check the logs for details.")
            return 1
        
    except KeyboardInterrupt:
        print("\n\nInitialization interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error during initialization: {e}")
        logger.error(f"Fatal initialization error: {e}")
        return 1
    finally:
        await initializer.cleanup()
    
    return 0


if __name__ == "__main__":
    # Set environment variables if needed
    if not os.getenv("DATA_PATH"):
        os.environ["DATA_PATH"] = "/tmp/exoplanetai_data"
        print(f"Using default data path: {os.environ['DATA_PATH']}")
    
    # Run initialization
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
