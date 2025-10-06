"""
Automated Data Ingestion Service for ExoplanetAI
Continuously monitors and ingests new lightcurve data from MAST/NASA/ExoFOP
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set, Any
from pathlib import Path
import json
import hashlib
import numpy as np
from dataclasses import dataclass, asdict

try:
    import lightkurve as lk
    from astroquery.mast import Observations
    ASTRO_AVAILABLE = True
except ImportError:
    ASTRO_AVAILABLE = False

from core.cache import get_cache
from core.logging import get_logger
from core.config import config

logger = get_logger(__name__)


@dataclass
class DataIngestionRecord:
    """Record of ingested data"""
    target_name: str
    tic_id: Optional[str]
    mission: str
    sector: Optional[int]
    download_time: datetime
    file_path: str
    data_points: int
    time_span_days: float
    quality_score: float
    checksum: str
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'download_time': self.download_time.isoformat()
        }


class DataIngestService:
    """
    Automated data ingestion service
    
    Features:
    - Monitors MAST for new TESS/Kepler observations
    - Downloads and validates lightcurves
    - Stores data with metadata and checksums
    - Tracks ingestion history
    - Handles incremental updates
    """
    
    def __init__(
        self,
        data_dir: Path = Path("data/raw"),
        check_interval_hours: int = 6,
        max_concurrent_downloads: int = 3,
        min_data_points: int = 100,
        min_time_span_days: float = 10.0
    ):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.check_interval = timedelta(hours=check_interval_hours)
        self.max_concurrent = max_concurrent_downloads
        self.min_data_points = min_data_points
        self.min_time_span_days = min_time_span_days
        
        # State tracking
        self.ingestion_history: List[DataIngestionRecord] = []
        self.processed_targets: Set[str] = set()
        self.is_running = False
        self.last_check_time: Optional[datetime] = None
        
        # Load existing history
        self._load_ingestion_history()
        
        logger.info(f"DataIngestService initialized, data_dir={data_dir}")
    
    def _load_ingestion_history(self):
        """Load ingestion history from disk"""
        history_file = self.data_dir / "ingestion_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for record_data in data:
                        record_data['download_time'] = datetime.fromisoformat(record_data['download_time'])
                        record = DataIngestionRecord(**record_data)
                        self.ingestion_history.append(record)
                        if record.tic_id:
                            self.processed_targets.add(record.tic_id)
                
                logger.info(f"Loaded {len(self.ingestion_history)} ingestion records")
            except Exception as e:
                logger.error(f"Error loading ingestion history: {e}")
    
    def _save_ingestion_history(self):
        """Save ingestion history to disk"""
        history_file = self.data_dir / "ingestion_history.json"
        try:
            data = [record.to_dict() for record in self.ingestion_history]
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving ingestion history: {e}")
    
    async def start_continuous_ingestion(self):
        """Start continuous data ingestion"""
        self.is_running = True
        logger.info("🚀 Starting continuous data ingestion...")
        
        while self.is_running:
            try:
                await self._ingestion_cycle()
                await asyncio.sleep(self.check_interval.total_seconds())
            except Exception as e:
                logger.error(f"Error in ingestion cycle: {e}", exc_info=True)
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def stop_ingestion(self):
        """Stop continuous ingestion"""
        self.is_running = False
        logger.info("🛑 Stopping data ingestion...")
    
    async def _ingestion_cycle(self):
        """Execute one ingestion cycle"""
        cycle_start = datetime.now()
        logger.info(f"📥 Starting ingestion cycle at {cycle_start}")
        
        # Step 1: Query for new observations
        new_observations = await self._query_new_observations()
        logger.info(f"Found {len(new_observations)} new observations")
        
        if not new_observations:
            logger.info("No new observations found")
            return
        
        # Step 2: Download and process in batches
        ingested_count = await self._process_observations_batch(new_observations)
        
        # Step 3: Update history
        self._save_ingestion_history()
        
        logger.info(f"✅ Ingestion cycle completed: {ingested_count} new datasets")
        self.last_check_time = cycle_start
    
    async def _query_new_observations(self) -> List[Dict]:
        """Query MAST for new observations"""
        if not ASTRO_AVAILABLE:
            logger.warning("Astroquery not available, using demo data")
            return await self._get_demo_observations()
        
        try:
            # Query recent TESS observations
            cutoff_date = datetime.now() - timedelta(days=30)
            
            # This would be the real MAST query in production
            # obs_table = Observations.query_criteria(
            #     obs_collection="TESS",
            #     dataproduct_type="timeseries",
            #     t_min=cutoff_date.timestamp()
            # )
            
            # For now, use demo data
            return await self._get_demo_observations()
            
        except Exception as e:
            logger.error(f"Error querying MAST: {e}")
            return []
    
    async def _get_demo_observations(self) -> List[Dict]:
        """Get demo observations for testing"""
        demo_targets = [
            {
                "target_name": "TOI-715",
                "tic_id": "441420236",
                "mission": "TESS",
                "sector": 26,
                "priority": "high"
            },
            {
                "target_name": "TOI-700",
                "tic_id": "307210830", 
                "mission": "TESS",
                "sector": 15,
                "priority": "high"
            },
            {
                "target_name": "TOI-1452",
                "tic_id": "460205581",
                "mission": "TESS", 
                "sector": 28,
                "priority": "medium"
            },
            {
                "target_name": "TOI-849",
                "tic_id": "139270665",
                "mission": "TESS",
                "sector": 25,
                "priority": "medium"
            }
        ]
        
        # Filter out already processed
        new_targets = [
            t for t in demo_targets
            if t["tic_id"] not in self.processed_targets
        ]
        
        return new_targets
    
    async def _process_observations_batch(self, observations: List[Dict]) -> int:
        """Process observations in parallel batches"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_semaphore(obs):
            async with semaphore:
                return await self._ingest_single_observation(obs)
        
        tasks = [process_with_semaphore(obs) for obs in observations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful ingestions
        successful = sum(1 for r in results if isinstance(r, DataIngestionRecord))
        return successful
    
    async def _ingest_single_observation(self, observation: Dict) -> Optional[DataIngestionRecord]:
        """Ingest a single observation"""
        try:
            target_name = observation["target_name"]
            tic_id = observation["tic_id"]
            mission = observation["mission"]
            sector = observation.get("sector")
            
            logger.info(f"📥 Ingesting {target_name} (TIC {tic_id})")
            
            # Step 1: Download lightcurve
            lightcurve_data = await self._download_lightcurve(tic_id, mission, sector)
            if not lightcurve_data:
                logger.warning(f"Failed to download {target_name}")
                return None
            
            time_data = lightcurve_data["time"]
            flux_data = lightcurve_data["flux"]
            
            # Step 2: Validate data quality
            quality_score = self._assess_data_quality(time_data, flux_data)
            if quality_score < 0.3:  # Minimum quality threshold
                logger.warning(f"Low quality data for {target_name}: {quality_score:.2f}")
                return None
            
            # Step 3: Calculate metadata
            data_points = len(time_data)
            time_span_days = float(np.max(time_data) - np.min(time_data))
            
            if data_points < self.min_data_points:
                logger.warning(f"Insufficient data points for {target_name}: {data_points}")
                return None
            
            if time_span_days < self.min_time_span_days:
                logger.warning(f"Insufficient time span for {target_name}: {time_span_days:.1f} days")
                return None
            
            # Step 4: Save to disk
            file_path = await self._save_lightcurve_data(
                target_name, tic_id, mission, sector, time_data, flux_data
            )
            
            # Step 5: Calculate checksum
            checksum = self._calculate_checksum(time_data, flux_data)
            
            # Step 6: Create ingestion record
            record = DataIngestionRecord(
                target_name=target_name,
                tic_id=tic_id,
                mission=mission,
                sector=sector,
                download_time=datetime.now(),
                file_path=str(file_path),
                data_points=data_points,
                time_span_days=time_span_days,
                quality_score=quality_score,
                checksum=checksum
            )
            
            # Step 7: Update tracking
            self.ingestion_history.append(record)
            self.processed_targets.add(tic_id)
            
            logger.info(
                f"✅ {target_name}: {data_points} points, "
                f"{time_span_days:.1f} days, quality={quality_score:.2f}"
            )
            
            return record
            
        except Exception as e:
            logger.error(f"Error ingesting {observation.get('target_name', 'unknown')}: {e}")
            return None
    
    async def _download_lightcurve(
        self, tic_id: str, mission: str, sector: Optional[int] = None
    ) -> Optional[Dict]:
        """Download lightcurve data"""
        cache_key = f"lightcurve:{mission}:{tic_id}:{sector}"
        cache = get_cache()
        
        # Check cache first
        try:
            cached_data = await cache.get(cache_key)
        except Exception as e:
            logger.debug(f"Cache error: {e}")
            cached_data = None
        if cached_data:
            logger.debug(f"Using cached lightcurve for TIC {tic_id}")
            return cached_data
        
        try:
            if ASTRO_AVAILABLE and mission == "TESS":
                # Real download using lightkurve
                search_result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
                if len(search_result) == 0:
                    return None
                
                # Download the first available lightcurve
                lc = search_result[0].download()
                if lc is None:
                    return None
                
                # Clean and extract data
                lc = lc.remove_nans().remove_outliers()
                
                # Convert MaskedNDArray to regular numpy arrays, then to lists for JSON serialization
                def safe_extract(attr):
                    """Safely extract data from lightkurve attributes, handling MaskedNDArray"""
                    if hasattr(lc, attr):
                        data = getattr(lc, attr).value
                        logger.debug(f"Extracting {attr}: type={type(data)}, hasattr tolist={hasattr(data, 'tolist')}")
                        
                        # Convert MaskedNDArray to regular array
                        if hasattr(data, 'filled'):
                            data = data.filled(fill_value=0.0)
                            logger.debug(f"After filled: type={type(data)}")
                        
                        # Convert to list for JSON serialization
                        if hasattr(data, 'tolist') and not isinstance(data, list):
                            result = data.tolist()
                            logger.debug(f"Converted to list: type={type(result)}")
                            return result
                        elif isinstance(data, list):
                            logger.debug(f"Already a list")
                            return data
                        else:
                            result = float(data) if hasattr(data, '__float__') else data
                            logger.debug(f"Converted to scalar: type={type(result)}")
                            return result
                    return None
                
                lightcurve_data = {
                    "time": safe_extract('time'),
                    "flux": safe_extract('flux'),
                    "flux_err": safe_extract('flux_err'),
                    "quality": safe_extract('quality'),
                    "mission": mission,
                    "sector": getattr(lc, 'sector', sector)
                }
            else:
                # Generate demo data
                lightcurve_data = self._generate_demo_lightcurve(tic_id, mission)
            
            # Cache the result
            try:
                await cache.set(cache_key, lightcurve_data, ttl=86400)  # 24 hours
            except Exception as e:
                logger.debug(f"Cache set error: {e}")
            
            return lightcurve_data
            
        except Exception as e:
            logger.error(f"Error downloading lightcurve for TIC {tic_id}: {e}")
            return None
    
    def _generate_demo_lightcurve(self, tic_id: str, mission: str) -> Dict:
        """Generate demo lightcurve data"""
        np.random.seed(int(tic_id) % 2**32)  # Deterministic based on TIC ID
        
        # Generate time series (27.4 days for TESS sector)
        n_points = 1000
        time = np.linspace(0, 27.4, n_points)
        
        # Base flux with noise
        flux = np.ones(n_points) + np.random.normal(0, 0.001, n_points)
        
        # Add transit signal for some targets
        if int(tic_id) % 3 == 0:  # 1/3 of targets have transits
            period = 5.0 + np.random.uniform(0, 20)  # 5-25 day period
            depth = 0.005 + np.random.uniform(0, 0.015)  # 0.5-2% depth
            duration = 0.1 + np.random.uniform(0, 0.2)  # 0.1-0.3 day duration
            
            # Add transit events
            phase = (time % period) / period
            transit_mask = np.abs(phase - 0.5) < (duration / period / 2)
            flux[transit_mask] -= depth
        
        return {
            "time": time,
            "flux": flux,
            "flux_err": np.full(n_points, 0.001),
            "quality": np.zeros(n_points, dtype=int),
            "mission": mission,
            "sector": np.random.randint(1, 50)
        }
    
    def _assess_data_quality(self, time: np.ndarray, flux: np.ndarray) -> float:
        """Assess lightcurve data quality (0-1 score)"""
        try:
            # Check for basic requirements
            if len(time) < 50 or len(flux) < 50:
                return 0.0
            
            # Check for NaN/inf values
            if np.any(~np.isfinite(time)) or np.any(~np.isfinite(flux)):
                return 0.0
            
            # Calculate quality metrics
            time_coverage = (np.max(time) - np.min(time)) / 27.4  # Normalized to TESS sector
            flux_std = np.std(flux)
            flux_median = np.median(flux)
            
            # Relative noise level
            noise_level = flux_std / flux_median if flux_median > 0 else 1.0
            
            # Data completeness (gaps)
            time_diffs = np.diff(time)
            median_cadence = np.median(time_diffs)
            large_gaps = np.sum(time_diffs > 5 * median_cadence)
            completeness = 1.0 - (large_gaps / len(time_diffs))
            
            # Combined quality score
            quality_score = (
                min(time_coverage, 1.0) * 0.3 +  # Time coverage
                min(1.0 / (1.0 + noise_level * 100), 1.0) * 0.4 +  # Noise level
                completeness * 0.3  # Data completeness
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return 0.0
    
    async def _save_lightcurve_data(
        self, target_name: str, tic_id: str, mission: str, 
        sector: Optional[int], time: np.ndarray, flux: np.ndarray
    ) -> Path:
        """Save lightcurve data to disk"""
        # Create directory structure
        mission_dir = self.data_dir / mission.lower()
        mission_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{target_name}_{tic_id}_s{sector}_{timestamp}.json"
        file_path = mission_dir / filename
        
        # Prepare data with safe conversion to lists
        def safe_to_list(data):
            """Safely convert data to list for JSON serialization"""
            if isinstance(data, list):
                return data
            elif hasattr(data, 'tolist'):
                return data.tolist()
            else:
                return list(data)
        
        data = {
            "target_name": target_name,
            "tic_id": tic_id,
            "mission": mission,
            "sector": sector,
            "download_time": datetime.now().isoformat(),
            "time": safe_to_list(time),
            "flux": safe_to_list(flux),
            "data_points": len(time),
            "time_span_days": float(np.max(time) - np.min(time))
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Saved lightcurve data to {file_path}")
        return file_path
    
    def _calculate_checksum(self, time, flux) -> str:
        """Calculate SHA256 checksum of lightcurve data"""
        # Convert to numpy arrays if they're lists
        if isinstance(time, list):
            time = np.array(time)
        if isinstance(flux, list):
            flux = np.array(flux)
        
        data_str = f"{time.tobytes()}{flux.tobytes()}"
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics"""
        total_ingested = len(self.ingestion_history)
        recent_ingested = len([
            r for r in self.ingestion_history
            if (datetime.now() - r.download_time).total_seconds() < 86400  # 24 hours
        ])
        
        if self.ingestion_history:
            avg_quality = sum(float(r.quality_score) for r in self.ingestion_history) / len(self.ingestion_history)
            last_ingestion = max(self.ingestion_history, key=lambda x: x.download_time).download_time
        else:
            avg_quality = 0.0
            last_ingestion = None
        
        return {
            "total_ingested": total_ingested,
            "recent_ingested": recent_ingested,
            "average_quality": avg_quality,
            "last_ingestion": last_ingestion.isoformat() if last_ingestion else None,
            "is_running": self.is_running
        }
    
    def get_recent_ingestions(self, hours: int = 24) -> List[DataIngestionRecord]:
        """Get recent ingestion records"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            r for r in self.ingestion_history
            if r.download_time > cutoff_time
        ]


# Global service instance
_ingest_service: Optional[DataIngestService] = None


def get_ingest_service() -> DataIngestService:
    """Get global ingest service instance"""
    global _ingest_service
    if _ingest_service is None:
        _ingest_service = DataIngestService()
    return _ingest_service
