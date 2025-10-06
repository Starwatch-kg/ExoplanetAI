"""
Automated Exoplanet Discovery Service
Continuously monitors TESS/Kepler/K2 data for new exoplanet candidates
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json

from data_sources.real_nasa_client import RealNASAClient
from ml.lightcurve_preprocessor import LightCurvePreprocessor
from ml.feature_extractor import ExoplanetFeatureExtractor
from ml.exoplanet_classifier import ExoplanetEnsembleClassifier
from core.cache import get_cache
from core.logging import get_logger
from services.data_ingest import get_ingest_service
from services.model_registry import get_model_registry

logger = get_logger(__name__)


@dataclass
class DiscoveryCandidate:
    """Represents a potential exoplanet candidate"""
    target_name: str
    tic_id: Optional[str] = None
    mission: str = "TESS"
    confidence: float = 0.0
    predicted_class: str = "Unknown"
    period: Optional[float] = None
    depth: Optional[float] = None
    snr: Optional[float] = None
    discovery_time: datetime = field(default_factory=datetime.now)
    lightcurve_path: Optional[str] = None
    features: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "target_name": self.target_name,
            "tic_id": self.tic_id,
            "mission": self.mission,
            "confidence": float(self.confidence),
            "predicted_class": self.predicted_class,
            "period": float(self.period) if self.period else None,
            "depth": float(self.depth) if self.depth else None,
            "snr": float(self.snr) if self.snr else None,
            "discovery_time": self.discovery_time.isoformat(),
            "lightcurve_path": self.lightcurve_path,
            "features": self.features
        }


class AutoDiscoveryService:
    """
    Automated exoplanet discovery pipeline
    
    Features:
    - Continuous monitoring of TESS/Kepler/K2 data
    - Automatic preprocessing and feature extraction
    - ML-based candidate detection
    - Result storage and reporting
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.85,
        check_interval_hours: int = 6,
        max_concurrent_tasks: int = 5,
        data_dir: Path = Path("data/auto_discovery")
    ):
        self.confidence_threshold = confidence_threshold
        self.check_interval = timedelta(hours=check_interval_hours)
        self.max_concurrent = max_concurrent_tasks
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Integrated services
        self.ingest_service = get_ingest_service()
        self.model_registry = get_model_registry()
        self.cache = get_cache()
        
        # Components (lazy initialization)
        self._nasa_client = None
        self._preprocessor = None
        self._feature_extractor = None
        self._classifier = None
        
        # State tracking
        self.last_check_time: Optional[datetime] = None
        self.processed_targets: set = set()
        self.candidates: List[DiscoveryCandidate] = []
        self.discovery_stats = {
            "total_processed": 0,
            "total_candidates": 0,
            "high_confidence_candidates": 0,
            "last_model_update": None
        }
        self.is_running = False
        
        logger.info(f"AutoDiscoveryService initialized with threshold={confidence_threshold}")
    
    @property
    def nasa_client(self):
        """Lazy load NASA client"""
        if self._nasa_client is None:
            self._nasa_client = RealNASAClient()
        return self._nasa_client
    
    @property
    def preprocessor(self):
        """Lazy load preprocessor"""
        if self._preprocessor is None:
            self._preprocessor = LightCurvePreprocessor()
        return self._preprocessor
    
    @property
    def feature_extractor(self):
        """Lazy load feature extractor"""
        if self._feature_extractor is None:
            self._feature_extractor = ExoplanetFeatureExtractor()
        return self._feature_extractor
    
    @property
    def classifier(self):
        """Lazy load classifier"""
        if self._classifier is None:
            self._classifier = ExoplanetEnsembleClassifier()
        return self._classifier
    
    def get_active_model(self):
        """Get the active exoplanet classification model"""
        model = self.model_registry.load_active_model("exoplanet_classifier")
        if model is None:
            # Fallback to lazy-loaded classifier
            if self._classifier is None:
                self._classifier = ExoplanetEnsembleClassifier()
            return self._classifier
        return model
    
    async def start(self):
        """Start the automated discovery pipeline"""
        self.is_running = True
        logger.info("🚀 Starting automated exoplanet discovery service...")
        
        while self.is_running:
            try:
                await self._discovery_cycle()
                await asyncio.sleep(self.check_interval.total_seconds())
            except Exception as e:
                logger.error(f"Error in discovery cycle: {e}", exc_info=True)
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def stop(self):
        """Stop the automated discovery pipeline"""
        self.is_running = False
        logger.info("🛑 Stopping automated discovery service...")
    
    async def _discovery_cycle(self):
        """Execute one complete discovery cycle"""
        cycle_start = datetime.now()
        logger.info(f"🔍 Starting discovery cycle at {cycle_start}")
        
        # Step 1: Check for new ingested data
        ingestion_stats = self.ingest_service.get_ingestion_stats()
        recent_count = ingestion_stats.get('recent_ingested', 0)
        total_count = ingestion_stats.get('total_ingested', 0)
        logger.info(f"📊 Ingestion stats: {recent_count} recent, {total_count} total")
        
        # Step 2: Get targets from ingested data
        new_targets = await self._get_targets_from_ingested_data()
        logger.info(f"📥 Found {len(new_targets)} new targets to analyze")
        
        if not new_targets:
            logger.info("No new targets found in this cycle")
            return
        
        # Step 3: Check model freshness and update if needed
        await self._check_and_update_model()
        
        # Step 4: Process targets in parallel batches
        candidates = await self._process_targets_batch(new_targets)
        
        # Step 5: Filter high-confidence candidates
        high_confidence = [c for c in candidates if c.confidence >= self.confidence_threshold]
        logger.info(f"✨ Found {len(high_confidence)} high-confidence candidates")
        
        # Step 6: Update statistics
        self.discovery_stats["total_processed"] += len(candidates)
        self.discovery_stats["total_candidates"] += len(candidates)
        self.discovery_stats["high_confidence_candidates"] += len(high_confidence)
        
        # Step 7: Save results and generate report
        await self._save_candidates(high_confidence)
        await self._generate_report(cycle_start, candidates, high_confidence)
        
        self.last_check_time = cycle_start
    
    async def _fetch_new_targets(self) -> List[Dict]:
        """Fetch new lightcurve targets from MAST/ExoFOP"""
        try:
            # Check cache for recently processed targets
            cache_key = f"auto_discovery:last_targets:{datetime.now().date()}"
            cached_targets = await self.cache.get(cache_key)
            
            if cached_targets:
                logger.info("Using cached target list")
                return cached_targets
            
            # Fetch from TESS Input Catalog (TIC)
            # In production, this would query MAST for new observations
            new_targets = await self._query_mast_for_new_data()
            
            # Cache the target list
            await self.cache.set(cache_key, new_targets, ttl=3600)
            
            return new_targets
            
        except Exception as e:
            logger.error(f"Error fetching new targets: {e}")
            return []
    
    async def _query_mast_for_new_data(self) -> List[Dict]:
        """Query MAST for new TESS observations"""
        # This is a placeholder - in production, use astroquery.mast
        # to fetch recent observations
        
        # For demo, return some test targets
        demo_targets = [
            {"tic_id": "441420236", "name": "TOI-715", "mission": "TESS"},
            {"tic_id": "307210830", "name": "TOI-700", "mission": "TESS"},
            {"tic_id": "460205581", "name": "TOI-1452", "mission": "TESS"},
        ]
        
        # Filter out already processed targets
        new_targets = [
            t for t in demo_targets 
            if t["tic_id"] not in self.processed_targets
        ]
        
        return new_targets
    
    async def _get_targets_from_ingested_data(self) -> List[Dict]:
        """Get targets from recently ingested data"""
        try:
            # Get recent ingestion records
            recent_records = self.ingest_service.get_recent_ingestions(hours=24)
            
            targets = []
            for record in recent_records:
                targets.append({
                    "tic_id": record.tic_id,
                    "name": record.target_name,
                    "mission": record.mission,
                    "file_path": record.file_path,
                    "quality_score": record.quality_score
                })
            
            return targets
            
        except Exception as e:
            logger.error(f"Error getting targets from ingested data: {e}")
            return []
    
    async def _load_lightcurve_from_files(self, tic_id: str, mission: str = "TESS") -> Optional[Dict]:
        """Load lightcurve data from ingested files"""
        try:
            import json
            import os
            from pathlib import Path
            
            # Look for files matching the TIC ID
            data_dir = Path("data/raw") / mission.lower()
            if not data_dir.exists():
                return None
            
            # Find files with this TIC ID
            pattern = f"*{tic_id}*.json"
            matching_files = list(data_dir.glob(pattern))
            
            if not matching_files:
                logger.debug(f"No files found for TIC {tic_id} in {data_dir}")
                return None
            
            # Use the most recent file
            latest_file = max(matching_files, key=os.path.getmtime)
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            logger.debug(f"Loaded lightcurve for TIC {tic_id} from {latest_file}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading lightcurve for TIC {tic_id}: {e}")
            return None
    
    async def _check_and_update_model(self):
        """Check if model needs updating and deploy new version if available"""
        try:
            # Get current active model info
            active_models = self.model_registry.get_active_models()
            current_model = active_models.get("exoplanet_classifier")
            
            # Get all available models
            all_models = self.model_registry.list_models("exoplanet_classifier")
            
            if not all_models.get("exoplanet_classifier"):
                logger.info("No models available in registry")
                return
            
            # Find the latest model
            latest_model = all_models["exoplanet_classifier"][0]  # Already sorted by creation time
            
            # Check if we need to update
            should_update = False
            if not current_model:
                logger.info("No active model, deploying latest")
                should_update = True
            elif latest_model["version"] != current_model["version"]:
                # Check if latest model is significantly better
                latest_auc = latest_model["performance_metrics"].get("auc", 0)
                current_auc = current_model["performance_metrics"].get("auc", 0)
                
                if latest_auc > current_auc + 0.02:  # 2% improvement threshold
                    logger.info(f"New model shows improvement: {latest_auc:.3f} vs {current_auc:.3f}")
                    should_update = True
            
            if should_update:
                success = self.model_registry.deploy_model("exoplanet_classifier", latest_model["version"])
                if success:
                    self.discovery_stats["last_model_update"] = datetime.now().isoformat()
                    logger.info(f"✅ Updated to model version {latest_model['version']}")
                    
                    # Clear classifier cache to force reload
                    self._classifier = None
            
        except Exception as e:
            logger.error(f"Error checking/updating model: {e}")
    
    async def _process_targets_batch(self, targets: List[Dict]) -> List[DiscoveryCandidate]:
        """Process multiple targets in parallel"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_semaphore(target):
            async with semaphore:
                return await self._process_single_target(target)
        
        tasks = [process_with_semaphore(t) for t in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors and None results
        candidates = [r for r in results if isinstance(r, DiscoveryCandidate)]
        return candidates
    
    async def _process_single_target(self, target: Dict) -> Optional[DiscoveryCandidate]:
        """Process a single target through the ML pipeline"""
        try:
            tic_id = target.get("tic_id")
            target_name = target.get("name", f"TIC-{tic_id}")
            mission = target.get("mission", "TESS")
            
            logger.info(f"🔬 Processing {target_name} (TIC {tic_id})")
            
            # Step 4: Load lightcurve data from ingested files
            lightcurve_data = await self._load_lightcurve_from_files(tic_id, mission)
            if not lightcurve_data:
                logger.warning(f"No lightcurve data for {target_name}")
                return None
            
            time_data = lightcurve_data["time"]
            flux_data = lightcurve_data["flux"]
            
            # Step 2: Validate data quality
            if not self._validate_lightcurve(time_data, flux_data):
                logger.warning(f"Invalid lightcurve for {target_name}")
                return None
            
            # Step 3: Preprocess
            processed = await self._preprocess_lightcurve(time_data, flux_data)
            if processed is None:
                return None
            
            time_clean, flux_clean = processed
            
            # Step 4: Extract features
            features = await self._extract_features(time_clean, flux_clean)
            
            # Step 5: ML prediction
            prediction = await self._predict_candidate(features, time_clean, flux_clean)
            
            # Step 6: Save lightcurve
            lc_path = await self._save_lightcurve(target_name, time_clean, flux_clean)
            
            # Create candidate object
            candidate = DiscoveryCandidate(
                target_name=target_name,
                tic_id=tic_id,
                mission=mission,
                confidence=prediction["confidence"],
                predicted_class=prediction["class"],
                period=prediction.get("period"),
                depth=prediction.get("depth"),
                snr=prediction.get("snr"),
                lightcurve_path=str(lc_path),
                features=features
            )
            
            # Mark as processed
            self.processed_targets.add(tic_id)
            
            logger.info(
                f"✅ {target_name}: {prediction['class']} "
                f"(confidence={prediction['confidence']:.2%})"
            )
            
            return candidate
            
        except Exception as e:
            logger.error(f"Error processing {target.get('name', 'unknown')}: {e}")
            return None
    
    async def _download_lightcurve(self, tic_id: str, mission: str) -> Optional[Dict]:
        """Download lightcurve from MAST"""
        try:
            # Use real NASA client
            result = await self.nasa_client.fetch_lightcurve(
                target_id=tic_id,
                mission=mission
            )
            
            if result and "time" in result and "flux" in result:
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error downloading lightcurve for TIC {tic_id}: {e}")
            return None
    
    def _validate_lightcurve(self, time: np.ndarray, flux: np.ndarray) -> bool:
        """Validate lightcurve data quality"""
        if len(time) < 100:
            return False
        
        if len(time) != len(flux):
            return False
        
        # Check for too many NaNs
        nan_fraction = np.isnan(flux).sum() / len(flux)
        if nan_fraction > 0.5:
            return False
        
        # Check time coverage
        time_span = np.nanmax(time) - np.nanmin(time)
        if time_span < 1.0:  # Less than 1 day
            return False
        
        return True
    
    async def _preprocess_lightcurve(
        self, 
        time: np.ndarray, 
        flux: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Preprocess lightcurve data"""
        try:
            # Remove NaNs
            mask = ~(np.isnan(time) | np.isnan(flux))
            time_clean = time[mask]
            flux_clean = flux[mask]
            
            # Normalize
            flux_normalized = self.preprocessor.normalize_flux(flux_clean)
            
            # Remove outliers
            flux_clean = self.preprocessor.remove_outliers(
                flux_normalized,
                sigma=5.0
            )
            
            # Detrend
            flux_detrended = self.preprocessor.detrend(
                time_clean,
                flux_clean,
                method="savgol"
            )
            
            return time_clean, flux_detrended
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None
    
    async def _extract_features(
        self, 
        time: np.ndarray, 
        flux: np.ndarray
    ) -> Dict:
        """Extract ML features from lightcurve"""
        try:
            features = self.feature_extractor.extract_all_features(time, flux)
            return features
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {}
    
    async def _predict_candidate(
        self, 
        features: Dict,
        time: np.ndarray,
        flux: np.ndarray
    ) -> Dict:
        """Predict if target is an exoplanet candidate"""
        try:
            # Use ensemble classifier
            prediction = self.classifier.predict(features, time, flux)
            
            return {
                "class": prediction.get("predicted_class", "Unknown"),
                "confidence": prediction.get("confidence_score", 0.0),
                "period": prediction.get("planet_parameters", {}).get("period"),
                "depth": prediction.get("planet_parameters", {}).get("depth"),
                "snr": prediction.get("planet_parameters", {}).get("snr")
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "class": "Error",
                "confidence": 0.0
            }
    
    async def _save_lightcurve(
        self, 
        target_name: str,
        time: np.ndarray,
        flux: np.ndarray
    ) -> Path:
        """Save lightcurve data to disk"""
        filename = f"{target_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        filepath = self.data_dir / "lightcurves" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            filepath,
            time=time,
            flux=flux,
            target_name=target_name
        )
        
        return filepath
    
    async def _save_candidates(self, candidates: List[DiscoveryCandidate]):
        """Save candidates to database/file"""
        if not candidates:
            return
        
        # Save to JSON file
        output_file = self.data_dir / "candidates" / f"candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(
                [c.to_dict() for c in candidates],
                f,
                indent=2
            )
        
        # Add to internal list
        self.candidates.extend(candidates)
        
        logger.info(f"💾 Saved {len(candidates)} candidates to {output_file}")
    
    async def _generate_report(
        self,
        cycle_start: datetime,
        all_candidates: List[DiscoveryCandidate],
        high_confidence: List[DiscoveryCandidate]
    ):
        """Generate discovery report"""
        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        
        report = {
            "cycle_start": cycle_start.isoformat(),
            "cycle_duration_seconds": cycle_duration,
            "total_processed": len(all_candidates),
            "high_confidence_count": len(high_confidence),
            "confidence_threshold": self.confidence_threshold,
            "candidates": [c.to_dict() for c in high_confidence]
        }
        
        # Save report
        report_file = self.data_dir / "reports" / f"report_{cycle_start.strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Discovery report saved to {report_file}")
        
        # Log summary
        if high_confidence:
            logger.info("🎯 High-confidence candidates:")
            for c in high_confidence:
                logger.info(
                    f"  - {c.target_name}: {c.predicted_class} "
                    f"({c.confidence:.2%}, period={c.period:.2f}d)"
                )
    
    async def get_recent_candidates(
        self,
        hours: int = 24,
        min_confidence: float = 0.0
    ) -> List[DiscoveryCandidate]:
        """Get candidates discovered in the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent = [
            c for c in self.candidates
            if c.discovery_time >= cutoff_time and c.confidence >= min_confidence
        ]
        
        return sorted(recent, key=lambda x: x.confidence, reverse=True)
    
    async def get_statistics(self) -> Dict:
        """Get discovery statistics"""
        total_candidates = len(self.candidates)
        high_confidence = len([c for c in self.candidates if c.confidence >= self.confidence_threshold])
        
        return {
            "total_processed": len(self.processed_targets),
            "total_candidates": total_candidates,
            "high_confidence_candidates": high_confidence,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "is_running": self.is_running,
            "confidence_threshold": self.confidence_threshold
        }
