"""
Model Registry and Versioning System for ExoplanetAI
Manages ML model artifacts, versions, and deployment lifecycle
"""

import asyncio
import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import hashlib
import pickle
import numpy as np

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from core.logging import get_logger
from core.cache import get_cache
from core.config import config

logger = get_logger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata and performance metrics"""
    name: str
    version: str
    created_at: datetime
    model_type: str  # "lightgbm", "xgboost", "ensemble", "cnn"
    training_data_version: str
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    artifact_path: str
    artifact_size_mb: float
    checksum: str
    is_active: bool = False
    deployment_time: Optional[datetime] = None
    rollback_count: int = 0
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.deployment_time:
            data['deployment_time'] = self.deployment_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetadata':
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('deployment_time'):
            data['deployment_time'] = datetime.fromisoformat(data['deployment_time'])
        return cls(**data)


@dataclass
class ModelPerformanceReport:
    """Model performance evaluation report"""
    model_version: str
    evaluation_time: datetime
    test_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    drift_score: float
    recommendation: str  # "deploy", "retrain", "rollback"
    notes: str = ""
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['evaluation_time'] = self.evaluation_time.isoformat()
        return data


class ModelRegistry:
    """
    Model registry for managing ML model lifecycle
    
    Features:
    - Model versioning and metadata tracking
    - Performance monitoring and drift detection
    - Automated deployment and rollback
    - A/B testing support
    - Model artifact storage and retrieval
    """
    
    def __init__(
        self,
        registry_dir: Path = Path("models"),
        max_versions_per_model: int = 10,
        performance_threshold: float = 0.85,
        drift_threshold: float = 0.1
    ):
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_versions = max_versions_per_model
        self.performance_threshold = performance_threshold
        self.drift_threshold = drift_threshold
        
        # Registry state
        self.models: Dict[str, List[ModelMetadata]] = {}
        self.active_models: Dict[str, ModelMetadata] = {}
        self.performance_history: List[ModelPerformanceReport] = []
        
        # Load existing registry
        self._load_registry()
        
        logger.info(f"ModelRegistry initialized with {len(self.models)} model families")
    
    def _load_registry(self):
        """Load registry metadata from disk"""
        registry_file = self.registry_dir / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                
                # Load models
                for model_name, versions_data in data.get("models", {}).items():
                    self.models[model_name] = [
                        ModelMetadata.from_dict(v) for v in versions_data
                    ]
                
                # Load active models
                for model_name, active_data in data.get("active_models", {}).items():
                    self.active_models[model_name] = ModelMetadata.from_dict(active_data)
                
                # Load performance history
                for perf_data in data.get("performance_history", []):
                    perf_data['evaluation_time'] = datetime.fromisoformat(perf_data['evaluation_time'])
                    self.performance_history.append(ModelPerformanceReport(**perf_data))
                
                logger.info(f"Loaded registry with {len(self.models)} model families")
                
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
    
    def _save_registry(self):
        """Save registry metadata to disk"""
        registry_file = self.registry_dir / "registry.json"
        try:
            data = {
                "models": {
                    name: [model.to_dict() for model in versions]
                    for name, versions in self.models.items()
                },
                "active_models": {
                    name: model.to_dict()
                    for name, model in self.active_models.items()
                },
                "performance_history": [
                    report.to_dict() for report in self.performance_history
                ],
                "last_updated": datetime.now().isoformat()
            }
            
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def register_model(
        self,
        name: str,
        model_artifact: Any,
        model_type: str,
        training_data_version: str,
        performance_metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        feature_names: List[str],
        version: Optional[str] = None
    ) -> ModelMetadata:
        """Register a new model version"""
        
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create artifact directory
        model_dir = self.registry_dir / name
        model_dir.mkdir(exist_ok=True)
        
        # Save model artifact
        artifact_path = model_dir / f"{name}_v{version}.pkl"
        try:
            if JOBLIB_AVAILABLE:
                joblib.dump(model_artifact, artifact_path)
            else:
                with open(artifact_path, 'wb') as f:
                    pickle.dump(model_artifact, f)
        except Exception as e:
            logger.error(f"Error saving model artifact: {e}")
            raise
        
        # Calculate artifact metadata
        artifact_size_mb = artifact_path.stat().st_size / (1024 * 1024)
        checksum = self._calculate_file_checksum(artifact_path)
        
        # Create metadata
        metadata = ModelMetadata(
            name=name,
            version=version,
            created_at=datetime.now(),
            model_type=model_type,
            training_data_version=training_data_version,
            performance_metrics=performance_metrics,
            hyperparameters=hyperparameters,
            feature_names=feature_names,
            artifact_path=str(artifact_path),
            artifact_size_mb=artifact_size_mb,
            checksum=checksum
        )
        
        # Add to registry
        if name not in self.models:
            self.models[name] = []
        
        self.models[name].append(metadata)
        
        # Sort by creation time (newest first)
        self.models[name].sort(key=lambda m: m.created_at, reverse=True)
        
        # Cleanup old versions
        self._cleanup_old_versions(name)
        
        # Save registry
        self._save_registry()
        
        logger.info(
            f"✅ Registered model {name} v{version} "
            f"(AUC: {performance_metrics.get('auc', 0):.3f})"
        )
        
        return metadata
    
    def deploy_model(self, name: str, version: str) -> bool:
        """Deploy a specific model version"""
        try:
            # Find the model
            model_metadata = self.get_model_metadata(name, version)
            if not model_metadata:
                logger.error(f"Model {name} v{version} not found")
                return False
            
            # Check performance threshold
            auc = model_metadata.performance_metrics.get('auc', 0)
            if auc < self.performance_threshold:
                logger.warning(
                    f"Model {name} v{version} below performance threshold "
                    f"(AUC: {auc:.3f} < {self.performance_threshold})"
                )
            
            # Deactivate current active model
            if name in self.active_models:
                self.active_models[name].is_active = False
            
            # Activate new model
            model_metadata.is_active = True
            model_metadata.deployment_time = datetime.now()
            self.active_models[name] = model_metadata
            
            # Save registry
            self._save_registry()
            
            logger.info(f"🚀 Deployed model {name} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying model {name} v{version}: {e}")
            return False
    
    def rollback_model(self, name: str) -> bool:
        """Rollback to previous model version"""
        try:
            if name not in self.models or len(self.models[name]) < 2:
                logger.error(f"No previous version available for rollback: {name}")
                return False
            
            # Find current and previous versions
            versions = self.models[name]
            current_active = self.active_models.get(name)
            
            # Find previous version (skip current active)
            previous_version = None
            for model in versions:
                if current_active and model.version != current_active.version:
                    previous_version = model
                    break
                elif not current_active:
                    previous_version = model
                    break
            
            if not previous_version:
                logger.error(f"No suitable previous version found for {name}")
                return False
            
            # Increment rollback count
            if current_active:
                current_active.rollback_count += 1
            
            # Deploy previous version
            success = self.deploy_model(name, previous_version.version)
            
            if success:
                logger.info(f"🔄 Rolled back {name} to v{previous_version.version}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error rolling back model {name}: {e}")
            return False
    
    def load_active_model(self, name: str) -> Optional[Any]:
        """Load the active model artifact"""
        if name not in self.active_models:
            logger.error(f"No active model found: {name}")
            return None
        
        metadata = self.active_models[name]
        return self._load_model_artifact(metadata.artifact_path)
    
    def _load_model_artifact(self, artifact_path: str) -> Optional[Any]:
        """Load model artifact from disk"""
        try:
            path = Path(artifact_path)
            if not path.exists():
                logger.error(f"Model artifact not found: {artifact_path}")
                return None
            
            if JOBLIB_AVAILABLE:
                return joblib.load(path)
            else:
                with open(path, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading model artifact {artifact_path}: {e}")
            return None
    
    def evaluate_model_performance(
        self, 
        name: str, 
        version: str,
        test_data: Tuple[np.ndarray, np.ndarray],
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> ModelPerformanceReport:
        """Evaluate model performance and detect drift"""
        
        model = self.load_model_by_version(name, version)
        if model is None:
            raise ValueError(f"Model {name} v{version} not found")
        
        X_test, y_test = test_data
        
        # Calculate test metrics
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                from sklearn.metrics import roc_auc_score, precision_score, recall_score
                test_metrics = {
                    'auc': float(roc_auc_score(y_test, y_pred_proba)),
                    'precision': float(precision_score(y_test, y_pred_proba > 0.5)),
                    'recall': float(recall_score(y_test, y_pred_proba > 0.5))
                }
            else:
                y_pred = model.predict(X_test)
                from sklearn.metrics import accuracy_score, precision_score, recall_score
                test_metrics = {
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'precision': float(precision_score(y_test, y_pred)),
                    'recall': float(recall_score(y_test, y_pred))
                }
        except Exception as e:
            logger.error(f"Error calculating test metrics: {e}")
            test_metrics = {}
        
        # Calculate validation metrics if provided
        validation_metrics = {}
        if validation_data:
            try:
                X_val, y_val = validation_data
                if hasattr(model, 'predict_proba'):
                    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
                    validation_metrics = {
                        'auc': float(roc_auc_score(y_val, y_val_pred_proba)),
                        'precision': float(precision_score(y_val, y_val_pred_proba > 0.5)),
                        'recall': float(recall_score(y_val, y_val_pred_proba > 0.5))
                    }
                else:
                    y_val_pred = model.predict(X_val)
                    validation_metrics = {
                        'accuracy': float(accuracy_score(y_val, y_val_pred)),
                        'precision': float(precision_score(y_val, y_val_pred)),
                        'recall': float(recall_score(y_val, y_val_pred))
                    }
            except Exception as e:
                logger.error(f"Error calculating validation metrics: {e}")
        
        # Calculate drift score (simplified)
        drift_score = self._calculate_drift_score(name, version, test_metrics)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(test_metrics, drift_score)
        
        # Create performance report
        report = ModelPerformanceReport(
            model_version=f"{name}_v{version}",
            evaluation_time=datetime.now(),
            test_metrics=test_metrics,
            validation_metrics=validation_metrics,
            drift_score=drift_score,
            recommendation=recommendation,
            notes=f"Evaluated on {len(X_test)} test samples"
        )
        
        # Add to history
        self.performance_history.append(report)
        
        # Keep only recent history (last 100 evaluations)
        self.performance_history = self.performance_history[-100:]
        
        # Save registry
        self._save_registry()
        
        logger.info(
            f"📊 Evaluated {name} v{version}: "
            f"AUC={test_metrics.get('auc', 0):.3f}, "
            f"drift={drift_score:.3f}, "
            f"recommendation={recommendation}"
        )
        
        return report
    
    def _calculate_drift_score(self, name: str, version: str, current_metrics: Dict[str, float]) -> float:
        """Calculate model drift score compared to historical performance"""
        try:
            # Get historical performance for this model
            historical_reports = [
                r for r in self.performance_history
                if r.model_version.startswith(f"{name}_v")
            ]
            
            if len(historical_reports) < 2:
                return 0.0  # No drift if insufficient history
            
            # Calculate average historical AUC
            historical_aucs = [r.test_metrics.get('auc', 0) for r in historical_reports[-5:]]
            avg_historical_auc = np.mean(historical_aucs)
            
            # Calculate drift as relative change
            current_auc = current_metrics.get('auc', 0)
            if avg_historical_auc > 0:
                drift_score = abs(current_auc - avg_historical_auc) / avg_historical_auc
            else:
                drift_score = 0.0
            
            return min(1.0, drift_score)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating drift score: {e}")
            return 0.0
    
    def _generate_recommendation(self, metrics: Dict[str, float], drift_score: float) -> str:
        """Generate deployment recommendation based on metrics"""
        auc = metrics.get('auc', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        
        # Check performance thresholds
        if auc < 0.7 or precision < 0.7 or recall < 0.7:
            return "retrain"
        
        # Check drift
        if drift_score > self.drift_threshold:
            return "retrain"
        
        # Check if significantly better than current
        if auc > self.performance_threshold and drift_score < 0.05:
            return "deploy"
        
        return "monitor"
    
    def load_model_by_version(self, name: str, version: str) -> Optional[Any]:
        """Load a specific model version"""
        metadata = self.get_model_metadata(name, version)
        if not metadata:
            return None
        
        return self._load_model_artifact(metadata.artifact_path)
    
    def get_model_metadata(self, name: str, version: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model version"""
        if name not in self.models:
            return None
        
        for model in self.models[name]:
            if model.version == version:
                return model
        
        return None
    
    def list_models(self, name: Optional[str] = None) -> Dict[str, List[Dict]]:
        """List all models or models for a specific name"""
        if name:
            if name in self.models:
                return {name: [model.to_dict() for model in self.models[name]]}
            else:
                return {}
        
        return {
            model_name: [model.to_dict() for model in versions]
            for model_name, versions in self.models.items()
        }
    
    def get_active_models(self) -> Dict[str, Dict]:
        """Get all active models"""
        return {
            name: model.to_dict()
            for name, model in self.active_models.items()
        }
    
    def get_performance_history(self, name: Optional[str] = None, days: int = 30) -> List[Dict]:
        """Get performance history"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        reports = [
            r for r in self.performance_history
            if r.evaluation_time > cutoff_date
        ]
        
        if name:
            reports = [r for r in reports if r.model_version.startswith(f"{name}_v")]
        
        return [report.to_dict() for report in reports]
    
    def _cleanup_old_versions(self, name: str):
        """Remove old model versions beyond the limit"""
        if name not in self.models:
            return
        
        versions = self.models[name]
        if len(versions) <= self.max_versions:
            return
        
        # Keep the most recent versions
        to_remove = versions[self.max_versions:]
        self.models[name] = versions[:self.max_versions]
        
        # Remove artifact files
        for model in to_remove:
            try:
                artifact_path = Path(model.artifact_path)
                if artifact_path.exists():
                    artifact_path.unlink()
                    logger.info(f"Removed old model artifact: {artifact_path}")
            except Exception as e:
                logger.error(f"Error removing old artifact {model.artifact_path}: {e}")
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()[:16]
    
    def get_registry_stats(self) -> Dict:
        """Get registry statistics"""
        total_models = sum(len(versions) for versions in self.models.values())
        total_size_mb = 0
        
        for versions in self.models.values():
            for model in versions:
                total_size_mb += model.artifact_size_mb
        
        return {
            "model_families": len(self.models),
            "total_versions": total_models,
            "active_models": len(self.active_models),
            "total_size_mb": round(total_size_mb, 2),
            "performance_evaluations": len(self.performance_history),
            "registry_dir": str(self.registry_dir)
        }


# Global registry instance
_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get global model registry instance"""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry
