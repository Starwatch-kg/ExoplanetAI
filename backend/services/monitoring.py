"""
Monitoring and Metrics Service for Automated Discovery
Tracks performance, errors, and generates dashboards
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryMetrics:
    """Metrics for discovery operations"""
    timestamp: datetime
    targets_processed: int = 0
    candidates_found: int = 0
    high_confidence_count: int = 0
    processing_time_seconds: float = 0.0
    errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class SystemHealth:
    """System health status"""
    status: str = "healthy"  # healthy, degraded, unhealthy
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    active_tasks: int = 0
    error_rate: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)


class MonitoringService:
    """
    Monitoring and metrics service for automated discovery
    
    Features:
    - Real-time metrics collection
    - Performance tracking
    - Error monitoring and alerting
    - Dashboard data generation
    - Historical trend analysis
    """
    
    def __init__(self, metrics_dir: Path = Path("data/metrics")):
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory metrics (last 1000 entries)
        self.metrics_history: deque = deque(maxlen=1000)
        
        # Aggregated statistics
        self.hourly_stats: Dict[str, Dict] = defaultdict(dict)
        self.daily_stats: Dict[str, Dict] = defaultdict(dict)
        
        # Error tracking
        self.errors: deque = deque(maxlen=100)
        
        # System health
        self.health = SystemHealth()
        
        logger.info("MonitoringService initialized")
    
    def record_discovery_cycle(
        self,
        targets_processed: int,
        candidates_found: int,
        high_confidence_count: int,
        processing_time: float,
        errors: int = 0
    ):
        """Record metrics from a discovery cycle"""
        metrics = DiscoveryMetrics(
            timestamp=datetime.now(),
            targets_processed=targets_processed,
            candidates_found=candidates_found,
            high_confidence_count=high_confidence_count,
            processing_time_seconds=processing_time,
            errors=errors
        )
        
        self.metrics_history.append(metrics)
        self._update_aggregated_stats(metrics)
        
        logger.info(
            f"📊 Metrics recorded: {targets_processed} targets, "
            f"{candidates_found} candidates, {processing_time:.2f}s"
        )
    
    def record_error(self, error_type: str, error_message: str, context: Dict = None):
        """Record an error for monitoring"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": error_message,
            "context": context or {}
        }
        
        self.errors.append(error_entry)
        logger.error(f"🚨 Error recorded: {error_type} - {error_message}")
    
    def record_cache_stats(self, hits: int, misses: int):
        """Record cache performance"""
        if self.metrics_history:
            latest = self.metrics_history[-1]
            latest.cache_hits = hits
            latest.cache_misses = misses
    
    def _update_aggregated_stats(self, metrics: DiscoveryMetrics):
        """Update hourly and daily aggregated statistics"""
        hour_key = metrics.timestamp.strftime("%Y-%m-%d %H:00")
        day_key = metrics.timestamp.strftime("%Y-%m-%d")
        
        # Update hourly stats
        if hour_key not in self.hourly_stats:
            self.hourly_stats[hour_key] = {
                "targets_processed": 0,
                "candidates_found": 0,
                "high_confidence": 0,
                "total_time": 0.0,
                "errors": 0,
                "cycles": 0
            }
        
        self.hourly_stats[hour_key]["targets_processed"] += metrics.targets_processed
        self.hourly_stats[hour_key]["candidates_found"] += metrics.candidates_found
        self.hourly_stats[hour_key]["high_confidence"] += metrics.high_confidence_count
        self.hourly_stats[hour_key]["total_time"] += metrics.processing_time_seconds
        self.hourly_stats[hour_key]["errors"] += metrics.errors
        self.hourly_stats[hour_key]["cycles"] += 1
        
        # Update daily stats
        if day_key not in self.daily_stats:
            self.daily_stats[day_key] = {
                "targets_processed": 0,
                "candidates_found": 0,
                "high_confidence": 0,
                "total_time": 0.0,
                "errors": 0,
                "cycles": 0
            }
        
        self.daily_stats[day_key]["targets_processed"] += metrics.targets_processed
        self.daily_stats[day_key]["candidates_found"] += metrics.candidates_found
        self.daily_stats[day_key]["high_confidence"] += metrics.high_confidence_count
        self.daily_stats[day_key]["total_time"] += metrics.processing_time_seconds
        self.daily_stats[day_key]["errors"] += metrics.errors
        self.daily_stats[day_key]["cycles"] += 1
    
    def get_realtime_metrics(self) -> Dict:
        """Get real-time metrics from the last hour"""
        one_hour_ago = datetime.now() - timedelta(hours=1)
        
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= one_hour_ago
        ]
        
        if not recent_metrics:
            return {
                "period": "last_hour",
                "targets_processed": 0,
                "candidates_found": 0,
                "high_confidence": 0,
                "avg_processing_time": 0.0,
                "error_rate": 0.0
            }
        
        total_targets = sum(m.targets_processed for m in recent_metrics)
        total_candidates = sum(m.candidates_found for m in recent_metrics)
        total_high_conf = sum(m.high_confidence_count for m in recent_metrics)
        avg_time = sum(m.processing_time_seconds for m in recent_metrics) / len(recent_metrics)
        total_errors = sum(m.errors for m in recent_metrics)
        error_rate = total_errors / len(recent_metrics) if recent_metrics else 0.0
        
        return {
            "period": "last_hour",
            "targets_processed": total_targets,
            "candidates_found": total_candidates,
            "high_confidence": total_high_conf,
            "avg_processing_time": round(avg_time, 2),
            "error_rate": round(error_rate, 2),
            "cycles": len(recent_metrics)
        }
    
    def get_hourly_trends(self, hours: int = 24) -> List[Dict]:
        """Get hourly trends for the last N hours"""
        trends = []
        now = datetime.now()
        
        for i in range(hours):
            hour_time = now - timedelta(hours=i)
            hour_key = hour_time.strftime("%Y-%m-%d %H:00")
            
            if hour_key in self.hourly_stats:
                stats = self.hourly_stats[hour_key]
                trends.append({
                    "hour": hour_key,
                    "targets_processed": stats["targets_processed"],
                    "candidates_found": stats["candidates_found"],
                    "high_confidence": stats["high_confidence"],
                    "avg_time": round(stats["total_time"] / stats["cycles"], 2) if stats["cycles"] > 0 else 0,
                    "errors": stats["errors"]
                })
            else:
                trends.append({
                    "hour": hour_key,
                    "targets_processed": 0,
                    "candidates_found": 0,
                    "high_confidence": 0,
                    "avg_time": 0,
                    "errors": 0
                })
        
        return sorted(trends, key=lambda x: x["hour"])
    
    def get_daily_trends(self, days: int = 7) -> List[Dict]:
        """Get daily trends for the last N days"""
        trends = []
        now = datetime.now()
        
        for i in range(days):
            day_time = now - timedelta(days=i)
            day_key = day_time.strftime("%Y-%m-%d")
            
            if day_key in self.daily_stats:
                stats = self.daily_stats[day_key]
                trends.append({
                    "date": day_key,
                    "targets_processed": stats["targets_processed"],
                    "candidates_found": stats["candidates_found"],
                    "high_confidence": stats["high_confidence"],
                    "avg_time": round(stats["total_time"] / stats["cycles"], 2) if stats["cycles"] > 0 else 0,
                    "errors": stats["errors"],
                    "cycles": stats["cycles"]
                })
            else:
                trends.append({
                    "date": day_key,
                    "targets_processed": 0,
                    "candidates_found": 0,
                    "high_confidence": 0,
                    "avg_time": 0,
                    "errors": 0,
                    "cycles": 0
                })
        
        return sorted(trends, key=lambda x: x["date"])
    
    def get_error_summary(self) -> Dict:
        """Get summary of recent errors"""
        if not self.errors:
            return {
                "total_errors": 0,
                "error_types": {},
                "recent_errors": []
            }
        
        # Count by type
        error_types = defaultdict(int)
        for error in self.errors:
            error_types[error["type"]] += 1
        
        # Get last 10 errors
        recent_errors = list(self.errors)[-10:]
        
        return {
            "total_errors": len(self.errors),
            "error_types": dict(error_types),
            "recent_errors": recent_errors
        }
    
    def update_system_health(self):
        """Update system health metrics"""
        try:
            import psutil
            
            # CPU and memory usage
            self.health.cpu_usage = psutil.cpu_percent(interval=1)
            self.health.memory_usage = psutil.virtual_memory().percent
            self.health.disk_usage = psutil.disk_usage('/').percent
            
            # Calculate error rate
            recent_metrics = list(self.metrics_history)[-10:] if self.metrics_history else []
            if recent_metrics:
                total_errors = sum(m.errors for m in recent_metrics)
                self.health.error_rate = total_errors / len(recent_metrics)
            
            # Determine health status
            if self.health.error_rate > 0.5 or self.health.cpu_usage > 90 or self.health.memory_usage > 90:
                self.health.status = "unhealthy"
            elif self.health.error_rate > 0.2 or self.health.cpu_usage > 70 or self.health.memory_usage > 70:
                self.health.status = "degraded"
            else:
                self.health.status = "healthy"
            
            self.health.last_check = datetime.now()
            
        except ImportError:
            logger.warning("psutil not installed, system health metrics unavailable")
    
    def get_system_health(self) -> Dict:
        """Get current system health"""
        self.update_system_health()
        
        return {
            "status": self.health.status,
            "cpu_usage": round(self.health.cpu_usage, 2),
            "memory_usage": round(self.health.memory_usage, 2),
            "disk_usage": round(self.health.disk_usage, 2),
            "active_tasks": self.health.active_tasks,
            "error_rate": round(self.health.error_rate, 2),
            "last_check": self.health.last_check.isoformat()
        }
    
    def get_dashboard_data(self) -> Dict:
        """Get comprehensive dashboard data"""
        return {
            "realtime": self.get_realtime_metrics(),
            "hourly_trends": self.get_hourly_trends(24),
            "daily_trends": self.get_daily_trends(7),
            "errors": self.get_error_summary(),
            "system_health": self.get_system_health(),
            "timestamp": datetime.now().isoformat()
        }
    
    def export_metrics(self, filepath: Optional[Path] = None) -> Path:
        """Export metrics to JSON file"""
        if filepath is None:
            filepath = self.metrics_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "export_time": datetime.now().isoformat(),
            "metrics_count": len(self.metrics_history),
            "hourly_stats": dict(self.hourly_stats),
            "daily_stats": dict(self.daily_stats),
            "errors": list(self.errors),
            "system_health": self.get_system_health()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"📁 Metrics exported to {filepath}")
        return filepath
    
    def generate_report(self) -> str:
        """Generate a text report of current metrics"""
        dashboard = self.get_dashboard_data()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║          EXOPLANET DISCOVERY MONITORING REPORT               ║
╚══════════════════════════════════════════════════════════════╝

📊 REAL-TIME METRICS (Last Hour)
  • Targets Processed: {dashboard['realtime']['targets_processed']}
  • Candidates Found: {dashboard['realtime']['candidates_found']}
  • High Confidence: {dashboard['realtime']['high_confidence']}
  • Avg Processing Time: {dashboard['realtime']['avg_processing_time']}s
  • Error Rate: {dashboard['realtime']['error_rate']}

💓 SYSTEM HEALTH
  • Status: {dashboard['system_health']['status'].upper()}
  • CPU Usage: {dashboard['system_health']['cpu_usage']}%
  • Memory Usage: {dashboard['system_health']['memory_usage']}%
  • Disk Usage: {dashboard['system_health']['disk_usage']}%
  • Error Rate: {dashboard['system_health']['error_rate']}

📈 DAILY TRENDS (Last 7 Days)
"""
        
        for day in dashboard['daily_trends'][-7:]:
            report += f"  {day['date']}: {day['targets_processed']} targets, {day['candidates_found']} candidates\n"
        
        report += f"""
🚨 ERROR SUMMARY
  • Total Errors: {dashboard['errors']['total_errors']}
  • Error Types: {', '.join(f"{k}: {v}" for k, v in dashboard['errors']['error_types'].items())}

Generated: {dashboard['timestamp']}
"""
        
        return report
