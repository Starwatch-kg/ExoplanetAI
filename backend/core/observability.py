"""
Enterprise-grade observability system for ExoplanetAI v2.0
Система наблюдаемости уровня enterprise для ExoplanetAI v2.0
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
import threading

from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
from prometheus_client.core import REGISTRY

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    INFO = "info"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Alert:
    """Alert data structure"""
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    check_function: Callable
    interval_seconds: int = 30
    timeout_seconds: int = 10
    critical: bool = False
    last_check: Optional[datetime] = None
    last_result: Optional[bool] = None
    last_error: Optional[str] = None


class MetricsCollector:
    """Enterprise metrics collection system"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self._setup_default_metrics()
    
    def _setup_default_metrics(self):
        """Setup default application metrics"""
        # Request metrics
        self.metrics['http_requests_total'] = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.metrics['http_request_duration_seconds'] = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # System metrics
        self.metrics['system_cpu_usage_percent'] = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.metrics['system_memory_usage_bytes'] = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            registry=self.registry
        )
        
        self.metrics['system_disk_usage_bytes'] = Gauge(
            'system_disk_usage_bytes',
            'System disk usage in bytes',
            ['path'],
            registry=self.registry
        )
        
        # Application metrics
        self.metrics['active_connections'] = Gauge(
            'active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.metrics['cache_hits_total'] = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.metrics['cache_misses_total'] = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        self.metrics['data_ingestion_records_total'] = Counter(
            'data_ingestion_records_total',
            'Total records ingested',
            ['source', 'table_type'],
            registry=self.registry
        )
        
        self.metrics['data_ingestion_duration_seconds'] = Histogram(
            'data_ingestion_duration_seconds',
            'Data ingestion duration in seconds',
            ['source', 'table_type'],
            registry=self.registry
        )
        
        # Rate limiting metrics
        self.metrics['rate_limit_violations_total'] = Counter(
            'rate_limit_violations_total',
            'Total rate limit violations',
            ['user_role', 'endpoint'],
            registry=self.registry
        )
        
        # Error metrics
        self.metrics['errors_total'] = Counter(
            'errors_total',
            'Total errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Business metrics
        self.metrics['lightcurve_analyses_total'] = Counter(
            'lightcurve_analyses_total',
            'Total lightcurve analyses performed',
            ['mission', 'analysis_type'],
            registry=self.registry
        )
        
        self.metrics['planet_discoveries_total'] = Counter(
            'planet_discoveries_total',
            'Total planet discoveries processed',
            ['source', 'status'],
            registry=self.registry
        )
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        self.metrics['http_requests_total'].labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()
        
        self.metrics['http_request_duration_seconds'].labels(
            method=method, endpoint=endpoint
        ).observe(duration)
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit"""
        self.metrics['cache_hits_total'].labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss"""
        self.metrics['cache_misses_total'].labels(cache_type=cache_type).inc()
    
    def record_data_ingestion(self, source: str, table_type: str, records: int, duration: float):
        """Record data ingestion metrics"""
        self.metrics['data_ingestion_records_total'].labels(
            source=source, table_type=table_type
        ).inc(records)
        
        self.metrics['data_ingestion_duration_seconds'].labels(
            source=source, table_type=table_type
        ).observe(duration)
    
    def record_rate_limit_violation(self, user_role: str, endpoint: str):
        """Record rate limit violation"""
        self.metrics['rate_limit_violations_total'].labels(
            user_role=user_role, endpoint=endpoint
        ).inc()
    
    def record_error(self, error_type: str, component: str):
        """Record error occurrence"""
        self.metrics['errors_total'].labels(
            error_type=error_type, component=component
        ).inc()
    
    def record_lightcurve_analysis(self, mission: str, analysis_type: str):
        """Record lightcurve analysis"""
        self.metrics['lightcurve_analyses_total'].labels(
            mission=mission, analysis_type=analysis_type
        ).inc()
    
    def update_system_metrics(self):
        """Update system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['system_cpu_usage_percent'].set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics['system_memory_usage_bytes'].set(memory.used)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics['system_disk_usage_bytes'].labels(path='/').set(disk.used)
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format"""
        return generate_latest(self.registry).decode('utf-8')


class AlertManager:
    """Alert management system"""
    
    def __init__(self, max_alerts: int = 1000):
        self.alerts: deque = deque(maxlen=max_alerts)
        self.alert_rules: Dict[str, Callable] = {}
        self.notification_channels: List[Callable] = []
        self._lock = threading.Lock()
    
    def add_alert_rule(self, name: str, rule_function: Callable):
        """Add alert rule"""
        self.alert_rules[name] = rule_function
    
    def add_notification_channel(self, channel: Callable):
        """Add notification channel"""
        self.notification_channels.append(channel)
    
    def create_alert(
        self, 
        name: str, 
        severity: AlertSeverity, 
        message: str, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create new alert"""
        alert = Alert(
            name=name,
            severity=severity,
            message=message,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self.alerts.append(alert)
        
        # Send notifications
        self._send_notifications(alert)
        
        logger.warning(f"Alert created: {name} ({severity.value}) - {message}")
    
    def resolve_alert(self, alert_name: str):
        """Resolve alert by name"""
        with self._lock:
            for alert in reversed(self.alerts):
                if alert.name == alert_name and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    logger.info(f"Alert resolved: {alert_name}")
                    break
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        with self._lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity"""
        with self._lock:
            return [alert for alert in self.alerts if alert.severity == severity]
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                logger.error(f"Failed to send alert notification: {e}")
    
    def check_alert_rules(self, metrics_data: Dict[str, Any]):
        """Check all alert rules against current metrics"""
        for rule_name, rule_function in self.alert_rules.items():
            try:
                rule_function(metrics_data, self)
            except Exception as e:
                logger.error(f"Alert rule {rule_name} failed: {e}")


class HealthCheckManager:
    """Health check management system"""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.overall_health: bool = True
        self.last_health_update: datetime = datetime.utcnow()
        self._lock = threading.Lock()
    
    def add_health_check(self, health_check: HealthCheck):
        """Add health check"""
        with self._lock:
            self.health_checks[health_check.name] = health_check
    
    async def run_health_check(self, name: str) -> bool:
        """Run specific health check"""
        if name not in self.health_checks:
            return False
        
        health_check = self.health_checks[name]
        
        try:
            start_time = time.time()
            
            # Run check with timeout
            result = await asyncio.wait_for(
                health_check.check_function(),
                timeout=health_check.timeout_seconds
            )
            
            duration = time.time() - start_time
            
            with self._lock:
                health_check.last_check = datetime.utcnow()
                health_check.last_result = result
                health_check.last_error = None
            
            logger.debug(f"Health check {name}: {'PASS' if result else 'FAIL'} ({duration:.3f}s)")
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Health check {name} timed out after {health_check.timeout_seconds}s"
            logger.error(error_msg)
            
            with self._lock:
                health_check.last_check = datetime.utcnow()
                health_check.last_result = False
                health_check.last_error = error_msg
            
            return False
            
        except Exception as e:
            error_msg = f"Health check {name} failed: {e}"
            logger.error(error_msg)
            
            with self._lock:
                health_check.last_check = datetime.utcnow()
                health_check.last_result = False
                health_check.last_error = error_msg
            
            return False
    
    async def run_all_health_checks(self) -> Dict[str, bool]:
        """Run all health checks"""
        results = {}
        
        for name in self.health_checks.keys():
            results[name] = await self.run_health_check(name)
        
        # Update overall health
        critical_checks = [
            name for name, check in self.health_checks.items() 
            if check.critical
        ]
        
        if critical_checks:
            self.overall_health = all(
                results.get(name, False) for name in critical_checks
            )
        else:
            self.overall_health = all(results.values())
        
        self.last_health_update = datetime.utcnow()
        
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        with self._lock:
            return {
                "overall_health": self.overall_health,
                "last_update": self.last_health_update.isoformat(),
                "checks": {
                    name: {
                        "status": "pass" if check.last_result else "fail",
                        "last_check": check.last_check.isoformat() if check.last_check else None,
                        "error": check.last_error,
                        "critical": check.critical
                    }
                    for name, check in self.health_checks.items()
                }
            }


class PerformanceTracker:
    """Performance tracking and analysis"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.response_times: deque = deque(maxlen=window_size)
        self.throughput_data: deque = deque(maxlen=window_size)
        self.error_rates: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()
    
    def record_response_time(self, duration: float):
        """Record response time"""
        with self._lock:
            self.response_times.append({
                'duration': duration,
                'timestamp': time.time()
            })
    
    def record_throughput(self, requests_per_second: float):
        """Record throughput"""
        with self._lock:
            self.throughput_data.append({
                'rps': requests_per_second,
                'timestamp': time.time()
            })
    
    def record_error_rate(self, error_rate: float):
        """Record error rate"""
        with self._lock:
            self.error_rates.append({
                'rate': error_rate,
                'timestamp': time.time()
            })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        with self._lock:
            now = time.time()
            recent_cutoff = now - 300  # Last 5 minutes
            
            # Recent response times
            recent_response_times = [
                item['duration'] for item in self.response_times
                if item['timestamp'] > recent_cutoff
            ]
            
            # Recent throughput
            recent_throughput = [
                item['rps'] for item in self.throughput_data
                if item['timestamp'] > recent_cutoff
            ]
            
            # Recent error rates
            recent_error_rates = [
                item['rate'] for item in self.error_rates
                if item['timestamp'] > recent_cutoff
            ]
            
            return {
                "response_times": {
                    "avg": sum(recent_response_times) / len(recent_response_times) if recent_response_times else 0,
                    "min": min(recent_response_times) if recent_response_times else 0,
                    "max": max(recent_response_times) if recent_response_times else 0,
                    "p95": self._percentile(recent_response_times, 95) if recent_response_times else 0,
                    "p99": self._percentile(recent_response_times, 99) if recent_response_times else 0,
                },
                "throughput": {
                    "avg_rps": sum(recent_throughput) / len(recent_throughput) if recent_throughput else 0,
                    "max_rps": max(recent_throughput) if recent_throughput else 0,
                },
                "error_rate": {
                    "avg": sum(recent_error_rates) / len(recent_error_rates) if recent_error_rates else 0,
                    "max": max(recent_error_rates) if recent_error_rates else 0,
                },
                "sample_size": len(recent_response_times),
                "time_window": "5 minutes"
            }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class ObservabilityManager:
    """Central observability management"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.health_check_manager = HealthCheckManager()
        self.performance_tracker = PerformanceTracker()
        self._setup_default_health_checks()
        self._setup_default_alert_rules()
    
    def _setup_default_health_checks(self):
        """Setup default health checks"""
        # Database health check
        async def check_database():
            # This would check database connectivity
            return True  # Placeholder
        
        self.health_check_manager.add_health_check(HealthCheck(
            name="database",
            check_function=check_database,
            interval_seconds=30,
            critical=True
        ))
        
        # Cache health check
        async def check_cache():
            # This would check Redis connectivity
            return True  # Placeholder
        
        self.health_check_manager.add_health_check(HealthCheck(
            name="cache",
            check_function=check_cache,
            interval_seconds=30,
            critical=False
        ))
        
        # External API health check
        async def check_external_apis():
            # This would check NASA/ESA API connectivity
            return True  # Placeholder
        
        self.health_check_manager.add_health_check(HealthCheck(
            name="external_apis",
            check_function=check_external_apis,
            interval_seconds=60,
            critical=False
        ))
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        def high_error_rate_rule(metrics_data: Dict[str, Any], alert_manager: AlertManager):
            # Check for high error rate
            error_rate = metrics_data.get('error_rate', 0)
            if error_rate > 0.05:  # 5% error rate
                alert_manager.create_alert(
                    name="high_error_rate",
                    severity=AlertSeverity.HIGH,
                    message=f"Error rate is {error_rate:.2%}",
                    metadata={"error_rate": error_rate}
                )
        
        def high_response_time_rule(metrics_data: Dict[str, Any], alert_manager: AlertManager):
            # Check for high response time
            avg_response_time = metrics_data.get('avg_response_time', 0)
            if avg_response_time > 2.0:  # 2 seconds
                alert_manager.create_alert(
                    name="high_response_time",
                    severity=AlertSeverity.MEDIUM,
                    message=f"Average response time is {avg_response_time:.2f}s",
                    metadata={"avg_response_time": avg_response_time}
                )
        
        def low_throughput_rule(metrics_data: Dict[str, Any], alert_manager: AlertManager):
            # Check for low throughput
            throughput = metrics_data.get('throughput', 0)
            if throughput < 10:  # Less than 10 requests per second
                alert_manager.create_alert(
                    name="low_throughput",
                    severity=AlertSeverity.MEDIUM,
                    message=f"Throughput is {throughput:.2f} req/s",
                    metadata={"throughput": throughput}
                )
        
        self.alert_manager.add_alert_rule("high_error_rate", high_error_rate_rule)
        self.alert_manager.add_alert_rule("high_response_time", high_response_time_rule)
        self.alert_manager.add_alert_rule("low_throughput", low_throughput_rule)
    
    async def start_background_tasks(self):
        """Start background monitoring tasks"""
        # Start system metrics collection
        asyncio.create_task(self._system_metrics_loop())
        
        # Start health check loop
        asyncio.create_task(self._health_check_loop())
        
        # Start alert rule checking
        asyncio.create_task(self._alert_check_loop())
    
    async def _system_metrics_loop(self):
        """Background task for system metrics collection"""
        while True:
            try:
                self.metrics_collector.update_system_metrics()
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"System metrics collection failed: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _health_check_loop(self):
        """Background task for health checks"""
        while True:
            try:
                await self.health_check_manager.run_all_health_checks()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Health check loop failed: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _alert_check_loop(self):
        """Background task for alert rule checking"""
        while True:
            try:
                # Get current metrics data
                performance_summary = self.performance_tracker.get_performance_summary()
                health_status = self.health_check_manager.get_health_status()
                
                metrics_data = {
                    **performance_summary.get("response_times", {}),
                    **performance_summary.get("throughput", {}),
                    **performance_summary.get("error_rate", {}),
                    "overall_health": health_status["overall_health"]
                }
                
                # Check alert rules
                self.alert_manager.check_alert_rules(metrics_data)
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Alert check loop failed: {e}")
                await asyncio.sleep(120)  # Wait longer on error
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        return {
            "health": self.health_check_manager.get_health_status(),
            "performance": self.performance_tracker.get_performance_summary(),
            "alerts": {
                "active": len(self.alert_manager.get_active_alerts()),
                "critical": len(self.alert_manager.get_alerts_by_severity(AlertSeverity.CRITICAL)),
                "recent": [
                    {
                        "name": alert.name,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in list(self.alert_manager.alerts)[-10:]  # Last 10 alerts
                ]
            },
            "metrics": self.metrics_collector.get_metrics_text(),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global observability instance
_observability_manager: Optional[ObservabilityManager] = None


def get_observability_manager() -> ObservabilityManager:
    """Get global observability manager"""
    global _observability_manager
    
    if _observability_manager is None:
        _observability_manager = ObservabilityManager()
    
    return _observability_manager


async def initialize_observability():
    """Initialize observability system"""
    manager = get_observability_manager()
    await manager.start_background_tasks()
    logger.info("✅ Observability system initialized")


async def cleanup_observability():
    """Cleanup observability system"""
    global _observability_manager
    
    if _observability_manager:
        # Stop background tasks would be implemented here
        _observability_manager = None
        logger.info("✅ Observability system cleaned up")
