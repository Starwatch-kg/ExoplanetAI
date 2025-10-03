"""
Advanced Monitoring System for AstroManas
Comprehensive performance tracking, metrics collection, and system health monitoring
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric measurement"""

    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System performance metrics"""

    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_sent_mb: float
    network_recv_mb: float
    active_connections: int
    load_average: List[float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class APIMetrics:
    """API performance metrics"""

    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    request_size_bytes: int
    response_size_bytes: int
    timestamp: float = field(default_factory=time.time)
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None


class MetricsCollector:
    """Advanced metrics collection and aggregation"""

    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.system_metrics: deque = deque(maxlen=1000)
        self.api_metrics: deque = deque(maxlen=5000)
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "response_time_ms": 5000.0,
            "error_rate": 0.05,
        }
        self._lock = threading.Lock()
        self._collection_active = False

    def start_collection(self):
        """Start background metrics collection"""
        self._collection_active = True
        threading.Thread(target=self._collect_system_metrics, daemon=True).start()
        logger.info("📊 Metrics collection started")

    def stop_collection(self):
        """Stop background metrics collection"""
        self._collection_active = False
        logger.info("📊 Metrics collection stopped")

    def _collect_system_metrics(self):
        """Background system metrics collection"""
        while self._collection_active:
            try:
                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()

                # Disk usage
                disk = psutil.disk_usage("/")

                # Network stats
                network = psutil.net_io_counters()

                # Load average (Unix-like systems)
                try:
                    load_avg = list(psutil.getloadavg())
                except AttributeError:
                    load_avg = [0.0, 0.0, 0.0]  # Windows fallback

                # Active connections
                connections = len(psutil.net_connections())

                metrics = SystemMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / (1024 * 1024),
                    memory_available_mb=memory.available / (1024 * 1024),
                    disk_usage_percent=disk.percent,
                    network_sent_mb=network.bytes_sent / (1024 * 1024),
                    network_recv_mb=network.bytes_recv / (1024 * 1024),
                    active_connections=connections,
                    load_average=load_avg,
                )

                with self._lock:
                    self.system_metrics.append(metrics)

                # Check for alerts
                self._check_system_alerts(metrics)

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")

            time.sleep(30)  # Collect every 30 seconds

    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a custom metric"""
        point = MetricPoint(timestamp=time.time(), value=value, labels=labels or {})

        with self._lock:
            self.metrics[name].append(point)

    def record_api_metric(self, metric: APIMetrics):
        """Record API performance metric"""
        with self._lock:
            self.api_metrics.append(metric)

        # Check for API alerts
        self._check_api_alerts(metric)

    def get_system_metrics(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get system metrics for the last N minutes"""
        cutoff = time.time() - (minutes * 60)
        with self._lock:
            return [m for m in self.system_metrics if m.timestamp >= cutoff]

    def get_api_metrics(self, minutes: int = 60) -> List[APIMetrics]:
        """Get API metrics for the last N minutes"""
        cutoff = time.time() - (minutes * 60)
        with self._lock:
            return [m for m in self.api_metrics if m.timestamp >= cutoff]

    def get_metric_summary(self, name: str, minutes: int = 60) -> Dict[str, float]:
        """Get statistical summary of a metric"""
        cutoff = time.time() - (minutes * 60)

        with self._lock:
            points = [p for p in self.metrics[name] if p.timestamp >= cutoff]

        if not points:
            return {}

        values = [p.value for p in points]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else 0,
        }

    def get_endpoint_performance(
        self, minutes: int = 60
    ) -> Dict[str, Dict[str, float]]:
        """Get performance metrics by endpoint"""
        metrics = self.get_api_metrics(minutes)

        endpoint_stats = defaultdict(list)
        for metric in metrics:
            key = f"{metric.method} {metric.endpoint}"
            endpoint_stats[key].append(metric.response_time_ms)

        result = {}
        for endpoint, times in endpoint_stats.items():
            result[endpoint] = {
                "count": len(times),
                "avg_response_time": sum(times) / len(times),
                "min_response_time": min(times),
                "max_response_time": max(times),
                "p95_response_time": (
                    sorted(times)[int(len(times) * 0.95)] if times else 0
                ),
            }

        return result

    def get_error_rate(self, minutes: int = 60) -> float:
        """Calculate error rate for the last N minutes"""
        metrics = self.get_api_metrics(minutes)

        if not metrics:
            return 0.0

        total_requests = len(metrics)
        error_requests = sum(1 for m in metrics if m.status_code >= 400)

        return error_requests / total_requests if total_requests > 0 else 0.0

    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics against thresholds"""
        alerts = []

        if metrics.cpu_percent > self.thresholds["cpu_percent"]:
            alerts.append(
                {
                    "type": "system",
                    "severity": "warning",
                    "message": f"High CPU usage: {metrics.cpu_percent:.1f}%",
                    "timestamp": metrics.timestamp,
                }
            )

        if metrics.memory_percent > self.thresholds["memory_percent"]:
            alerts.append(
                {
                    "type": "system",
                    "severity": "warning",
                    "message": f"High memory usage: {metrics.memory_percent:.1f}%",
                    "timestamp": metrics.timestamp,
                }
            )

        for alert in alerts:
            self.alerts.append(alert)
            logger.warning(f"🚨 Alert: {alert['message']}")

    def _check_api_alerts(self, metric: APIMetrics):
        """Check API metrics against thresholds"""
        if metric.response_time_ms > self.thresholds["response_time_ms"]:
            alert = {
                "type": "api",
                "severity": "warning",
                "message": f"Slow response: {metric.endpoint} took {metric.response_time_ms:.0f}ms",
                "timestamp": metric.timestamp,
            }
            self.alerts.append(alert)
            logger.warning(f"🚨 Alert: {alert['message']}")

    def get_recent_alerts(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get alerts from the last N minutes"""
        cutoff = time.time() - (minutes * 60)
        return [a for a in self.alerts if a["timestamp"] >= cutoff]

    def clear_old_alerts(self, hours: int = 24):
        """Clear alerts older than N hours"""
        cutoff = time.time() - (hours * 3600)
        self.alerts = [a for a in self.alerts if a["timestamp"] >= cutoff]


class PerformanceProfiler:
    """Advanced performance profiling and optimization suggestions"""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.profiles: Dict[str, List[float]] = defaultdict(list)

    @asynccontextmanager
    async def profile_async(self, operation_name: str):
        """Profile async operation"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = (time.time() - start_time) * 1000
            self.profiles[operation_name].append(duration)
            self.collector.record_metric(f"operation_time_{operation_name}", duration)

    def profile_sync(self, operation_name: str):
        """Decorator for profiling sync operations"""

        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration = (time.time() - start_time) * 1000
                    self.profiles[operation_name].append(duration)
                    self.collector.record_metric(
                        f"operation_time_{operation_name}", duration
                    )

            return wrapper

        return decorator

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            "system_health": self._analyze_system_health(),
            "api_performance": self._analyze_api_performance(),
            "bottlenecks": self._identify_bottlenecks(),
            "recommendations": self._generate_recommendations(),
            "timestamp": datetime.now().isoformat(),
        }
        return report

    def _analyze_system_health(self) -> Dict[str, Any]:
        """Analyze overall system health"""
        metrics = self.collector.get_system_metrics(60)

        if not metrics:
            return {"status": "unknown", "reason": "No metrics available"}

        latest = metrics[-1]
        avg_cpu = sum(m.cpu_percent for m in metrics) / len(metrics)
        avg_memory = sum(m.memory_percent for m in metrics) / len(metrics)

        health_score = 100
        issues = []

        if avg_cpu > 70:
            health_score -= 20
            issues.append("High CPU usage")

        if avg_memory > 80:
            health_score -= 25
            issues.append("High memory usage")

        if latest.disk_usage_percent > 90:
            health_score -= 15
            issues.append("Low disk space")

        status = (
            "excellent"
            if health_score >= 90
            else (
                "good"
                if health_score >= 70
                else "warning" if health_score >= 50 else "critical"
            )
        )

        return {
            "status": status,
            "score": health_score,
            "issues": issues,
            "current_cpu": latest.cpu_percent,
            "current_memory": latest.memory_percent,
            "avg_cpu_1h": avg_cpu,
            "avg_memory_1h": avg_memory,
        }

    def _analyze_api_performance(self) -> Dict[str, Any]:
        """Analyze API performance patterns"""
        metrics = self.collector.get_api_metrics(60)

        if not metrics:
            return {"status": "no_data"}

        total_requests = len(metrics)
        avg_response_time = sum(m.response_time_ms for m in metrics) / total_requests
        error_rate = self.collector.get_error_rate(60)

        # Response time percentiles
        times = sorted([m.response_time_ms for m in metrics])
        p50 = times[int(len(times) * 0.5)] if times else 0
        p95 = times[int(len(times) * 0.95)] if times else 0
        p99 = times[int(len(times) * 0.99)] if times else 0

        performance_grade = (
            "A"
            if avg_response_time < 500 and error_rate < 0.01
            else (
                "B"
                if avg_response_time < 1000 and error_rate < 0.05
                else "C" if avg_response_time < 2000 and error_rate < 0.1 else "D"
            )
        )

        return {
            "grade": performance_grade,
            "total_requests": total_requests,
            "avg_response_time": avg_response_time,
            "error_rate": error_rate * 100,
            "percentiles": {"p50": p50, "p95": p95, "p99": p99},
        }

    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        # Check endpoint performance
        endpoint_perf = self.collector.get_endpoint_performance(60)
        for endpoint, stats in endpoint_perf.items():
            if stats["avg_response_time"] > 2000:
                bottlenecks.append(
                    {
                        "type": "slow_endpoint",
                        "endpoint": endpoint,
                        "avg_time": stats["avg_response_time"],
                        "severity": (
                            "high" if stats["avg_response_time"] > 5000 else "medium"
                        ),
                    }
                )

        # Check system resources
        system_metrics = self.collector.get_system_metrics(30)
        if system_metrics:
            avg_cpu = sum(m.cpu_percent for m in system_metrics) / len(system_metrics)
            avg_memory = sum(m.memory_percent for m in system_metrics) / len(
                system_metrics
            )

            if avg_cpu > 80:
                bottlenecks.append(
                    {"type": "high_cpu", "value": avg_cpu, "severity": "high"}
                )

            if avg_memory > 85:
                bottlenecks.append(
                    {"type": "high_memory", "value": avg_memory, "severity": "high"}
                )

        return bottlenecks

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        bottlenecks = self._identify_bottlenecks()

        for bottleneck in bottlenecks:
            if bottleneck["type"] == "slow_endpoint":
                recommendations.append(
                    f"Optimize {bottleneck['endpoint']} - consider caching, database indexing, or async processing"
                )
            elif bottleneck["type"] == "high_cpu":
                recommendations.append(
                    "High CPU usage detected - consider scaling horizontally or optimizing CPU-intensive operations"
                )
            elif bottleneck["type"] == "high_memory":
                recommendations.append(
                    "High memory usage detected - check for memory leaks or consider increasing available RAM"
                )

        # General recommendations
        api_perf = self._analyze_api_performance()
        if api_perf.get("error_rate", 0) > 5:
            recommendations.append(
                "High error rate detected - review error handling and input validation"
            )

        if not recommendations:
            recommendations.append(
                "System performance is optimal - no immediate optimizations needed"
            )

        return recommendations


# Global metrics collector instance
metrics_collector = MetricsCollector()
performance_profiler = PerformanceProfiler(metrics_collector)


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    return metrics_collector


def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance"""
    return performance_profiler
