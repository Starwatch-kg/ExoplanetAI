"""
Real Metrics collection module for Exoplanet AI backend
Реальная система сбора метрик
"""

import asyncio
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import Request, Response


class MetricsCollector:
    """Real metrics collector with in-memory storage"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.lock = threading.Lock()

        # Application metrics
        self.app_status = "starting"
        self.start_time = time.time()

        # Request metrics
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.response_times = deque(maxlen=max_history)

        # Analysis metrics
        self.analysis_history = deque(maxlen=max_history)
        self.analysis_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "total_candidates_found": 0,
            "avg_processing_time": 0.0,
        }

        # ML metrics
        self.ml_metrics = {
            "total_inferences": 0,
            "successful_inferences": 0,
            "avg_inference_time": 0.0,
            "model_usage": defaultdict(int),
        }

        # NASA API metrics
        self.nasa_api_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "data_points_downloaded": 0,
        }

    def record_exoplanet_analysis(
        self,
        catalog: str,
        mission: str,
        status: str,
        duration: float,
        candidates_found: int = 0,
        max_snr: float = 0.0,
    ):
        """Record exoplanet analysis metrics"""
        with self.lock:
            analysis_record = {
                "timestamp": datetime.utcnow(),
                "catalog": catalog,
                "mission": mission,
                "status": status,
                "duration": duration,
                "candidates_found": candidates_found,
                "max_snr": max_snr,
            }

            self.analysis_history.append(analysis_record)
            self.analysis_stats["total_analyses"] += 1

            if status == "success":
                self.analysis_stats["successful_analyses"] += 1
                self.analysis_stats["total_candidates_found"] += candidates_found
            else:
                self.analysis_stats["failed_analyses"] += 1

            # Update average processing time
            total_time = sum(r["duration"] for r in self.analysis_history)
            self.analysis_stats["avg_processing_time"] = total_time / len(
                self.analysis_history
            )

    def record_error(self, error_type: str, service: str = "unknown"):
        """Record error metrics"""
        with self.lock:
            self.error_counts[f"{service}:{error_type}"] += 1

    def record_api_call(
        self, endpoint: str, method: str, status_code: int, duration: float
    ):
        """Record API call metrics"""
        with self.lock:
            key = f"{method} {endpoint}"
            self.request_counts[key] += 1
            self.response_times.append(
                {
                    "timestamp": datetime.utcnow(),
                    "endpoint": endpoint,
                    "method": method,
                    "status_code": status_code,
                    "duration": duration,
                }
            )

    def record_ml_inference(self, model_name: str, duration: float, success: bool):
        """Record ML inference metrics"""
        with self.lock:
            self.ml_metrics["total_inferences"] += 1
            self.ml_metrics["model_usage"][model_name] += 1

            if success:
                self.ml_metrics["successful_inferences"] += 1

            # Update average inference time
            if self.ml_metrics["total_inferences"] > 0:
                current_avg = self.ml_metrics["avg_inference_time"]
                total = self.ml_metrics["total_inferences"]
                self.ml_metrics["avg_inference_time"] = (
                    current_avg * (total - 1) + duration
                ) / total

    def record_nasa_api_call(
        self, success: bool, duration: float, data_points: int = 0
    ):
        """Record NASA API call metrics"""
        with self.lock:
            self.nasa_api_metrics["total_requests"] += 1

            if success:
                self.nasa_api_metrics["successful_requests"] += 1
                self.nasa_api_metrics["data_points_downloaded"] += data_points
            else:
                self.nasa_api_metrics["failed_requests"] += 1

            # Update average response time
            total = self.nasa_api_metrics["total_requests"]
            current_avg = self.nasa_api_metrics["avg_response_time"]
            self.nasa_api_metrics["avg_response_time"] = (
                current_avg * (total - 1) + duration
            ) / total

    def set_app_status(self, status: str):
        """Set application status"""
        with self.lock:
            self.app_status = status

    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health and status metrics"""
        with self.lock:
            uptime = time.time() - self.start_time

            return {
                "status": self.app_status,
                "uptime_seconds": uptime,
                "uptime_human": str(timedelta(seconds=int(uptime))),
                "total_requests": sum(self.request_counts.values()),
                "total_errors": sum(self.error_counts.values()),
                "analysis_stats": self.analysis_stats.copy(),
                "ml_metrics": self.ml_metrics.copy(),
                "nasa_api_metrics": self.nasa_api_metrics.copy(),
            }

    def get_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis records"""
        with self.lock:
            recent = list(self.analysis_history)[-limit:]
            return [
                {**record, "timestamp": record["timestamp"].isoformat()}
                for record in recent
            ]

    def get_error_summary(self) -> Dict[str, int]:
        """Get error summary"""
        with self.lock:
            return dict(self.error_counts)


# Global metrics collector instance
metrics_collector = MetricsCollector()


class MetricsMiddleware:
    """Middleware to collect metrics for all requests"""

    def __init__(self, app, metrics_collector_instance=metrics_collector):
        self.app = app
        self.metrics = metrics_collector_instance

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        status_code = 200

        # Capture response to get status code
        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        await self.app(scope, receive, send_wrapper)

        # Record metrics
        duration = time.time() - start_time
        path = scope.get("path", "unknown")
        method = scope.get("method", "unknown")

        self.metrics.record_api_call(path, method, status_code, duration)
