"""
Advanced Monitoring Middleware for FastAPI
Automatic request/response tracking, performance monitoring, and analytics
"""

import json
import logging
import time
from typing import Callable, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from core.monitoring import APIMetrics, get_metrics_collector

logger = logging.getLogger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Advanced monitoring middleware with comprehensive tracking"""

    def __init__(
        self,
        app: ASGIApp,
        track_body_size: bool = True,
        exclude_paths: Optional[list] = None,
        sample_rate: float = 1.0,
    ):
        super().__init__(app)
        self.track_body_size = track_body_size
        self.exclude_paths = exclude_paths or [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",
        ]
        self.sample_rate = sample_rate
        self.metrics_collector = get_metrics_collector()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip monitoring for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Sample requests based on sample_rate
        if self.sample_rate < 1.0:
            import secrets

            if secrets.randbelow(100) / 100.0 > self.sample_rate:
                return await call_next(request)

        # Start timing
        start_time = time.time()

        # Get request info
        method = request.method
        path = request.url.path
        user_agent = request.headers.get("user-agent", "Unknown")
        client_ip = self._get_client_ip(request)

        # Get request size
        request_size = 0
        if self.track_body_size and hasattr(request, "body"):
            try:
                body = await request.body()
                request_size = len(body) if body else 0
                # Restore body for downstream processing
                request._body = body
            except Exception as e:
                logger.warning(f"Failed to read request body: {e}")

        # Process request
        response = None
        status_code = 500
        response_size = 0

        try:
            response = await call_next(request)
            status_code = response.status_code

            # Get response size
            if self.track_body_size and hasattr(response, "body"):
                try:
                    if hasattr(response, "body_iterator"):
                        # For streaming responses, estimate size
                        response_size = 0
                    else:
                        response_size = len(getattr(response, "body", b""))
                except Exception as e:
                    logger.warning(f"Failed to measure response size: {e}")

        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            status_code = 500
            response = JSONResponse(
                status_code=500, content={"detail": "Internal server error"}
            )

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000

        # Record metrics
        api_metric = APIMetrics(
            endpoint=path,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            request_size_bytes=request_size,
            response_size_bytes=response_size,
            user_agent=user_agent,
            ip_address=client_ip,
        )

        self.metrics_collector.record_api_metric(api_metric)

        # Log slow requests
        if response_time_ms > 1000:
            logger.warning(
                f"Slow request: {method} {path} took {response_time_ms:.0f}ms "
                f"(Status: {status_code}, IP: {client_ip})"
            )

        # Add performance headers
        if response:
            response.headers["X-Response-Time"] = f"{response_time_ms:.2f}ms"
            response.headers["X-Request-ID"] = str(int(start_time * 1000000))

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fallback to client host
        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting with different strategies"""

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        burst_limit: int = 10,
        exclude_paths: Optional[list] = None,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.exclude_paths = exclude_paths or ["/docs", "/redoc", "/openapi.json"]

        # Simple in-memory rate limiting (use Redis in production)
        self.request_counts = {}
        self.burst_counts = {}
        self.last_cleanup = time.time()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        current_time = time.time()

        # Cleanup old entries every 5 minutes
        if current_time - self.last_cleanup > 300:
            self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time

        # Check burst limit (requests per second)
        burst_key = f"{client_ip}:{int(current_time)}"
        burst_count = self.burst_counts.get(burst_key, 0)

        if burst_count >= self.burst_limit:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests - burst limit exceeded",
                    "retry_after": 1,
                },
                headers={"Retry-After": "1"},
            )

        # Check per-minute limit
        minute_key = f"{client_ip}:{int(current_time // 60)}"
        minute_count = self.request_counts.get(minute_key, 0)

        if minute_count >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests - rate limit exceeded",
                    "retry_after": 60,
                },
                headers={"Retry-After": "60"},
            )

        # Update counters
        self.burst_counts[burst_key] = burst_count + 1
        self.request_counts[minute_key] = minute_count + 1

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_minute - minute_count - 1)
        )
        response.headers["X-RateLimit-Reset"] = str(
            int((int(current_time // 60) + 1) * 60)
        )

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"

    def _cleanup_old_entries(self, current_time: float):
        """Remove old rate limiting entries"""
        # Remove burst entries older than 2 seconds
        burst_cutoff = int(current_time) - 2
        self.burst_counts = {
            k: v
            for k, v in self.burst_counts.items()
            if int(k.split(":")[1]) > burst_cutoff
        }

        # Remove minute entries older than 2 minutes
        minute_cutoff = int(current_time // 60) - 2
        self.request_counts = {
            k: v
            for k, v in self.request_counts.items()
            if int(k.split(":")[1]) > minute_cutoff
        }


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security headers and basic protection"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self' https:; "
            "font-src 'self' https:; "
        )

        return response


class CompressionMiddleware(BaseHTTPMiddleware):
    """Advanced response compression"""

    def __init__(self, app: ASGIApp, minimum_size: int = 1000, compress_level: int = 6):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compress_level = compress_level

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Check if client accepts compression
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return response

        # Check content type
        content_type = response.headers.get("content-type", "")
        compressible_types = [
            "application/json",
            "text/html",
            "text/css",
            "text/javascript",
            "application/javascript",
            "text/plain",
        ]

        if not any(ct in content_type for ct in compressible_types):
            return response

        # Check response size
        if hasattr(response, "body"):
            body = getattr(response, "body", b"")
            if len(body) < self.minimum_size:
                return response

            # Compress response
            import gzip

            compressed_body = gzip.compress(body, compresslevel=self.compress_level)

            # Update response
            response.body = compressed_body
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Content-Length"] = str(len(compressed_body))

        return response
