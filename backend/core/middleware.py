"""
Middleware module for Exoplanet AI backend
"""

import time
import uuid
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse


def setup_middleware(app: FastAPI, config: Dict[str, Any] = None):
    """Setup application middleware"""
    if config is None:
        config = {}

    # Setup CORS middleware
    allowed_origins = config.get(
        "allowed_origins",
        [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ],
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
        allow_headers=["*"],
    )

    # Setup GZip middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add custom middleware for request ID and timing
    app.middleware("http")(request_id_middleware)
    app.middleware("http")(timing_middleware)


async def request_id_middleware(request: Request, call_next):
    """Middleware to add request ID to each request"""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    response = await call_next(request)

    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(time.time() - start_time)

    return response


async def timing_middleware(request: Request, call_next):
    """Middleware to time requests"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


def get_request_context(request: Request) -> Dict[str, Any]:
    """Get request context including request ID and trace info"""
    context = {
        "request_id": request.headers.get("X-Request-ID") or str(uuid.uuid4()),
        "timestamp": time.time(),
        "method": request.method,
        "url": str(request.url),
        "user_agent": request.headers.get("user-agent", ""),
        "client_host": request.client.host if request.client else None,
    }

    # Add trace ID if available
    trace_id = request.headers.get("X-Trace-ID")
    if trace_id:
        context["trace_id"] = trace_id
    else:
        context["trace_id"] = str(uuid.uuid4())

    return context
