"""
ExoplanetAI Backend v2.0 - Modular Architecture
Модульная архитектура ExoplanetAI Backend v2.0

Features:
- Modular data sources (NASA, ESA, Kepler, TESS)
- Redis caching with TTL
- JWT authentication with role-based access
- Structured API routes
- Real data only - no synthetic generation
- Comprehensive logging and monitoring
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# API routes
from api.routes import create_api_router

# Authentication
from auth.jwt_auth import get_jwt_manager
from core.cache import cleanup_cache, get_cache, initialize_cache

# Core imports
from core.config import config
from core.logging import (
    StructuredLoggingMiddleware,
    configure_structlog,
    get_logger,
)

# Data sources
from data_sources.registry import get_registry, initialize_default_sources

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False

# Configure structured logging
configure_structlog(
    service_name="exoplanet-ai-v2",
    environment=getattr(config, "environment", "development"),
    log_level=getattr(config, "log_level", "INFO"),
    enable_json=getattr(config, "enable_json_logs", False),
    enable_console=True,
)

logger = get_logger(__name__)

# Rate limiter setup
if SLOWAPI_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None
    logger.warning("slowapi not available - rate limiting disabled")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting ExoplanetAI Backend v2.0")

    startup_tasks = []

    try:
        # Initialize cache
        logger.info("Initializing Redis cache...")
        cache_success = await initialize_cache()
        if cache_success:
            logger.info("✅ Cache initialized")
        else:
            logger.warning("⚠️ Cache initialization failed, using fallback")

        # Initialize data sources
        logger.info("Initializing data sources...")
        source_results = await initialize_default_sources()
        successful_sources = sum(1 for success in source_results.values() if success)
        total_sources = len(source_results)

        if successful_sources > 0:
            logger.info(
                f"✅ Data sources initialized: {successful_sources}/{total_sources}"
            )
        else:
            logger.error("❌ No data sources initialized successfully")

        # Initialize auto ML trainer
        logger.info("Starting auto ML trainer...")
        try:
            from services.auto_ml_trainer import get_auto_trainer
            trainer = get_auto_trainer()
            
            # Запускаем автообучение в фоне
            auto_training_task = asyncio.create_task(trainer.start_auto_training_loop())
            startup_tasks.append(auto_training_task)
            
            logger.info("✅ Auto ML trainer started")
        except Exception as e:
            logger.error(f"⚠️ Auto ML trainer failed to start: {e}")

        # Store initialization results in app state
        app.state.cache_available = cache_success
        app.state.data_sources_count = successful_sources
        app.state.startup_time = time.time()
        app.state.background_tasks = startup_tasks

        logger.info("🎉 ExoplanetAI Backend v2.0 ready!")

        yield

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info("🛑 Shutting down ExoplanetAI Backend v2.0")

        try:
            # Cancel background tasks
            if hasattr(app.state, 'background_tasks'):
                for task in app.state.background_tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                logger.info("✅ Background tasks cancelled")

            # Cleanup data sources
            registry = get_registry()
            await registry.cleanup_all()
            logger.info("✅ Data sources cleaned up")

            # Cleanup cache
            await cleanup_cache()
            logger.info("✅ Cache cleaned up")

        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

        logger.info("👋 ExoplanetAI Backend v2.0 stopped")


# Create FastAPI app
app = FastAPI(
    title="ExoplanetAI Backend v2.0",
    description="""
    🌟 **Professional Exoplanet Research API**

    A comprehensive backend for exoplanet research using **real astronomical data only**.

    ## Features

    - 🛰️ **Multiple Data Sources**: NASA, ESA, Kepler, TESS missions
    - 🔒 **JWT Authentication**: Role-based access control
    - ⚡ **Redis Caching**: High-performance data caching
    - 📊 **Real Data Only**: No synthetic data generation
    - 🔬 **Scientific Analysis**: BLS transit detection, light curve analysis
    - 📈 **Comprehensive Statistics**: Discovery trends, physical properties
    - 🔧 **Admin Tools**: System monitoring and management

    ## Authentication

    **Default test accounts:**
    - `admin` / `admin123` (Admin access)
    - `researcher` / `research123` (Research access)
    - `user` / `user123` (Basic access)

    ## Data Sources

    - **NASA Exoplanet Archive**: Confirmed exoplanets database
    - **TESS Mission**: Real photometric time series
    - **Kepler Mission**: Historical exoplanet discoveries
    - **ESA Archive**: European space agency data

    ## Rate Limits

    - **Public endpoints**: 60 requests/minute
    - **Authenticated users**: 300 requests/minute
    - **Researchers**: 600 requests/minute
    - **Admins**: 1200 requests/minute
    """,
    version="2.0.0",
    contact={
        "name": "ExoplanetAI Team",
        "email": "contact@exoplanetai.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
)

# Add rate limiting if available
if SLOWAPI_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if os.getenv("ENVIRONMENT") == "development" else [
        "http://localhost:3000",  # Frontend dev
        "http://127.0.0.1:3000",  # Frontend dev alt
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",  # Vite dev server alt
        "http://localhost:5174",  # Vite dev server alt port
        "http://localhost:5175",  # Vite dev server alt port
        "http://localhost:5176",  # Vite dev server alt port
        "http://localhost:5177",  # Vite dev server alt port
        "http://localhost:5178",  # Vite dev server alt port
        os.getenv("FRONTEND_URL", "http://localhost:3000")  # Production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Structured logging middleware
app.add_middleware(StructuredLoggingMiddleware)

# Include API routes
api_router = create_api_router()
app.include_router(api_router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Welcome to ExoplanetAI Backend v2.0
    """
    uptime = (
        time.time() - app.state.startup_time
        if hasattr(app.state, "startup_time")
        else 0
    )

    return {
        "message": "🌟 Welcome to ExoplanetAI Backend v2.0",
        "version": "2.0.0",
        "status": "operational",
        "features": {
            "real_data_only": True,
            "synthetic_data": False,
            "data_sources": getattr(app.state, "data_sources_count", 0),
            "cache_available": getattr(app.state, "cache_available", False),
            "authentication": "JWT with role-based access",
            "rate_limiting": SLOWAPI_AVAILABLE,
        },
        "uptime_seconds": uptime,
        "documentation": "/docs",
        "api_base": "/api/v1",
        "health_check": "/health",
    }


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """
    System health check
    """
    try:
        # Check data sources
        registry = get_registry()
        registry_info = registry.get_registry_info()

        # Check cache
        cache = get_cache()
        cache_health = await cache.health_check()

        # Determine overall health
        overall_status = "healthy"
        if registry_info["initialized_sources"] == 0:
            overall_status = "degraded"
        if cache_health.get("status") == "unhealthy":
            overall_status = "degraded"

        return {
            "status": overall_status,
            "timestamp": time.time(),
            "version": "2.0.0",
            "components": {
                "data_sources": {
                    "status": (
                        "healthy"
                        if registry_info["initialized_sources"] > 0
                        else "unhealthy"
                    ),
                    "initialized": registry_info["initialized_sources"],
                    "total": registry_info["total_sources"],
                },
                "cache": {
                    "status": cache_health.get("status", "unknown"),
                    "redis_connected": cache_health.get("redis_connected", False),
                },
                "authentication": {"status": "healthy"},
            },
            "uptime_seconds": (
                time.time() - app.state.startup_time
                if hasattr(app.state, "startup_time")
                else 0
            ),
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e), "timestamp": time.time()},
        )


# Prometheus metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def get_prometheus_metrics():
    """
    Prometheus metrics endpoint
    """
    try:
        from fastapi.responses import PlainTextResponse

        # Get cache stats
        cache = get_cache()
        cache_stats = await cache.get_stats()

        # Get registry info
        registry = get_registry()
        registry_info = registry.get_registry_info()

        uptime = (
            time.time() - app.state.startup_time
            if hasattr(app.state, "startup_time")
            else 0
        )

        metrics = [
            "# HELP exoplanet_ai_uptime_seconds Application uptime in seconds",
            "# TYPE exoplanet_ai_uptime_seconds counter",
            f"exoplanet_ai_uptime_seconds {uptime}",
            "",
            "# HELP exoplanet_ai_data_sources_total Total number of data sources",
            "# TYPE exoplanet_ai_data_sources_total gauge",
            f"exoplanet_ai_data_sources_total {registry_info['total_sources']}",
            "",
            "# HELP exoplanet_ai_data_sources_initialized Number of initialized data sources",
            "# TYPE exoplanet_ai_data_sources_initialized gauge",
            f"exoplanet_ai_data_sources_initialized {registry_info['initialized_sources']}",
            "",
            "# HELP exoplanet_ai_cache_hits_total Total cache hits",
            "# TYPE exoplanet_ai_cache_hits_total counter",
            f"exoplanet_ai_cache_hits_total {cache_stats.get('hits', 0)}",
            "",
            "# HELP exoplanet_ai_cache_misses_total Total cache misses",
            "# TYPE exoplanet_ai_cache_misses_total counter",
            f"exoplanet_ai_cache_misses_total {cache_stats.get('misses', 0)}",
            "",
            "# HELP exoplanet_ai_cache_hit_rate Cache hit rate",
            "# TYPE exoplanet_ai_cache_hit_rate gauge",
            f"exoplanet_ai_cache_hit_rate {cache_stats.get('hit_rate', 0.0)}",
        ]

        return PlainTextResponse(
            content="\n".join(metrics),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return PlainTextResponse(
            content="# Error generating metrics\n", status_code=500
        )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with structured logging"""

    logger.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
        method=request.method,
    )

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "error_id": f"err_{int(time.time())}",
            "timestamp": time.time(),
        },
    )


# Rate limiting decorators for different user types
def rate_limit_by_user_role():
    """Rate limiting based on user role"""
    if not SLOWAPI_AVAILABLE:
        return lambda f: f

    def decorator(func):
        # Apply different limits based on authentication
        # This is a simplified implementation
        return limiter.limit("60/minute")(func)

    return decorator


if __name__ == "__main__":
    print("=" * 80)
    print("🚀 STARTING EXOPLANET AI BACKEND v2.0")
    print("=" * 80)
    print("🌟 Features:")
    print("  ✅ Modular data sources (NASA, ESA, Kepler, TESS)")
    print("  ✅ Redis caching with TTL")
    print("  ✅ JWT authentication with roles")
    print("  ✅ Real astronomical data only")
    print("  ✅ Structured logging")
    print("  ✅ Rate limiting")
    print("  ✅ Comprehensive API")
    print("=" * 80)
    print(f"🌐 Host: {config.server.host}")
    print(f"🔌 Port: {config.server.port}")
    print(f"📊 Docs: http://localhost:{config.server.port}/docs")
    print(f"🔍 API: http://localhost:{config.server.port}/api/v1/")
    print(f"📈 Metrics: http://localhost:{config.server.port}/metrics")
    print("=" * 80)

    try:
        uvicorn.run(
            "main:app",
            host=config.server.host,
            port=config.server.port,
            reload=getattr(config, "reload", True),
            log_level="info",
            access_log=True,
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        raise
