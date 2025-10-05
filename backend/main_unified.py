"""
ExoplanetAI Unified Backend - Serves React Frontend + API
Unified deployment serving both React frontend and FastAPI backend
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

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
    logging.warning("slowapi not available, rate limiting disabled")

# Configure structured logging
configure_structlog()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Starting ExoplanetAI Backend v2.0...")
    
    # Initialize cache
    await initialize_cache()
    logger.info("✅ Cache initialized")
    
    # Initialize data sources
    registry = get_registry()
    await initialize_default_sources(registry)
    logger.info("✅ Data sources initialized")
    
    # Initialize JWT manager
    jwt_manager = get_jwt_manager()
    logger.info("✅ JWT manager initialized")
    
    logger.info("🎉 ExoplanetAI Backend v2.0 started successfully!")
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down ExoplanetAI Backend...")
    await cleanup_cache()
    logger.info("✅ Cleanup completed")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="ExoplanetAI API",
        description="Advanced Exoplanet Detection and Analysis Platform",
        version="2.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.security.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Structured logging middleware
    app.add_middleware(StructuredLoggingMiddleware)
    
    # Rate limiting (if available)
    if SLOWAPI_AVAILABLE:
        limiter = Limiter(key_func=get_remote_address)
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        logger.info("✅ Rate limiting enabled")
    
    # API routes
    api_router = create_api_router()
    app.include_router(api_router, prefix="/api/v1")
    
    # Health check endpoint (outside API prefix)
    @app.get("/health")
    async def health_check():
        """Simple health check"""
        return {"status": "healthy", "timestamp": time.time()}
    
    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.error(f"HTTP {exc.status_code}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "status_code": exc.status_code}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "status_code": 500}
        )
    
    # Serve React frontend (if build exists)
    frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        logger.info(f"✅ Serving React frontend from {frontend_dist}")
        
        # Mount static assets first (with cache headers)
        app.mount(
            "/assets", 
            StaticFiles(directory=frontend_dist / "assets"), 
            name="assets"
        )
        
        # Mount React app (catch-all for SPA routing)
        app.mount(
            "/", 
            StaticFiles(directory=frontend_dist, html=True), 
            name="frontend"
        )
    else:
        logger.warning(f"❌ Frontend build not found at {frontend_dist}")
        logger.info("💡 Run 'npm run build' in frontend directory to enable unified deployment")
        
        # Serve a simple message for missing frontend
        @app.get("/")
        async def root():
            return {
                "message": "ExoplanetAI Backend v2.0",
                "status": "running",
                "frontend": "not_built",
                "api_docs": "/api/docs",
                "health": "/health"
            }
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main_unified:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        log_level="info",
        access_log=True,
    )
