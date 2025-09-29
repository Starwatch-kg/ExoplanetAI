"""
Error handlers module for Exoplanet AI backend
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
import logging


class ExoplanetAIException(HTTPException):
    """Custom exception for Exoplanet AI application"""
    
    def __init__(self, detail: str, error_code: str = "APPLICATION_ERROR", 
                 status_code: int = 500, **kwargs):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.details = kwargs.get("details", {})
    
    def to_dict(self):
        return {
            "error_code": self.error_code,
            "detail": self.detail,
            "status_code": self.status_code,
            "details": self.details
        }


def setup_error_handlers(app: FastAPI):
    """Setup error handlers for the application"""
    
    @app.exception_handler(ExoplanetAIException)
    async def handle_exoplanet_ai_exception(request: Request, exc: ExoplanetAIException):
        """Handle custom Exoplanet AI exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "ExoplanetAIException",
                "error_code": exc.error_code,
                "detail": exc.detail,
                "details": exc.details,
                "request_id": getattr(request.state, 'request_id', None)
            }
        )
    
    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTPException",
                "status_code": exc.status_code,
                "detail": exc.detail,
                "request_id": getattr(request.state, 'request_id', None)
            }
        )
    
    @app.exception_handler(Exception)
    async def handle_general_exception(request: Request, exc: Exception):
        """Handle general exceptions"""
        logging.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "status_code": 500,
                "detail": "An internal server error occurred",
                "request_id": getattr(request.state, 'request_id', None)
            }
        )


# Example usage of the exception
def create_error_response(error_code: str, detail: str, status_code: int = 400) -> Dict[str, Any]:
    """Create a standard error response"""
    return {
        "error_code": error_code,
        "detail": detail,
        "status_code": status_code
    }