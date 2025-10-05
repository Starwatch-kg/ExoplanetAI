"""
Security utilities for ExoplanetAI
Утилиты безопасности для ExoplanetAI
"""

import os
from typing import List
from pathlib import Path


def get_allowed_origins() -> List[str]:
    """Get allowed CORS origins based on environment"""
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        return [
            "https://exoplanetai.com",
            "https://www.exoplanetai.com",
            os.getenv("FRONTEND_URL", "https://exoplanetai.com")
        ]
    else:
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8080",
            os.getenv("FRONTEND_URL", "http://localhost:3000")
        ]


def validate_file_path(base_path: str, file_path: str) -> Path:
    """
    Validate file path to prevent directory traversal attacks
    
    Args:
        base_path: Base directory path
        file_path: User-provided file path
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path traversal is detected
    """
    base = Path(base_path).resolve()
    full_path = (base / file_path).resolve()
    
    # Check if the resolved path is within the base directory
    if not str(full_path).startswith(str(base)):
        raise ValueError(f"Path traversal attempt detected: {file_path}")
    
    return full_path


def sanitize_log_data(data: dict) -> dict:
    """
    Remove sensitive information from log data
    
    Args:
        data: Dictionary to sanitize
        
    Returns:
        Sanitized dictionary
    """
    sensitive_keys = [
        'api_key', 'token', 'password', 'secret', 'auth',
        'authorization', 'jwt', 'session', 'cookie'
    ]
    
    sanitized = {}
    for key, value in data.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            sanitized[key] = '***REDACTED***'
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_data(value)
        else:
            sanitized[key] = value
    
    return sanitized


def validate_api_input(data: dict, required_fields: List[str]) -> bool:
    """
    Validate API input data
    
    Args:
        data: Input data to validate
        required_fields: List of required field names
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(data, dict):
        return False
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            return False
    
    # Basic sanitization checks
    for key, value in data.items():
        if isinstance(value, str):
            # Check for potential injection attempts
            dangerous_patterns = ['<script', 'javascript:', 'data:', '../', '..\\']
            if any(pattern in value.lower() for pattern in dangerous_patterns):
                return False
    
    return True
