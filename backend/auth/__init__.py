"""
Authentication and authorization module for ExoplanetAI
Модуль аутентификации и авторизации
"""

from .dependencies import get_current_user, require_auth, require_role
from .jwt_auth import JWTManager, create_access_token, get_jwt_manager, verify_token
from .models import TokenData, User, UserRole

__all__ = [
    "JWTManager",
    "get_jwt_manager",
    "create_access_token",
    "verify_token",
    "User",
    "UserRole",
    "TokenData",
    "get_current_user",
    "require_auth",
    "require_role",
]
