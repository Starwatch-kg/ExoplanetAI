"""
FastAPI dependencies for authentication and authorization
Зависимости FastAPI для аутентификации и авторизации
"""

from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .jwt_auth import JWTManager, get_jwt_manager
from .models import TokenData, User, UserRole

# Security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
) -> Optional[User]:
    """
    Get current authenticated user from JWT token or API key

    Returns None if no valid authentication found
    """
    # Try JWT token first
    if credentials:
        token_data = jwt_manager.verify_token(credentials.credentials)
        if token_data:
            user = jwt_manager.get_user_by_username(token_data.username)
            if user and user.is_active:
                return user

    # Try API key from header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        user = jwt_manager.verify_api_key(api_key)
        if user and user.is_active:
            return user

    return None


async def require_auth(
    current_user: Optional[User] = Depends(get_current_user),
) -> User:
    """
    Require authentication - raises 401 if not authenticated
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return current_user


def require_role(required_role: UserRole):
    """
    Create dependency that requires specific role

    Usage:
        @app.get("/admin")
        async def admin_endpoint(user: User = Depends(require_role(UserRole.ADMIN))):
            ...
    """

    async def role_checker(current_user: User = Depends(require_auth)) -> User:
        # Define role hierarchy
        role_hierarchy = {
            UserRole.GUEST: 0,
            UserRole.USER: 1,
            UserRole.RESEARCHER: 2,
            UserRole.ADMIN: 3,
        }

        user_level = role_hierarchy.get(current_user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)

        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role.value}' or higher required",
            )

        return current_user

    return role_checker


async def get_optional_user(
    current_user: Optional[User] = Depends(get_current_user),
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise
    Useful for endpoints that work both with and without auth
    """
    return current_user


async def require_admin(
    current_user: User = Depends(require_role(UserRole.ADMIN)),
) -> User:
    """Require admin role"""
    return current_user


async def require_researcher(
    current_user: User = Depends(require_role(UserRole.RESEARCHER)),
) -> User:
    """Require researcher role or higher"""
    return current_user
