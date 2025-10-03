"""
Authentication API routes
Маршруты API аутентификации
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer

from auth.dependencies import get_current_user, require_auth
from auth.jwt_auth import JWTManager, get_jwt_manager
from auth.models import APIKey, APIKeyCreate, TokenResponse, User, UserCreate, UserLogin
from schemas.response import ErrorCode, create_error_response, create_success_response

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()


@router.post("/login", response_model=dict)
async def login(
    user_credentials: UserLogin, jwt_manager: JWTManager = Depends(get_jwt_manager)
):
    """
    Authenticate user and return JWT token

    **Default users for testing:**
    - admin / admin123 (Admin role)
    - researcher / research123 (Researcher role)
    - user / user123 (User role)
    """
    try:
        # Authenticate user
        user = jwt_manager.authenticate_user(
            user_credentials.username, user_credentials.password
        )

        if not user:
            return create_error_response(
                ErrorCode.VALIDATION_ERROR, "Invalid username or password"
            )

        # Create access token
        token_data = jwt_manager.create_access_token(user)

        logger.info(f"User '{user.username}' logged in successfully")

        return create_success_response(data=token_data, message="Login successful")

    except Exception as e:
        logger.error(f"Login error: {e}")
        return create_error_response(ErrorCode.INTERNAL_ERROR, "Login failed")


@router.post("/register", response_model=dict)
async def register(
    user_data: UserCreate, jwt_manager: JWTManager = Depends(get_jwt_manager)
):
    """
    Register new user account
    """
    try:
        # Check if user already exists
        existing_user = jwt_manager.get_user_by_username(user_data.username)
        if existing_user:
            return create_error_response(
                ErrorCode.VALIDATION_ERROR,
                f"Username '{user_data.username}' already exists",
            )

        # Create new user
        new_user = jwt_manager.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            institution=user_data.institution,
            research_area=user_data.research_area,
            orcid_id=user_data.orcid_id,
        )

        # Create access token
        token_data = jwt_manager.create_access_token(new_user)

        logger.info(f"New user registered: {user_data.username}")

        return create_success_response(
            data=token_data, message="Registration successful"
        )

    except ValueError as e:
        return create_error_response(ErrorCode.VALIDATION_ERROR, str(e))
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return create_error_response(ErrorCode.INTERNAL_ERROR, "Registration failed")


@router.get("/me", response_model=dict)
async def get_current_user_info(current_user: User = Depends(require_auth)):
    """
    Get current user information
    """
    return create_success_response(
        data=current_user.dict(), message="User information retrieved"
    )


@router.post("/api-key", response_model=dict)
async def create_api_key(
    api_key_data: APIKeyCreate,
    current_user: User = Depends(require_auth),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
):
    """
    Create API key for current user
    """
    try:
        # Generate API key
        api_key = jwt_manager.generate_api_key(current_user)

        # Create API key info (in production, store in database)
        api_key_info = APIKey(
            id=f"key-{current_user.id}-001",
            name=api_key_data.name,
            description=api_key_data.description,
            key_preview=f"{api_key[:8]}...",
            user_id=current_user.id,
            created_at=current_user.created_at,
        )

        logger.info(f"API key created for user: {current_user.username}")

        return create_success_response(
            data={"api_key": api_key, "key_info": api_key_info.dict()},
            message="API key created successfully",
        )

    except Exception as e:
        logger.error(f"API key creation error: {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR, "Failed to create API key"
        )


@router.post("/logout", response_model=dict)
async def logout(current_user: User = Depends(require_auth)):
    """
    Logout user (client should discard token)
    """
    logger.info(f"User '{current_user.username}' logged out")

    return create_success_response(message="Logout successful")


@router.get("/test-auth", response_model=dict)
async def test_authentication(current_user: User = Depends(require_auth)):
    """
    Test endpoint to verify authentication is working
    """
    return create_success_response(
        data={
            "authenticated": True,
            "user": current_user.username,
            "role": current_user.role.value,
        },
        message="Authentication test successful",
    )
