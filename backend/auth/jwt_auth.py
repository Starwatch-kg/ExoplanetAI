"""
JWT authentication implementation for ExoplanetAI
Реализация JWT аутентификации
"""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
from fastapi import HTTPException, status
from passlib.context import CryptContext

from core.config import config

from .models import TokenData, User, UserRole

logger = logging.getLogger(__name__)


class JWTManager:
    """JWT token manager"""

    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or getattr(
            config, "jwt_secret_key", self._generate_secret_key()
        )
        self.algorithm = "HS256"
        self.access_token_expire_minutes = getattr(
            config, "jwt_expire_minutes", 1440
        )  # 24 hours

        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # In-memory user store (in production, use proper database)
        self._users: Dict[str, User] = {}
        self._user_passwords: Dict[str, str] = {}

        # Create default admin user (disabled for import test)
        # self._create_default_users()

    def _generate_secret_key(self) -> str:
        """Generate a secure secret key"""
        return secrets.token_urlsafe(32)

    def _create_default_users(self):
        """Create default users for testing"""
        # Admin user
        admin_user = User(
            id="admin-001",
            username="admin",
            email="admin@exoplanetai.com",
            full_name="System Administrator",
            role=UserRole.ADMIN,
            created_at=datetime.now(),
            daily_request_limit=10000,
            monthly_request_limit=100000,
            institution="ExoplanetAI",
            research_area="System Administration",
        )

        # Researcher user
        researcher_user = User(
            id="researcher-001",
            username="researcher",
            email="researcher@university.edu",
            full_name="Dr. Jane Smith",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(),
            daily_request_limit=5000,
            monthly_request_limit=50000,
            institution="University Observatory",
            research_area="Exoplanet Detection",
        )

        # Regular user
        regular_user = User(
            id="user-001",
            username="user",
            email="user@example.com",
            full_name="John Doe",
            role=UserRole.USER,
            created_at=datetime.now(),
            institution="Amateur Astronomer",
            research_area="Citizen Science",
        )

        # Store users with default passwords
        self._users["admin"] = admin_user
        self._user_passwords["admin"] = self.hash_password("admin123")

        self._users["researcher"] = researcher_user
        self._user_passwords["researcher"] = self.hash_password("research123")

        self._users["user"] = regular_user
        self._user_passwords["user"] = self.hash_password("user123")

        logger.info("Default users created: admin, researcher, user")

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        if username not in self._users:
            return None

        user = self._users[username]
        if not user.is_active:
            return None

        stored_password = self._user_passwords.get(username)
        if not stored_password:
            return None

        if not self.verify_password(password, stored_password):
            return None

        # Update last login
        user.last_login = datetime.now()
        return user

    def create_access_token(self, user: User) -> Dict[str, Any]:
        """Create JWT access token"""
        now = datetime.utcnow()
        expire = now + timedelta(minutes=self.access_token_expire_minutes)

        token_data = {
            "sub": user.id,
            "username": user.username,
            "role": user.role.value,
            "iat": now,
            "exp": expire,
        }

        encoded_jwt = jwt.encode(token_data, self.secret_key, algorithm=self.algorithm)

        return {
            "access_token": encoded_jwt,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60,
            "user": user,
        }

    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            user_id: str = payload.get("sub")
            username: str = payload.get("username")
            role_str: str = payload.get("role")

            if user_id is None or username is None or role_str is None:
                return None

            # Verify user still exists and is active
            user = self.get_user_by_username(username)
            if not user or not user.is_active:
                return None

            role = UserRole(role_str)
            exp = datetime.fromtimestamp(payload.get("exp"))
            iat = datetime.fromtimestamp(payload.get("iat"))

            return TokenData(
                user_id=user_id, username=username, role=role, exp=exp, iat=iat
            )

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"JWT error: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self._users.get(username)

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        for user in self._users.values():
            if user.id == user_id:
                return user
        return None

    def create_user(self, username: str, email: str, password: str, **kwargs) -> User:
        """Create new user"""
        if username in self._users:
            raise ValueError(f"User '{username}' already exists")

        user_id = f"user-{len(self._users) + 1:03d}"

        user = User(
            id=user_id,
            username=username,
            email=email,
            created_at=datetime.now(),
            **kwargs,
        )

        self._users[username] = user
        self._user_passwords[username] = self.hash_password(password)

        logger.info(f"Created user: {username}")
        return user

    def generate_api_key(self, user: User) -> str:
        """Generate API key for user"""
        # Create API key with user info
        key_data = f"{user.id}:{user.username}:{datetime.now().isoformat()}"
        api_key = hashlib.sha256(key_data.encode()).hexdigest()

        # Store API key (in production, store in database)
        user.api_key = api_key

        return api_key

    def verify_api_key(self, api_key: str) -> Optional[User]:
        """Verify API key and return user"""
        for user in self._users.values():
            if user.api_key == api_key and user.is_active:
                return user
        return None

    def get_all_users(self) -> Dict[str, User]:
        """Get all users (admin only)"""
        return self._users.copy()


# Global JWT manager instance
_jwt_manager = JWTManager()


def get_jwt_manager() -> JWTManager:
    """Get the global JWT manager"""
    return _jwt_manager


def create_access_token(user: User) -> Dict[str, Any]:
    """Convenience function to create access token"""
    return _jwt_manager.create_access_token(user)


def verify_token(token: str) -> Optional[TokenData]:
    """Convenience function to verify token"""
    return _jwt_manager.verify_token(token)
