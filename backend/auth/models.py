"""
Authentication models for ExoplanetAI
Модели аутентификации
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


class UserRole(str, Enum):
    """User roles"""

    ADMIN = "admin"
    RESEARCHER = "researcher"
    USER = "user"
    GUEST = "guest"


class User(BaseModel):
    """User model"""

    id: str
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    api_key: Optional[str] = None

    # Research-specific fields
    institution: Optional[str] = None
    research_area: Optional[str] = None
    orcid_id: Optional[str] = None

    # Usage limits
    daily_request_limit: int = 1000
    monthly_request_limit: int = 10000

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class UserCreate(BaseModel):
    """User creation model"""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None, max_length=100)
    institution: Optional[str] = Field(None, max_length=200)
    research_area: Optional[str] = Field(None, max_length=200)
    orcid_id: Optional[str] = Field(None, pattern=r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")


class UserLogin(BaseModel):
    """User login model"""

    username: str
    password: str


class TokenData(BaseModel):
    """JWT token data"""

    user_id: str
    username: str
    role: UserRole
    exp: datetime
    iat: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class TokenResponse(BaseModel):
    """Token response model"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: User


class APIKeyCreate(BaseModel):
    """API key creation model"""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    expires_days: Optional[int] = Field(30, ge=1, le=365)


class APIKey(BaseModel):
    """API key model"""

    id: str
    name: str
    description: Optional[str] = None
    key_preview: str  # First 8 chars + "..."
    user_id: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class UserStats(BaseModel):
    """User usage statistics"""

    user_id: str
    requests_today: int = 0
    requests_this_month: int = 0
    total_requests: int = 0
    last_request: Optional[datetime] = None
    favorite_endpoints: List[str] = []

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
