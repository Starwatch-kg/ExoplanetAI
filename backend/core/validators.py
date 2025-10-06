"""
Centralized input validation and security checks
Централизованная валидация входных данных и проверки безопасности
"""

import html
import re
from typing import Any, Dict, List, Optional, Union

from fastapi import HTTPException
from pydantic import BaseModel, Field, validator


class SecurityError(Exception):
    """Raised when security validation fails"""

    pass


def sanitize_string(value: str, max_length: int = 100) -> str:
    """Sanitize string input to prevent XSS and injection attacks"""
    if not isinstance(value, str):
        raise ValueError("Value must be a string")

    # Remove null bytes and control characters
    value = value.replace("\x00", "").replace("\r", "").replace("\n", " ")

    # HTML escape to prevent XSS
    value = html.escape(value.strip())

    # Length validation
    if len(value) > max_length:
        raise ValueError(f"String too long (max {max_length} characters)")

    if len(value) == 0:
        raise ValueError("String cannot be empty")

    return value


def validate_target_name(target_name: str) -> str:
    """Validate astronomical target name with strict regex"""
    if not isinstance(target_name, str):
        raise ValueError("Target name must be a string")

    # Sanitize first
    target_name = sanitize_string(target_name, max_length=50)

    # Strict regex for astronomical target names
    # Allows: TIC 123456, KIC-123456, TOI-123.01, Kepler-452b, HD 209458, etc.
    pattern = r"^[A-Za-z0-9][A-Za-z0-9\s\-\.]{1,48}[A-Za-z0-9]$"

    if not re.match(pattern, target_name):
        raise ValueError(f"Invalid target name format: {target_name}")

    # Additional checks for common patterns
    if any(char in target_name for char in ["<", ">", '"', "'", "&", ";", "|", "`"]):
        raise ValueError("Target name contains invalid characters")

    return target_name


def validate_catalog_id(catalog: str, target_id: str) -> str:
    """Validate catalog-specific target ID format"""
    catalog = catalog.upper()

    if catalog == "TIC":
        # TESS Input Catalog: numeric ID
        if not re.match(r"^\d{1,10}$", target_id):
            raise ValueError("TIC ID must be numeric (1-10 digits)")
    elif catalog == "KIC":
        # Kepler Input Catalog: numeric ID
        if not re.match(r"^\d{1,10}$", target_id):
            raise ValueError("KIC ID must be numeric (1-10 digits)")
    elif catalog == "EPIC":
        # K2 catalog: numeric ID
        if not re.match(r"^\d{1,10}$", target_id):
            raise ValueError("EPIC ID must be numeric (1-10 digits)")
    else:
        raise ValueError(f"Unsupported catalog: {catalog}")

    return target_id


def validate_catalog(catalog: str) -> str:
    """Validate catalog name"""
    catalog = sanitize_string(catalog, max_length=10).upper()

    allowed_catalogs = {"TIC", "KIC", "EPIC"}
    if catalog not in allowed_catalogs:
        raise ValueError(f"Invalid catalog. Allowed: {allowed_catalogs}")

    return catalog


def validate_mission(mission: str) -> str:
    """Validate mission name"""
    mission = sanitize_string(mission, max_length=20)

    allowed_missions = {"TESS", "Kepler", "K2"}
    if mission not in allowed_missions:
        raise ValueError(f"Invalid mission. Allowed: {allowed_missions}")

    return mission


def validate_numeric_range(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float],
    name: str = "value",
) -> Union[int, float]:
    """Validate numeric value is within allowed range"""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")

    if value < min_val or value > max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}")

    return value


def validate_array_size(
    data: List[Any], max_size: int = 100000, name: str = "array"
) -> List[Any]:
    """Validate array size to prevent memory exhaustion"""
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a list")

    if len(data) > max_size:
        raise ValueError(f"{name} too large (max {max_size} elements)")

    if len(data) == 0:
        raise ValueError(f"{name} cannot be empty")

    return data


def validate_json_size(
    data: Dict[str, Any], max_size_mb: float = 10.0
) -> Dict[str, Any]:
    """Validate JSON payload size"""
    import json
    import sys

    # Estimate size
    json_str = json.dumps(data)
    size_mb = sys.getsizeof(json_str) / (1024 * 1024)

    if size_mb > max_size_mb:
        raise ValueError(
            f"JSON payload too large ({size_mb:.1f}MB, max {max_size_mb}MB)"
        )

    return data


class SecureBaseModel(BaseModel):
    """Base model with security validations"""

    class Config:
        # Prevent extra fields
        extra = "forbid"
        # Validate assignment
        validate_assignment = True
        # Use enum values
        use_enum_values = True


class ValidatedSearchRequest(SecureBaseModel):
    """Validated search request with security checks"""

    target_name: str = Field(..., min_length=1, max_length=50)
    catalog: str = Field("TIC", regex="^(TIC|KIC|EPIC)$")
    mission: str = Field("TESS", regex="^(TESS|Kepler|K2)$")
    period_min: float = Field(0.5, ge=0.1, le=100.0)
    period_max: float = Field(20.0, ge=0.1, le=100.0)
    snr_threshold: float = Field(7.0, ge=3.0, le=20.0)

    @validator("target_name")
    def validate_target_name_field(cls, v):
        return validate_target_name(v)

    @validator("catalog")
    def validate_catalog_field(cls, v):
        return validate_catalog(v)

    @validator("mission")
    def validate_mission_field(cls, v):
        return validate_mission(v)

    @validator("period_max")
    def validate_period_range(cls, v, values):
        if "period_min" in values and v <= values["period_min"]:
            raise ValueError("period_max must be greater than period_min")
        return v


class ValidatedGPIRequest(SecureBaseModel):
    """Validated GPI request with security checks"""

    target_name: str = Field(..., min_length=1, max_length=50)
    use_ai: bool = Field(True)
    phase_sensitivity: Optional[float] = Field(1e-12, ge=1e-15, le=1e-9)
    snr_threshold: Optional[float] = Field(5.0, ge=3.0, le=20.0)
    period_min: Optional[float] = Field(0.1, ge=0.01, le=1000.0)
    period_max: Optional[float] = Field(1000.0, ge=0.01, le=10000.0)

    @validator("target_name")
    def validate_target_name_field(cls, v):
        return validate_target_name(v)


def validate_request_rate_limit(
    client_ip: str, endpoint: str, max_requests: int = 60
) -> bool:
    """Simple rate limiting validation (in-memory)"""
    import time
    from collections import defaultdict

    # In production, use Redis or proper rate limiting middleware
    _rate_limit_cache = defaultdict(list)

    now = time.time()
    key = f"{client_ip}:{endpoint}"

    # Clean old entries
    _rate_limit_cache[key] = [
        timestamp
        for timestamp in _rate_limit_cache[key]
        if now - timestamp < 3600  # 1 hour window
    ]

    # Check rate limit
    if len(_rate_limit_cache[key]) >= max_requests:
        return False

    # Add current request
    _rate_limit_cache[key].append(now)
    return True


def create_error_response(
    error_code: str, message: str, details: Optional[Dict] = None
) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "status": "error",
        "error": {"code": error_code, "message": message, "details": details or {}},
        "data": None,
    }


def create_success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
    """Create standardized success response"""
    return {"status": "ok", "message": message, "data": data, "error": None}
