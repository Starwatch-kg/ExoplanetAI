"""
Enterprise-grade input validation system for ExoplanetAI
Система валидации входных данных уровня enterprise для ExoplanetAI
"""

import re
import json
import html
import urllib.parse
from typing import Any, Dict, List, Optional, Union, Set
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import logging

logger = logging.getLogger(__name__)


class TableType(str, Enum):
    """Allowed table types for data ingestion"""
    KOI = "koi"
    TOI = "toi"
    K2 = "k2"


class MissionType(str, Enum):
    """Allowed mission types"""
    TESS = "TESS"
    KEPLER = "Kepler"
    K2 = "K2"


class DataType(str, Enum):
    """Allowed data types for validation"""
    KOI = "koi"
    TOI = "toi"
    K2 = "k2"
    LIGHTCURVE = "lightcurve"


class ProcessingMethod(str, Enum):
    """Allowed preprocessing methods"""
    MEDIAN_FILTER = "median_filter"
    SAVGOL_FILTER = "savgol_filter"
    WAVELET_DENOISING = "wavelet_denoising"
    OUTLIER_REMOVAL = "outlier_removal"
    NORMALIZATION = "normalization"


class SecurityValidator:
    """Security-focused input validation utilities"""
    
    # Dangerous patterns that could indicate injection attacks
    DANGEROUS_PATTERNS = [
        # SQL injection patterns
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(\b(UNION|OR|AND)\s+\d+\s*=\s*\d+)",
        r"(--|#|/\*|\*/)",
        r"(\bxp_cmdshell\b|\bsp_executesql\b)",
        
        # NoSQL injection patterns
        r"(\$where|\$ne|\$gt|\$lt|\$regex)",
        r"({\s*\$\w+\s*:)",
        
        # Command injection patterns
        r"(;|\||&|`|\$\(|\${)",
        r"(\b(rm|del|format|shutdown|reboot|kill)\b)",
        
        # Script injection patterns
        r"(<script|</script>|javascript:|vbscript:|onload=|onerror=)",
        r"(eval\s*\(|setTimeout\s*\(|setInterval\s*\()",
        
        # Path traversal patterns
        r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c)",
        r"(/etc/passwd|/proc/|C:\\Windows\\)",
        
        # LDAP injection patterns
        r"(\*\)|\(\||\)\(|\(\&)",
        
        # XML injection patterns
        r"(<!ENTITY|<!DOCTYPE|<\?xml)",
        
        # File inclusion patterns
        r"(file://|ftp://|data:)",
    ]
    
    # Compiled regex patterns for performance
    COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in DANGEROUS_PATTERNS]
    
    # Characters that are never allowed in any input
    FORBIDDEN_CHARS = {'\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07', 
                      '\x08', '\x0b', '\x0c', '\x0e', '\x0f', '\x10', '\x11', '\x12', 
                      '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a', 
                      '\x1b', '\x1c', '\x1d', '\x1e', '\x1f', '\x7f'}
    
    @classmethod
    def is_safe_string(cls, value: str, allow_html: bool = False) -> bool:
        """
        Check if string is safe from injection attacks
        
        Args:
            value: String to validate
            allow_html: Whether to allow HTML tags
            
        Returns:
            bool: True if string is safe
        """
        if not isinstance(value, str):
            return False
        
        # Check for forbidden control characters
        if any(char in cls.FORBIDDEN_CHARS for char in value):
            logger.warning(f"Forbidden control characters detected in input")
            return False
        
        # Check for dangerous patterns
        for pattern in cls.COMPILED_PATTERNS:
            if pattern.search(value):
                logger.warning(f"Dangerous pattern detected: {pattern.pattern}")
                return False
        
        # If HTML is not allowed, check for HTML tags
        if not allow_html and ('<' in value or '>' in value):
            # More strict HTML detection
            html_pattern = re.compile(r'<[^>]*>', re.IGNORECASE)
            if html_pattern.search(value):
                logger.warning("HTML tags detected in input where not allowed")
                return False
        
        return True
    
    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 1000) -> str:
        """
        Sanitize string by removing/escaping dangerous content
        
        Args:
            value: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            str: Sanitized string
        """
        if not isinstance(value, str):
            return str(value)[:max_length]
        
        # Remove forbidden characters
        sanitized = ''.join(char for char in value if char not in cls.FORBIDDEN_CHARS)
        
        # HTML escape
        sanitized = html.escape(sanitized)
        
        # URL decode to prevent double encoding attacks
        try:
            sanitized = urllib.parse.unquote(sanitized)
        except:
            pass  # Keep original if URL decoding fails
        
        # Truncate to max length
        sanitized = sanitized[:max_length]
        
        # Strip whitespace
        sanitized = sanitized.strip()
        
        return sanitized
    
    @classmethod
    def validate_target_name(cls, value: str) -> str:
        """Validate astronomical target name"""
        if not value or not isinstance(value, str):
            raise ValueError("Target name is required")
        
        # Clean the input
        value = value.strip()
        
        # Check length
        if len(value) < 1:
            raise ValueError("Target name cannot be empty")
        if len(value) > 100:
            raise ValueError("Target name too long (max 100 characters)")
        
        # Check for safe characters only
        if not cls.is_safe_string(value):
            raise ValueError("Target name contains unsafe characters")
        
        # Astronomical target name pattern (letters, numbers, spaces, hyphens, dots, plus)
        pattern = re.compile(r'^[A-Za-z0-9\s\-\.\+]+$')
        if not pattern.match(value):
            raise ValueError("Target name contains invalid characters. Allowed: letters, numbers, spaces, hyphens, dots, plus signs")
        
        return value
    
    @classmethod
    def validate_json_params(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """Validate JSON parameters for safety"""
        if not isinstance(value, dict):
            raise ValueError("Parameters must be a dictionary")
        
        # Check depth to prevent deeply nested attacks
        def check_depth(obj, current_depth=0, max_depth=10):
            if current_depth > max_depth:
                raise ValueError("JSON too deeply nested")
            
            if isinstance(obj, dict):
                for v in obj.values():
                    check_depth(v, current_depth + 1, max_depth)
            elif isinstance(obj, list):
                for item in obj:
                    check_depth(item, current_depth + 1, max_depth)
        
        check_depth(value)
        
        # Check size
        json_str = json.dumps(value)
        if len(json_str) > 10000:  # 10KB limit
            raise ValueError("JSON parameters too large (max 10KB)")
        
        # Validate string values in the JSON
        def validate_json_strings(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(k, str) and not cls.is_safe_string(k):
                        raise ValueError(f"Unsafe key in JSON: {k}")
                    validate_json_strings(v)
            elif isinstance(obj, list):
                for item in obj:
                    validate_json_strings(item)
            elif isinstance(obj, str):
                if not cls.is_safe_string(obj):
                    raise ValueError(f"Unsafe string value in JSON: {obj[:50]}...")
        
        validate_json_strings(value)
        
        return value


class SafeString(str):
    """A string type that is validated for security"""
    
    @classmethod
    def __get_validators__(cls):
        yield str
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not SecurityValidator.is_safe_string(v):
            raise ValueError('String contains unsafe content')
        return cls(v)


class TargetName(str):
    """Validated astronomical target name"""
    
    @classmethod
    def __get_validators__(cls):
        yield str
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        return cls(SecurityValidator.validate_target_name(v))


# Enhanced Pydantic models with enterprise-grade validation

class DataIngestionRequest(BaseModel):
    """Enhanced request model for data ingestion with security validation"""
    table_type: TableType = Field(..., description="Type of table to ingest")
    force_refresh: bool = Field(False, description="Force refresh even if cached")
    
    class Config:
        use_enum_values = True
        validate_assignment = True


class LightCurveIngestionRequest(BaseModel):
    """Enhanced request model for light curve ingestion with security validation"""
    target_name: TargetName = Field(..., description="Target identifier (validated)")
    mission: MissionType = Field(MissionType.TESS, description="Mission name")
    sector_quarter: Optional[int] = Field(None, ge=1, le=1000, description="Specific sector/quarter")
    force_refresh: bool = Field(False, description="Force refresh even if cached")
    
    @field_validator('sector_quarter')
    @classmethod
    def validate_sector_quarter(cls, v):
        if v is not None and (v < 1 or v > 1000):
            raise ValueError('Sector/quarter must be between 1 and 1000')
        return v
    
    class Config:
        use_enum_values = True
        validate_assignment = True


class PreprocessingRequest(BaseModel):
    """Enhanced request model for light curve preprocessing with security validation"""
    target_name: TargetName = Field(..., description="Target identifier (validated)")
    mission: MissionType = Field(MissionType.TESS, description="Mission name")
    processing_params: Optional[Dict[str, Any]] = Field(None, description="Custom processing parameters")
    methods: Optional[List[ProcessingMethod]] = Field(None, description="Processing methods to apply")
    
    @field_validator('processing_params')
    @classmethod
    def validate_processing_params(cls, v):
        if v is not None:
            return SecurityValidator.validate_json_params(v)
        return v
    
    @field_validator('methods')
    @classmethod
    def validate_methods(cls, v):
        if v is not None:
            if len(v) > 10:
                raise ValueError('Too many processing methods (max 10)')
            # Remove duplicates while preserving order
            seen = set()
            unique_methods = []
            for method in v:
                if method not in seen:
                    seen.add(method)
                    unique_methods.append(method)
            return unique_methods
        return v
    
    class Config:
        use_enum_values = True
        validate_assignment = True


class VersionCreateRequest(BaseModel):
    """Enhanced request model for creating data version with security validation"""
    version_name: SafeString = Field(..., min_length=1, max_length=50, description="Version identifier")
    description: SafeString = Field(..., min_length=1, max_length=500, description="Version description")
    file_patterns: List[SafeString] = Field(["*.csv", "*.fits"], description="File patterns to include")
    
    @field_validator('version_name')
    @classmethod
    def validate_version_name(cls, v):
        # Version name should be alphanumeric with hyphens and underscores only
        pattern = re.compile(r'^[A-Za-z0-9\-_\.]+$')
        if not pattern.match(v):
            raise ValueError('Version name can only contain letters, numbers, hyphens, underscores, and dots')
        return v
    
    @field_validator('file_patterns')
    @classmethod
    def validate_file_patterns(cls, v):
        if len(v) > 20:
            raise ValueError('Too many file patterns (max 20)')
        
        for pattern in v:
            # Basic validation for file patterns
            if len(pattern) > 100:
                raise ValueError('File pattern too long (max 100 characters)')
            
            # Check for dangerous patterns in file paths
            if not SecurityValidator.is_safe_string(pattern):
                raise ValueError(f'Unsafe file pattern: {pattern}')
        
        return v
    
    class Config:
        validate_assignment = True


class DataValidationRequest(BaseModel):
    """Enhanced request model for data validation with security validation"""
    data_type: DataType = Field(..., description="Type of data to validate")
    target_name: Optional[TargetName] = Field(None, description="Target name for lightcurve validation")
    validation_params: Optional[Dict[str, Any]] = Field(None, description="Custom validation parameters")
    
    @field_validator('validation_params')
    @classmethod
    def validate_validation_params(cls, v):
        if v is not None:
            return SecurityValidator.validate_json_params(v)
        return v
    
    class Config:
        use_enum_values = True
        validate_assignment = True


class FileUploadRequest(BaseModel):
    """Request model for secure file uploads"""
    filename: SafeString = Field(..., min_length=1, max_length=255, description="Original filename")
    content_type: SafeString = Field(..., description="MIME content type")
    file_size: int = Field(..., ge=1, le=100*1024*1024, description="File size in bytes (max 100MB)")
    checksum: Optional[str] = Field(None, pattern=r'^[a-fA-F0-9]{64}$', description="SHA-256 checksum")
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v):
        # Filename validation
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError('Filename cannot contain path traversal characters')
        
        # Check for executable extensions
        dangerous_extensions = {'.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js', '.jar', '.sh'}
        file_ext = '.' + v.split('.')[-1].lower() if '.' in v else ''
        if file_ext in dangerous_extensions:
            raise ValueError(f'File type not allowed: {file_ext}')
        
        return v
    
    @field_validator('content_type')
    @classmethod
    def validate_content_type(cls, v):
        # Allowed MIME types
        allowed_types = {
            'application/fits',
            'text/csv',
            'application/json',
            'text/plain',
            'application/octet-stream'
        }
        
        if v not in allowed_types:
            raise ValueError(f'Content type not allowed: {v}')
        
        return v
    
    class Config:
        validate_assignment = True


# Validation utilities

def validate_pagination_params(page: int = 1, size: int = 20) -> tuple[int, int]:
    """Validate pagination parameters"""
    if page < 1:
        raise ValueError("Page number must be >= 1")
    if page > 10000:
        raise ValueError("Page number too large (max 10000)")
    
    if size < 1:
        raise ValueError("Page size must be >= 1")
    if size > 1000:
        raise ValueError("Page size too large (max 1000)")
    
    return page, size


def validate_search_query(query: str) -> str:
    """Validate search query string"""
    if not query or not isinstance(query, str):
        raise ValueError("Search query is required")
    
    query = query.strip()
    
    if len(query) < 1:
        raise ValueError("Search query cannot be empty")
    if len(query) > 500:
        raise ValueError("Search query too long (max 500 characters)")
    
    if not SecurityValidator.is_safe_string(query):
        raise ValueError("Search query contains unsafe characters")
    
    return query


def validate_date_range(start_date: Optional[str], end_date: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Validate date range parameters"""
    import datetime
    
    if start_date:
        try:
            datetime.datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError("Invalid start_date format. Use ISO 8601 format")
    
    if end_date:
        try:
            datetime.datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError("Invalid end_date format. Use ISO 8601 format")
    
    if start_date and end_date:
        start = datetime.datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end = datetime.datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        if start >= end:
            raise ValueError("start_date must be before end_date")
        
        # Limit date range to prevent abuse
        if (end - start).days > 3650:  # 10 years
            raise ValueError("Date range too large (max 10 years)")
    
    return start_date, end_date
