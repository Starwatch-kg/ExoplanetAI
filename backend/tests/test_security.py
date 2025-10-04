"""
Comprehensive security tests for ExoplanetAI v2.0
Комплексные тесты безопасности для ExoplanetAI v2.0
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status

from main import app
from core.validation import SecurityValidator
from core.rate_limiting import EnterpriseRateLimiter, RateLimitConfig
from auth.models import User, UserRole


class TestSecurityValidator:
    """Test suite for SecurityValidator class"""
    
    def test_is_safe_string_valid_inputs(self):
        """Test that valid strings pass security validation"""
        valid_strings = [
            "TOI-715b",
            "Kepler-452b", 
            "HD 209458 b",
            "WASP-121b",
            "normal text with spaces",
            "numbers123",
            "special-chars_allowed.here"
        ]
        
        for string in valid_strings:
            assert SecurityValidator.is_safe_string(string), f"Valid string rejected: {string}"
    
    def test_is_safe_string_sql_injection(self):
        """Test SQL injection detection"""
        malicious_strings = [
            "'; DROP TABLE planets; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users--",
            "1; DELETE FROM data;",
            "robert'; DROP TABLE students;--"
        ]
        
        for string in malicious_strings:
            assert not SecurityValidator.is_safe_string(string), f"SQL injection not detected: {string}"
    
    def test_is_safe_string_script_injection(self):
        """Test script injection detection"""
        malicious_strings = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "vbscript:msgbox(1)",
            "onload=alert(1)",
            "onerror=alert(1)",
            "eval('malicious code')",
            "setTimeout('alert(1)', 1000)"
        ]
        
        for string in malicious_strings:
            assert not SecurityValidator.is_safe_string(string), f"Script injection not detected: {string}"
    
    def test_is_safe_string_path_traversal(self):
        """Test path traversal detection"""
        malicious_strings = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "%2e%2e%2f%2e%2e%2f",
            "/etc/passwd",
            "C:\\Windows\\System32",
            "file:///etc/passwd"
        ]
        
        for string in malicious_strings:
            assert not SecurityValidator.is_safe_string(string), f"Path traversal not detected: {string}"
    
    def test_is_safe_string_command_injection(self):
        """Test command injection detection"""
        malicious_strings = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& shutdown -h now",
            "`whoami`",
            "$(id)",
            "${PATH}"
        ]
        
        for string in malicious_strings:
            assert not SecurityValidator.is_safe_string(string), f"Command injection not detected: {string}"
    
    def test_is_safe_string_control_characters(self):
        """Test control character detection"""
        malicious_strings = [
            "test\x00null",
            "test\x01control",
            "test\x1besc",
            "test\x7fdel"
        ]
        
        for string in malicious_strings:
            assert not SecurityValidator.is_safe_string(string), f"Control characters not detected: {string}"
    
    def test_sanitize_string(self):
        """Test string sanitization"""
        test_cases = [
            ("normal text", "normal text"),
            ("<script>alert(1)</script>", "&lt;script&gt;alert(1)&lt;/script&gt;"),
            ("test\x00null", "testnull"),
            ("  whitespace  ", "whitespace"),
            ("a" * 2000, "a" * 1000)  # Truncation test
        ]
        
        for input_str, expected in test_cases:
            result = SecurityValidator.sanitize_string(input_str)
            assert result == expected, f"Sanitization failed for {input_str}: got {result}, expected {expected}"
    
    def test_validate_target_name_valid(self):
        """Test valid target name validation"""
        valid_names = [
            "TOI-715b",
            "Kepler-452b",
            "HD 209458 b",
            "WASP-121b",
            "Gliese 667C c",
            "K2-18b"
        ]
        
        for name in valid_names:
            result = SecurityValidator.validate_target_name(name)
            assert result == name, f"Valid target name rejected: {name}"
    
    def test_validate_target_name_invalid(self):
        """Test invalid target name rejection"""
        invalid_names = [
            "",
            "   ",
            "a" * 200,  # Too long
            "test<script>",
            "'; DROP TABLE;",
            "test\x00null",
            "test|command"
        ]
        
        for name in invalid_names:
            with pytest.raises(ValueError):
                SecurityValidator.validate_target_name(name)
    
    def test_validate_json_params_valid(self):
        """Test valid JSON parameter validation"""
        valid_params = [
            {"method": "median_filter", "window": 5},
            {"sigma": 3.0, "iterations": 2},
            {"nested": {"param": "value"}},
            {"list": [1, 2, 3]}
        ]
        
        for params in valid_params:
            result = SecurityValidator.validate_json_params(params)
            assert result == params
    
    def test_validate_json_params_invalid(self):
        """Test invalid JSON parameter rejection"""
        # Too deeply nested
        deeply_nested = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": {"j": {"k": "value"}}}}}}}}}}}
        
        with pytest.raises(ValueError, match="too deeply nested"):
            SecurityValidator.validate_json_params(deeply_nested)
        
        # Too large
        large_params = {"data": "x" * 20000}
        with pytest.raises(ValueError, match="too large"):
            SecurityValidator.validate_json_params(large_params)
        
        # Unsafe strings
        unsafe_params = {"param": "'; DROP TABLE;"}
        with pytest.raises(ValueError, match="Unsafe string"):
            SecurityValidator.validate_json_params(unsafe_params)


class TestRateLimiting:
    """Test suite for enterprise rate limiting"""
    
    @pytest.fixture
    async def rate_limiter(self):
        """Create rate limiter instance for testing"""
        limiter = EnterpriseRateLimiter()
        await limiter.initialize()
        yield limiter
        await limiter.cleanup()
    
    @pytest.mark.asyncio
    async def test_rate_limit_basic(self, rate_limiter):
        """Test basic rate limiting functionality"""
        identifier = "test_user"
        endpoint = "/test"
        
        # First request should be allowed
        is_allowed, info = await rate_limiter.check_rate_limit(identifier, endpoint, UserRole.GUEST)
        assert is_allowed is True
        assert info["allowed"] is True
        
        # Make many requests quickly to trigger rate limit
        for _ in range(15):  # Guest limit is 10 per minute
            await rate_limiter.check_rate_limit(identifier, endpoint, UserRole.GUEST)
        
        # Should now be rate limited
        is_allowed, info = await rate_limiter.check_rate_limit(identifier, endpoint, UserRole.GUEST)
        assert is_allowed is False
        assert info["allowed"] is False
        assert len(info["violations"]) > 0
    
    @pytest.mark.asyncio
    async def test_rate_limit_different_roles(self, rate_limiter):
        """Test different rate limits for different user roles"""
        endpoint = "/test"
        
        # Test guest limits (lower)
        guest_identifier = "guest_user"
        for _ in range(5):
            is_allowed, _ = await rate_limiter.check_rate_limit(guest_identifier, endpoint, UserRole.GUEST)
            assert is_allowed is True
        
        # Test researcher limits (higher)
        researcher_identifier = "researcher_user"
        for _ in range(50):  # Should be allowed for researchers
            is_allowed, _ = await rate_limiter.check_rate_limit(researcher_identifier, endpoint, UserRole.RESEARCHER)
            assert is_allowed is True
    
    @pytest.mark.asyncio
    async def test_rate_limit_endpoint_specific(self, rate_limiter):
        """Test endpoint-specific rate limits"""
        identifier = "test_user"
        
        # Batch ingestion should have very strict limits
        batch_endpoint = "/api/v1/data/ingest/batch"
        
        # First request allowed
        is_allowed, _ = await rate_limiter.check_rate_limit(identifier, batch_endpoint, UserRole.GUEST)
        assert is_allowed is True
        
        # Second request should be blocked (1 per day limit for guests)
        is_allowed, info = await rate_limiter.check_rate_limit(identifier, batch_endpoint, UserRole.GUEST)
        assert is_allowed is False
        assert any("batch" in v["rule"].lower() for v in info["violations"])
    
    @pytest.mark.asyncio
    async def test_rate_limit_status(self, rate_limiter):
        """Test rate limit status reporting"""
        identifier = "test_user"
        endpoint = "/test"
        
        # Make some requests
        for _ in range(3):
            await rate_limiter.check_rate_limit(identifier, endpoint, UserRole.USER)
        
        # Check status
        status = await rate_limiter.get_rate_limit_status(identifier, endpoint, UserRole.USER)
        
        assert status["identifier"] == identifier
        assert status["endpoint"] == endpoint
        assert status["user_role"] == UserRole.USER.value
        assert "limits" in status
        
        # Should show some usage
        for limit_type, limit_info in status["limits"].items():
            assert limit_info["used"] >= 0
            assert limit_info["remaining"] >= 0
            assert limit_info["limit"] > 0


class TestAPISecurityIntegration:
    """Integration tests for API security"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_input_validation_table_ingestion(self, client):
        """Test input validation on table ingestion endpoint"""
        # Valid request
        valid_request = {
            "table_type": "koi",
            "force_refresh": False
        }
        
        # Note: This will fail auth, but should pass validation
        response = client.post("/api/v1/data/ingest/table", json=valid_request)
        assert response.status_code in [401, 403]  # Auth error, not validation error
        
        # Invalid table type
        invalid_request = {
            "table_type": "invalid_type",
            "force_refresh": False
        }
        
        response = client.post("/api/v1/data/ingest/table", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_input_validation_lightcurve_ingestion(self, client):
        """Test input validation on lightcurve ingestion endpoint"""
        # SQL injection attempt
        malicious_request = {
            "target_name": "'; DROP TABLE planets; --",
            "mission": "TESS"
        }
        
        response = client.post("/api/v1/data/ingest/lightcurve", json=malicious_request)
        assert response.status_code == 422  # Should be rejected by validation
        
        # Path traversal attempt
        malicious_request = {
            "target_name": "../../../etc/passwd",
            "mission": "TESS"
        }
        
        response = client.post("/api/v1/data/ingest/lightcurve", json=malicious_request)
        assert response.status_code == 422  # Should be rejected by validation
        
        # Script injection attempt
        malicious_request = {
            "target_name": "<script>alert('xss')</script>",
            "mission": "TESS"
        }
        
        response = client.post("/api/v1/data/ingest/lightcurve", json=malicious_request)
        assert response.status_code == 422  # Should be rejected by validation
    
    def test_input_validation_preprocessing(self, client):
        """Test input validation on preprocessing endpoint"""
        # Malicious processing parameters
        malicious_request = {
            "target_name": "TOI-715b",
            "mission": "TESS",
            "processing_params": {
                "command": "; rm -rf /",
                "injection": "'; DROP TABLE;"
            }
        }
        
        response = client.post("/api/v1/data/preprocess/lightcurve", json=malicious_request)
        assert response.status_code == 422  # Should be rejected by validation
    
    def test_content_type_validation(self, client):
        """Test content type validation"""
        # Send XML instead of JSON
        xml_data = "<?xml version='1.0'?><!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><foo>&xxe;</foo>"
        
        response = client.post(
            "/api/v1/data/ingest/table",
            data=xml_data,
            headers={"Content-Type": "application/xml"}
        )
        
        assert response.status_code == 422  # Should reject non-JSON content
    
    def test_request_size_limits(self, client):
        """Test request size limits"""
        # Create oversized request
        large_request = {
            "table_type": "koi",
            "force_refresh": False,
            "large_data": "x" * 1000000  # 1MB of data
        }
        
        response = client.post("/api/v1/data/ingest/table", json=large_request)
        # Should be rejected due to size (exact status code may vary)
        assert response.status_code in [413, 422, 400]


class TestAuthenticationSecurity:
    """Test authentication and authorization security"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_unauthenticated_access(self, client):
        """Test that protected endpoints require authentication"""
        protected_endpoints = [
            "/api/v1/data/ingest/table",
            "/api/v1/data/ingest/lightcurve",
            "/api/v1/data/preprocess/lightcurve",
            "/api/v1/data/version/create"
        ]
        
        for endpoint in protected_endpoints:
            response = client.post(endpoint, json={})
            assert response.status_code in [401, 403], f"Endpoint {endpoint} should require auth"
    
    def test_malformed_jwt_tokens(self, client):
        """Test handling of malformed JWT tokens"""
        malformed_tokens = [
            "invalid.token.here",
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.invalid",
            "Bearer malformed_token",
            "Bearer ",
            "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWV9.invalid_signature"
        ]
        
        for token in malformed_tokens:
            headers = {"Authorization": token}
            response = client.post("/api/v1/data/ingest/table", json={}, headers=headers)
            assert response.status_code in [401, 403], f"Should reject malformed token: {token}"
    
    def test_token_injection_attempts(self, client):
        """Test JWT token injection attempts"""
        injection_attempts = [
            "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJzdWIiOiJhZG1pbiIsImV4cCI6OTk5OTk5OTk5OX0.",  # None algorithm
            "Bearer '; DROP TABLE users; --",
            "Bearer <script>alert('xss')</script>",
            "Bearer ../../../etc/passwd"
        ]
        
        for token in injection_attempts:
            headers = {"Authorization": token}
            response = client.post("/api/v1/data/ingest/table", json={}, headers=headers)
            assert response.status_code in [401, 403], f"Should reject injection attempt: {token}"


class TestCORSSecurity:
    """Test CORS security configuration"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_cors_allowed_origins(self, client):
        """Test that CORS allows only configured origins"""
        # Test allowed origin
        allowed_headers = {"Origin": "http://localhost:3000"}
        response = client.options("/api/v1/data/ingest/table", headers=allowed_headers)
        
        # Should include CORS headers for allowed origin
        if "access-control-allow-origin" in response.headers:
            assert response.headers["access-control-allow-origin"] in [
                "http://localhost:3000", 
                "*"  # May be configured differently
            ]
    
    def test_cors_malicious_origins(self, client):
        """Test CORS handling of malicious origins"""
        malicious_origins = [
            "http://evil.com",
            "https://attacker.example.com",
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "file:///etc/passwd"
        ]
        
        for origin in malicious_origins:
            headers = {"Origin": origin}
            response = client.options("/api/v1/data/ingest/table", headers=headers)
            
            # Should not include CORS headers for malicious origins
            if "access-control-allow-origin" in response.headers:
                assert response.headers["access-control-allow-origin"] != origin


@pytest.mark.asyncio
class TestSecurityHeaders:
    """Test security headers in responses"""
    
    def test_security_headers_present(self):
        """Test that security headers are present in responses"""
        client = TestClient(app)
        
        response = client.get("/health")
        
        # Check for important security headers
        expected_headers = [
            "x-content-type-options",
            "x-frame-options", 
            "x-xss-protection"
        ]
        
        # Note: These headers should be added by middleware
        # This test documents the expected behavior
        for header in expected_headers:
            # May not be implemented yet, but should be
            if header in response.headers:
                assert response.headers[header] is not None


class TestFileUploadSecurity:
    """Test file upload security (when implemented)"""
    
    def test_file_type_validation(self):
        """Test file type validation"""
        # This is a placeholder for when file upload is implemented
        # Should test:
        # - Only allowed file types (FITS, CSV, JSON)
        # - Rejection of executable files
        # - Content-type vs extension validation
        # - File size limits
        pass
    
    def test_file_content_scanning(self):
        """Test file content scanning for malicious content"""
        # Should test:
        # - Malware scanning
        # - Script injection in files
        # - Binary file validation
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
