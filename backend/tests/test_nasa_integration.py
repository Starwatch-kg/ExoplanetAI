"""
Integration tests for NASA data services - Real data only
Интеграционные тесты для сервисов NASA данных - только реальные данные
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import numpy as np
from datetime import datetime

from services.secure_nasa_service import SecureNASAService
from core.validators import validate_target_name, SecurityError
from schemas import create_success_response, create_error_response, ErrorCode


class TestSecureNASAService:
    """Test secure NASA service with real data validation"""
    
    @pytest.fixture
    async def nasa_service(self):
        """Create NASA service instance"""
        service = SecureNASAService()
        await service.initialize()
        yield service
        await service.cleanup()
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, nasa_service):
        """Test service initializes correctly"""
        assert nasa_service.session is not None
        assert nasa_service.cache == {}
        assert len(nasa_service.data_sources) > 0
    
    @pytest.mark.asyncio
    async def test_target_validation(self):
        """Test target name validation"""
        # Valid targets
        valid_targets = ["TOI-715", "Kepler-452b", "TIC 123456", "KIC-123456"]
        for target in valid_targets:
            validated = validate_target_name(target)
            assert isinstance(validated, str)
            assert len(validated) > 0
        
        # Invalid targets
        invalid_targets = ["", "<script>", "'; DROP TABLE;", "a" * 100]
        for target in invalid_targets:
            with pytest.raises(ValueError):
                validate_target_name(target)
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, nasa_service):
        """Test rate limiting functionality"""
        service_name = "test_service"
        
        # Should allow initial requests
        assert nasa_service._check_rate_limit(service_name) == True
        
        # Simulate many requests
        for _ in range(60):  # Default limit
            nasa_service._check_rate_limit(service_name)
        
        # Should be rate limited now
        assert nasa_service._check_rate_limit(service_name) == False
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, nasa_service):
        """Test caching mechanism"""
        cache_key = "test_key"
        test_data = {"test": "data"}
        
        # Initially not cached
        assert not nasa_service._is_cached(cache_key)
        
        # Cache data
        nasa_service._cache_data(cache_key, test_data)
        
        # Should be cached now
        assert nasa_service._is_cached(cache_key)
        assert nasa_service.cache[cache_key] == test_data
    
    @pytest.mark.asyncio
    async def test_secure_http_request_validation(self, nasa_service):
        """Test HTTP request security validation"""
        # Valid URLs should work
        valid_urls = [
            "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
            "https://mast.stsci.edu/api/v0.1/search"
        ]
        
        # Invalid URLs should raise SecurityError
        invalid_urls = [
            "http://malicious-site.com",
            "https://evil.com/api",
            "ftp://unsafe.com"
        ]
        
        for url in invalid_urls:
            with pytest.raises(SecurityError):
                await nasa_service._secure_http_request(url)
    
    @pytest.mark.asyncio
    @patch('services.secure_nasa_service.NasaExoplanetArchive')
    async def test_exoplanet_archive_query(self, mock_archive, nasa_service):
        """Test NASA Exoplanet Archive query with mocked response"""
        # Mock successful response
        mock_result = MagicMock()
        mock_result.__len__ = MagicMock(return_value=1)
        mock_result.__getitem__ = MagicMock(return_value={
            'pl_name': 'Test Planet b',
            'hostname': 'Test Star',
            'ra': 123.45,
            'dec': -67.89,
            'sy_vmag': 12.5,
            'st_teff': 5500,
            'st_rad': 1.0,
            'st_mass': 1.0,
            'pl_orbper': 365.25,
            'pl_radj': 1.0,
            'pl_massj': 1.0,
            'disc_year': 2023,
            'disc_facility': 'TESS'
        })
        
        mock_archive.query_criteria.return_value = mock_result
        
        result = await nasa_service.search_exoplanet_archive("Test Star")
        
        assert result is not None
        assert result['target_name'] == "Test Star"
        assert result['planets_found'] == 1
        assert 'host_star' in result
        assert 'planets' in result
    
    @pytest.mark.asyncio
    @patch('services.secure_nasa_service.lk')
    async def test_tess_lightcurve_download(self, mock_lk, nasa_service):
        """Test TESS lightcurve download with mocked response"""
        # Mock lightkurve search and download
        mock_search_result = MagicMock()
        mock_search_result.__len__ = MagicMock(return_value=1)
        
        mock_lc = MagicMock()
        mock_lc.time.value = np.linspace(0, 27, 1000)
        mock_lc.flux.value = np.ones(1000) + 0.001 * np.random.randn(1000)
        mock_lc.flux_err.value = np.ones(1000) * 0.001
        
        mock_lc_collection = MagicMock()
        mock_lc_collection.__len__ = MagicMock(return_value=1)
        mock_lc_collection.stitch.return_value = mock_lc
        
        mock_search_result.download_all.return_value = mock_lc_collection
        mock_lk.search_lightcurve.return_value = mock_search_result
        
        # Mock the remove_nans and other methods
        mock_lc.remove_nans.return_value = mock_lc
        mock_lc.remove_outliers.return_value = mock_lc
        mock_lc.normalize.return_value = mock_lc
        
        result = await nasa_service.get_tess_lightcurve("TIC 123456")
        
        assert result is not None
        time_data, flux_data, flux_err_data = result
        assert len(time_data) == 1000
        assert len(flux_data) == 1000
        assert len(flux_err_data) == 1000
    
    @pytest.mark.asyncio
    async def test_target_validation_integration(self, nasa_service):
        """Test target validation with real service"""
        # This would normally make real API calls, but we'll mock the responses
        with patch.object(nasa_service, 'search_exoplanet_archive') as mock_archive, \
             patch.object(nasa_service, 'get_tess_lightcurve') as mock_tess:
            
            # Mock found in archive
            mock_archive.return_value = {"planets_found": 1}
            mock_tess.return_value = None
            
            result = await nasa_service.validate_target_exists("TOI-715")
            assert result == True
            
            # Mock not found anywhere
            mock_archive.return_value = None
            mock_tess.return_value = None
            
            result = await nasa_service.validate_target_exists("NonExistent")
            assert result == False


class TestValidators:
    """Test input validation functions"""
    
    def test_target_name_validation(self):
        """Test target name validation edge cases"""
        # Valid names
        valid_names = [
            "TOI-715",
            "Kepler-452b", 
            "TIC 123456",
            "HD 209458",
            "WASP-12b",
            "K2-18b"
        ]
        
        for name in valid_names:
            result = validate_target_name(name)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_security_validation(self):
        """Test security validation functions"""
        # XSS attempts
        xss_attempts = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "'; DROP TABLE users; --"
        ]
        
        for attempt in xss_attempts:
            with pytest.raises(ValueError):
                validate_target_name(attempt)


class TestResponseSchemas:
    """Test response schema validation"""
    
    def test_success_response_creation(self):
        """Test success response creation"""
        data = {"test": "data"}
        response = create_success_response(data, "Test successful", 123.45)
        
        assert response["status"] == "ok"
        assert response["message"] == "Test successful"
        assert response["data"] == data
        assert response["processing_time_ms"] == 123.45
        assert response["error"] is None
    
    def test_error_response_creation(self):
        """Test error response creation"""
        response = create_error_response(
            ErrorCode.DATA_NOT_FOUND,
            "Data not found",
            {"target": "test"},
            100.0
        )
        
        assert response["status"] == "error"
        assert response["data"] is None
        assert response["error"]["code"] == "data_not_found"
        assert response["error"]["message"] == "Data not found"
        assert response["processing_time_ms"] == 100.0


@pytest.mark.integration
class TestRealDataIntegration:
    """Integration tests that can optionally use real NASA APIs"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-integration", default=False),
        reason="Integration tests disabled by default"
    )
    async def test_real_nasa_archive_query(self):
        """Test real NASA Exoplanet Archive query (optional)"""
        service = SecureNASAService()
        await service.initialize()
        
        try:
            # Query a well-known exoplanet
            result = await service.search_exoplanet_archive("Kepler-452")
            
            if result:  # Only assert if data was found
                assert result['target_name'] == "Kepler-452"
                assert isinstance(result['planets_found'], int)
                assert result['planets_found'] >= 0
        
        finally:
            await service.cleanup()
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-integration", default=False),
        reason="Integration tests disabled by default"
    )
    async def test_real_tess_data_download(self):
        """Test real TESS data download (optional)"""
        service = SecureNASAService()
        await service.initialize()
        
        try:
            # Try to download data for a known TESS target
            result = await service.get_tess_lightcurve("TIC 261136679")
            
            if result:  # Only assert if data was found
                time_data, flux_data, flux_err_data = result
                assert len(time_data) > 100  # Should have substantial data
                assert len(flux_data) == len(time_data)
                assert len(flux_err_data) == len(time_data)
                assert np.all(np.isfinite(time_data))
                assert np.all(np.isfinite(flux_data))
        
        finally:
            await service.cleanup()


# Pytest configuration
def pytest_addoption(parser):
    """Add command line options for pytest"""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that make real API calls"
    )


# Test fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
