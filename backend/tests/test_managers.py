"""
Comprehensive unit tests for all managers
Комплексные unit тесты для всех менеджеров
"""

import asyncio
import pytest
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# Import managers
from ingest.data_manager import DataManager
from ingest.storage import StorageManager
from ingest.validator import DataValidator
from ingest.versioning import VersionManager
from preprocessing.lightcurve_processor import LightCurveProcessor


class TestDataManager:
    """Test DataManager functionality"""
    
    @pytest.fixture
    async def data_manager(self):
        """Create DataManager instance for testing"""
        manager = DataManager()
        yield manager
        # Cleanup
        if hasattr(manager, 'cleanup'):
            await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, data_manager):
        """Test successful initialization"""
        with patch.object(data_manager, '_setup_storage', return_value=True), \
             patch.object(data_manager, '_setup_cache', return_value=True):
            
            result = await data_manager.initialize()
            assert result is True
            assert data_manager.initialized is True
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self, data_manager):
        """Test initialization failure handling"""
        with patch.object(data_manager, '_setup_storage', side_effect=Exception("Storage failed")):
            
            result = await data_manager.initialize()
            assert result is False
            assert data_manager.initialized is False
    
    @pytest.mark.asyncio
    async def test_concurrent_initialization(self):
        """Test concurrent initialization safety"""
        manager = DataManager()
        
        with patch.object(manager, '_setup_storage', return_value=True), \
             patch.object(manager, '_setup_cache', return_value=True):
            
            # Start multiple initialization tasks
            tasks = [manager.initialize() for _ in range(5)]
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            assert all(results)
            assert manager.initialized is True
    
    @pytest.mark.asyncio
    async def test_ingest_koi_table(self, data_manager):
        """Test KOI table ingestion"""
        await data_manager.initialize()
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.text.return_value = "test,data\n1,2"
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await data_manager.ingest_koi_table(force_refresh=True)
            
            assert result is not None
            assert 'status' in result
    
    @pytest.mark.asyncio
    async def test_cleanup_on_error(self, data_manager):
        """Test cleanup when errors occur"""
        await data_manager.initialize()
        
        # Simulate error during operation
        with patch.object(data_manager, 'ingest_koi_table', side_effect=Exception("Test error")):
            with pytest.raises(Exception):
                await data_manager.ingest_koi_table()
        
        # Cleanup should still work
        await data_manager.cleanup()


class TestStorageManager:
    """Test StorageManager functionality"""
    
    @pytest.fixture
    async def storage_manager(self):
        """Create StorageManager instance for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StorageManager(base_path=temp_dir)
            yield manager
            if hasattr(manager, 'cleanup'):
                await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, storage_manager):
        """Test successful initialization"""
        result = await storage_manager.initialize()
        assert result is True
        assert storage_manager.initialized is True
    
    @pytest.mark.asyncio
    async def test_save_and_load_data(self, storage_manager):
        """Test data save and load operations"""
        await storage_manager.initialize()
        
        test_data = {"test": "data", "numbers": [1, 2, 3]}
        
        # Save data
        file_path = await storage_manager.save_data(test_data, "test_file.json")
        assert file_path.exists()
        
        # Load data
        loaded_data = await storage_manager.load_data(file_path)
        assert loaded_data == test_data
    
    @pytest.mark.asyncio
    async def test_storage_stats(self, storage_manager):
        """Test storage statistics"""
        await storage_manager.initialize()
        
        # Create some test files
        test_data = {"test": "data"}
        await storage_manager.save_data(test_data, "test1.json")
        await storage_manager.save_data(test_data, "test2.json")
        
        stats = await storage_manager.get_storage_stats()
        
        assert 'total_files' in stats
        assert 'total_size' in stats
        assert stats['total_files'] >= 2
    
    @pytest.mark.asyncio
    async def test_cleanup_old_files(self, storage_manager):
        """Test cleanup of old files"""
        await storage_manager.initialize()
        
        # Create test files
        old_file = await storage_manager.save_data({"old": "data"}, "old_file.json")
        
        # Mock file modification time to be old
        old_time = os.path.getmtime(old_file) - (8 * 24 * 3600)  # 8 days ago
        os.utime(old_file, (old_time, old_time))
        
        # Cleanup files older than 7 days
        cleaned_count = await storage_manager.cleanup_old_files(max_age_days=7)
        
        assert cleaned_count >= 1
        assert not old_file.exists()


class TestDataValidator:
    """Test DataValidator functionality"""
    
    @pytest.fixture
    def validator(self):
        """Create DataValidator instance for testing"""
        return DataValidator()
    
    def test_validate_koi_data(self, validator):
        """Test KOI data validation"""
        valid_data = {
            'kepoi_name': 'K00001.01',
            'koi_period': 10.5,
            'koi_depth': 100.0,
            'koi_disposition': 'CANDIDATE'
        }
        
        result = validator.validate_koi_data(valid_data)
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_invalid_koi_data(self, validator):
        """Test validation of invalid KOI data"""
        invalid_data = {
            'kepoi_name': '',  # Empty name
            'koi_period': -1,  # Negative period
            'koi_depth': 'invalid',  # Non-numeric depth
        }
        
        result = validator.validate_koi_data(invalid_data)
        assert result['is_valid'] is False
        assert len(result['errors']) > 0
    
    def test_validate_lightcurve_data(self, validator):
        """Test light curve data validation"""
        import numpy as np
        
        valid_data = {
            'time': np.array([1, 2, 3, 4, 5]),
            'flux': np.array([1.0, 0.99, 1.01, 0.98, 1.02]),
            'flux_err': np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        }
        
        result = validator.validate_lightcurve_data(valid_data)
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_empty_lightcurve(self, validator):
        """Test validation of empty light curve"""
        import numpy as np
        
        empty_data = {
            'time': np.array([]),
            'flux': np.array([]),
            'flux_err': np.array([])
        }
        
        result = validator.validate_lightcurve_data(empty_data)
        assert result['is_valid'] is False
        assert 'empty' in str(result['errors']).lower()


class TestVersionManager:
    """Test VersionManager functionality"""
    
    @pytest.fixture
    async def version_manager(self):
        """Create VersionManager instance for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VersionManager(base_path=temp_dir)
            yield manager
            if hasattr(manager, 'cleanup'):
                await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, version_manager):
        """Test successful initialization"""
        result = await version_manager.initialize()
        assert result is True
        assert version_manager.initialized is True
    
    @pytest.mark.asyncio
    async def test_create_version(self, version_manager):
        """Test version creation"""
        await version_manager.initialize()
        
        version_info = {
            'name': 'test_version_1.0',
            'description': 'Test version',
            'data_sources': ['koi', 'toi']
        }
        
        result = await version_manager.create_version(version_info)
        
        assert result is not None
        assert 'version_id' in result
        assert result['status'] == 'created'
    
    @pytest.mark.asyncio
    async def test_list_versions(self, version_manager):
        """Test version listing"""
        await version_manager.initialize()
        
        # Create test versions
        await version_manager.create_version({
            'name': 'v1.0',
            'description': 'Version 1'
        })
        await version_manager.create_version({
            'name': 'v2.0',
            'description': 'Version 2'
        })
        
        versions = await version_manager.list_versions()
        
        assert len(versions) >= 2
        assert any(v['name'] == 'v1.0' for v in versions)
        assert any(v['name'] == 'v2.0' for v in versions)
    
    @pytest.mark.asyncio
    async def test_get_version_info(self, version_manager):
        """Test getting version information"""
        await version_manager.initialize()
        
        # Create test version
        create_result = await version_manager.create_version({
            'name': 'test_version',
            'description': 'Test version for info retrieval'
        })
        
        version_info = await version_manager.get_version_info('test_version')
        
        assert version_info is not None
        assert version_info['name'] == 'test_version'
        assert 'description' in version_info
        assert 'created_at' in version_info


class TestLightCurveProcessor:
    """Test LightCurveProcessor functionality"""
    
    @pytest.fixture
    def processor(self):
        """Create LightCurveProcessor instance for testing"""
        return LightCurveProcessor()
    
    def test_normalize_lightcurve(self, processor):
        """Test light curve normalization"""
        import numpy as np
        
        # Create test light curve with trend
        time = np.linspace(0, 10, 100)
        flux = 1.0 + 0.1 * time + 0.01 * np.random.randn(100)  # Linear trend + noise
        flux_err = np.full_like(flux, 0.01)
        
        normalized = processor.normalize_lightcurve(time, flux, flux_err)
        
        assert 'time' in normalized
        assert 'flux' in normalized
        assert 'flux_err' in normalized
        
        # Check that normalization worked (mean should be close to 1)
        assert abs(np.mean(normalized['flux']) - 1.0) < 0.1
    
    def test_remove_outliers(self, processor):
        """Test outlier removal"""
        import numpy as np
        
        # Create data with outliers
        time = np.linspace(0, 10, 100)
        flux = np.ones(100)
        flux[50] = 10.0  # Add outlier
        flux_err = np.full_like(flux, 0.01)
        
        cleaned = processor.remove_outliers(time, flux, flux_err, sigma_threshold=3.0)
        
        assert len(cleaned['time']) < len(time)  # Some points should be removed
        assert 10.0 not in cleaned['flux']  # Outlier should be removed
    
    def test_detrend_lightcurve(self, processor):
        """Test light curve detrending"""
        import numpy as np
        
        # Create light curve with polynomial trend
        time = np.linspace(0, 10, 100)
        trend = 0.1 * time**2 - 0.05 * time + 1.0
        flux = trend + 0.01 * np.random.randn(100)
        flux_err = np.full_like(flux, 0.01)
        
        detrended = processor.detrend_lightcurve(time, flux, flux_err, method='polynomial', degree=2)
        
        assert 'time' in detrended
        assert 'flux' in detrended
        assert 'flux_err' in detrended
        
        # Detrended flux should have less variation
        original_std = np.std(flux)
        detrended_std = np.std(detrended['flux'])
        assert detrended_std < original_std
    
    def test_bin_lightcurve(self, processor):
        """Test light curve binning"""
        import numpy as np
        
        # Create high-cadence light curve
        time = np.linspace(0, 10, 1000)
        flux = np.ones(1000) + 0.01 * np.random.randn(1000)
        flux_err = np.full_like(flux, 0.01)
        
        binned = processor.bin_lightcurve(time, flux, flux_err, bin_size=0.1)
        
        assert len(binned['time']) < len(time)  # Should be fewer points
        assert 'time' in binned
        assert 'flux' in binned
        assert 'flux_err' in binned
        
        # Binned errors should be smaller (due to averaging)
        assert np.mean(binned['flux_err']) < np.mean(flux_err)


class TestManagerIntegration:
    """Integration tests for manager interactions"""
    
    @pytest.mark.asyncio
    async def test_full_data_pipeline(self):
        """Test complete data processing pipeline"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize all managers
            data_manager = DataManager()
            storage_manager = StorageManager(base_path=temp_dir)
            validator = DataValidator()
            processor = LightCurveProcessor()
            
            try:
                # Initialize managers
                await data_manager.initialize()
                await storage_manager.initialize()
                
                # Mock data ingestion
                with patch.object(data_manager, 'ingest_koi_table') as mock_ingest:
                    mock_ingest.return_value = {
                        'status': 'success',
                        'records_processed': 100,
                        'data': [{'kepoi_name': 'K00001.01', 'koi_period': 10.5}]
                    }
                    
                    # Test pipeline: ingest → validate → store → process
                    ingest_result = await data_manager.ingest_koi_table()
                    assert ingest_result['status'] == 'success'
                    
                    # Validate data
                    validation_result = validator.validate_koi_data(ingest_result['data'][0])
                    assert validation_result['is_valid'] is True
                    
                    # Store data
                    storage_path = await storage_manager.save_data(
                        ingest_result['data'], 
                        'koi_data.json'
                    )
                    assert storage_path.exists()
                    
                    # Load and verify
                    loaded_data = await storage_manager.load_data(storage_path)
                    assert loaded_data == ingest_result['data']
                    
            finally:
                # Cleanup
                await data_manager.cleanup()
                await storage_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_concurrent_manager_operations(self):
        """Test concurrent operations across managers"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple manager instances
            managers = [
                DataManager(),
                StorageManager(base_path=temp_dir),
                VersionManager(base_path=temp_dir)
            ]
            
            try:
                # Initialize all managers concurrently
                init_tasks = [manager.initialize() for manager in managers]
                init_results = await asyncio.gather(*init_tasks)
                
                # All should succeed
                assert all(init_results)
                
                # Test concurrent operations
                async def test_operation(manager, op_id):
                    if hasattr(manager, 'get_storage_stats'):
                        return await manager.get_storage_stats()
                    elif hasattr(manager, 'list_versions'):
                        return await manager.list_versions()
                    else:
                        return {'operation_id': op_id, 'status': 'completed'}
                
                # Run concurrent operations
                operation_tasks = [
                    test_operation(manager, i) 
                    for i, manager in enumerate(managers)
                ]
                
                operation_results = await asyncio.gather(*operation_tasks)
                assert len(operation_results) == len(managers)
                
            finally:
                # Cleanup all managers
                cleanup_tasks = [
                    manager.cleanup() for manager in managers 
                    if hasattr(manager, 'cleanup')
                ]
                if cleanup_tasks:
                    await asyncio.gather(*cleanup_tasks)
    
    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test error handling across manager boundaries"""
        data_manager = DataManager()
        
        try:
            # Test initialization failure propagation
            with patch.object(data_manager, '_setup_storage', side_effect=Exception("Storage failed")):
                result = await data_manager.initialize()
                assert result is False
                
            # Test operation failure handling
            with patch.object(data_manager, 'ingest_koi_table', side_effect=Exception("Network error")):
                with pytest.raises(Exception) as exc_info:
                    await data_manager.ingest_koi_table()
                assert "Network error" in str(exc_info.value)
                
        finally:
            if hasattr(data_manager, 'cleanup'):
                await data_manager.cleanup()


# Performance tests
class TestManagerPerformance:
    """Performance tests for managers"""
    
    @pytest.mark.asyncio
    async def test_concurrent_initialization_performance(self):
        """Test performance of concurrent manager initialization"""
        import time
        
        async def create_and_init_manager():
            manager = DataManager()
            with patch.object(manager, '_setup_storage', return_value=True), \
                 patch.object(manager, '_setup_cache', return_value=True):
                await manager.initialize()
            return manager
        
        # Measure time for concurrent initialization
        start_time = time.time()
        
        tasks = [create_and_init_manager() for _ in range(10)]
        managers = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (less than 5 seconds)
        assert duration < 5.0
        assert len(managers) == 10
        assert all(manager.initialized for manager in managers)
    
    @pytest.mark.asyncio
    async def test_storage_performance(self):
        """Test storage manager performance with large datasets"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_manager = StorageManager(base_path=temp_dir)
            await storage_manager.initialize()
            
            try:
                # Create large dataset
                large_data = {
                    'records': [
                        {'id': i, 'value': f'data_{i}', 'numbers': list(range(100))}
                        for i in range(1000)
                    ]
                }
                
                # Measure save performance
                import time
                start_time = time.time()
                
                file_path = await storage_manager.save_data(large_data, 'large_dataset.json')
                
                save_time = time.time() - start_time
                
                # Should save within reasonable time (less than 2 seconds)
                assert save_time < 2.0
                assert file_path.exists()
                
                # Measure load performance
                start_time = time.time()
                
                loaded_data = await storage_manager.load_data(file_path)
                
                load_time = time.time() - start_time
                
                # Should load within reasonable time (less than 1 second)
                assert load_time < 1.0
                assert loaded_data == large_data
                
            finally:
                await storage_manager.cleanup()
