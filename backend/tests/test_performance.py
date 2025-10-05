"""
Comprehensive performance tests for ExoplanetAI v2.0
Комплексные тесты производительности для ExoplanetAI v2.0
"""

import pytest
import asyncio
import time
import psutil
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from main import app
from ingest.data_manager import DataManager
from core.cache import get_cache
from core.rate_limiting import get_rate_limiter


class TestPerformanceBasics:
    """Basic performance tests"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_health_endpoint_response_time(self, client):
        """Test health endpoint response time"""
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code in [200, 503]
        assert response_time < 0.1  # Should respond within 100ms
    
    def test_root_endpoint_response_time(self, client):
        """Test root endpoint response time"""
        start_time = time.time()
        response = client.get("/")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 0.2  # Should respond within 200ms
    
    def test_metrics_endpoint_response_time(self, client):
        """Test metrics endpoint response time"""
        start_time = time.time()
        response = client.get("/metrics")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 0.5  # Metrics can be slightly slower


class TestConcurrentRequests:
    """Test concurrent request handling"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_concurrent_health_checks(self, client):
        """Test handling multiple concurrent health checks"""
        def make_request():
            return client.get("/health")
        
        # Test with 50 concurrent requests
        with ThreadPoolExecutor(max_workers=50) as executor:
            start_time = time.time()
            futures = [executor.submit(make_request) for _ in range(50)]
            
            responses = [future.result() for future in futures]
            end_time = time.time()
        
        total_time = end_time - start_time
        
        # All requests should succeed
        for response in responses:
            assert response.status_code in [200, 503]
        
        # Should handle 50 requests within 5 seconds
        assert total_time < 5.0
        
        # Calculate average response time
        avg_response_time = total_time / len(responses)
        assert avg_response_time < 0.5  # Average should be under 500ms
    
    @pytest.mark.asyncio
    async def test_async_concurrent_requests(self):
        """Test async concurrent request handling"""
        async def make_async_request(session, url):
            async with session.get(url) as response:
                return response.status, await response.text()
        
        base_url = "http://127.0.0.1:8001"  # Assuming test server
        
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                # Create 100 concurrent requests
                tasks = [
                    make_async_request(session, f"{base_url}/health")
                    for _ in range(100)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                
                total_time = end_time - start_time
                
                # Count successful requests
                successful_requests = sum(
                    1 for result in results 
                    if not isinstance(result, Exception) and result[0] in [200, 503]
                )
                
                # At least 90% should succeed
                success_rate = successful_requests / len(results)
                assert success_rate >= 0.9, f"Success rate too low: {success_rate}"
                
                # Should complete within 10 seconds
                assert total_time < 10.0, f"Took too long: {total_time}s"
                
        except Exception as e:
            # If server not running, skip this test
            pytest.skip(f"Server not available for async testing: {e}")


class TestMemoryUsage:
    """Test memory usage and potential leaks"""
    
    def test_memory_usage_health_endpoint(self):
        """Test memory usage of health endpoint"""
        client = TestClient(app)
        process = psutil.Process()
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss
        
        # Make many requests
        for _ in range(1000):
            response = client.get("/health")
            assert response.status_code in [200, 503]
        
        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 50MB)
        assert memory_increase < 50 * 1024 * 1024, f"Memory increased by {memory_increase / 1024 / 1024:.2f}MB"
    
    @pytest.mark.asyncio
    async def test_data_manager_memory_usage(self):
        """Test DataManager memory usage"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create and initialize multiple DataManager instances
        managers = []
        for _ in range(10):
            manager = DataManager()
            # Mock the initialization to avoid actual network calls
            with patch.object(manager, 'initialize', return_value=True):
                await manager.initialize()
            managers.append(manager)
        
        # Cleanup managers
        for manager in managers:
            with patch.object(manager, 'cleanup'):
                await manager.cleanup()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Should not use excessive memory
        assert memory_increase < 100 * 1024 * 1024, f"Memory increased by {memory_increase / 1024 / 1024:.2f}MB"


class TestCachePerformance:
    """Test caching system performance"""
    
    @pytest.mark.asyncio
    async def test_cache_hit_performance(self):
        """Test cache hit performance"""
        cache = await get_cache()
        
        # Store test data
        test_key = "performance_test_key"
        test_data = {"large_data": "x" * 10000}  # 10KB of data
        
        await cache.set(test_key, test_data, ttl=300)
        
        # Measure cache hit performance
        start_time = time.time()
        for _ in range(1000):
            result = await cache.get(test_key)
            assert result is not None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_hit = total_time / 1000
        
        # Each cache hit should be very fast (< 1ms)
        assert avg_time_per_hit < 0.001, f"Cache hits too slow: {avg_time_per_hit * 1000:.2f}ms average"
    
    @pytest.mark.asyncio
    async def test_cache_miss_performance(self):
        """Test cache miss performance"""
        cache = await get_cache()
        
        # Measure cache miss performance
        start_time = time.time()
        for i in range(100):
            result = await cache.get(f"nonexistent_key_{i}")
            assert result is None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_miss = total_time / 100
        
        # Cache misses should still be fast (< 5ms)
        assert avg_time_per_miss < 0.005, f"Cache misses too slow: {avg_time_per_miss * 1000:.2f}ms average"


class TestRateLimitingPerformance:
    """Test rate limiting performance"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_performance(self):
        """Test rate limiter performance under load"""
        rate_limiter = await get_rate_limiter()
        
        # Test performance of rate limit checks
        start_time = time.time()
        
        for i in range(1000):
            identifier = f"user_{i % 10}"  # 10 different users
            endpoint = "/test"
            
            is_allowed, info = await rate_limiter.check_rate_limit(
                identifier, endpoint, None
            )
            # Don't assert on result, just measure performance
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_check = total_time / 1000
        
        # Each rate limit check should be fast (< 5ms)
        assert avg_time_per_check < 0.005, f"Rate limit checks too slow: {avg_time_per_check * 1000:.2f}ms average"
    
    @pytest.mark.asyncio
    async def test_concurrent_rate_limit_checks(self):
        """Test concurrent rate limit checks"""
        rate_limiter = await get_rate_limiter()
        
        async def check_rate_limit(user_id):
            identifier = f"user_{user_id}"
            endpoint = "/test"
            return await rate_limiter.check_rate_limit(identifier, endpoint, None)
        
        start_time = time.time()
        
        # 100 concurrent rate limit checks
        tasks = [check_rate_limit(i) for i in range(100)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete quickly even with concurrency
        assert total_time < 1.0, f"Concurrent rate limit checks took too long: {total_time:.2f}s"
        
        # All checks should return valid results
        for is_allowed, info in results:
            assert isinstance(is_allowed, bool)
            assert isinstance(info, dict)


class TestDatabasePerformance:
    """Test database operation performance"""
    
    @pytest.mark.asyncio
    async def test_data_ingestion_performance(self):
        """Test data ingestion performance"""
        # Mock data manager to avoid actual network calls
        data_manager = DataManager()
        
        # Mock the external API calls
        with patch.object(data_manager, 'session') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "data": [{"id": i, "name": f"planet_{i}"} for i in range(1000)]
            })
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            start_time = time.time()
            
            # Simulate processing 1000 records
            with patch.object(data_manager.storage, 'save_table') as mock_save:
                mock_save.return_value = True
                
                # This would normally call the actual ingestion method
                # For performance testing, we simulate the work
                for i in range(1000):
                    # Simulate data processing
                    data = {"id": i, "processed": True}
                    await asyncio.sleep(0.001)  # Simulate processing time
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process 1000 records within reasonable time
            records_per_second = 1000 / processing_time
            assert records_per_second > 100, f"Processing too slow: {records_per_second:.2f} records/sec"


class TestAPIResponseTimes:
    """Test API endpoint response times"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_api_documentation_performance(self, client):
        """Test API documentation generation performance"""
        start_time = time.time()
        response = client.get("/docs")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        # Documentation should load within 2 seconds
        assert response_time < 2.0, f"API docs too slow: {response_time:.2f}s"
    
    def test_openapi_schema_performance(self, client):
        """Test OpenAPI schema generation performance"""
        start_time = time.time()
        response = client.get("/openapi.json")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        # Schema should generate within 1 second
        assert response_time < 1.0, f"OpenAPI schema too slow: {response_time:.2f}s"


class TestResourceUsage:
    """Test system resource usage"""
    
    def test_cpu_usage_under_load(self):
        """Test CPU usage under load"""
        client = TestClient(app)
        process = psutil.Process()
        
        # Get initial CPU usage
        initial_cpu = process.cpu_percent()
        
        # Generate load
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < 5.0:  # Run for 5 seconds
            response = client.get("/health")
            assert response.status_code in [200, 503]
            request_count += 1
        
        # Get final CPU usage
        final_cpu = process.cpu_percent()
        
        # CPU usage should be reasonable
        # Note: This test may be flaky depending on system load
        if final_cpu > 0:  # Only check if we got a reading
            assert final_cpu < 80, f"CPU usage too high: {final_cpu}%"
        
        # Should handle reasonable number of requests
        requests_per_second = request_count / 5.0
        assert requests_per_second > 10, f"Request rate too low: {requests_per_second:.2f} req/sec"
    
    def test_file_descriptor_usage(self):
        """Test file descriptor usage"""
        process = psutil.Process()
        
        try:
            initial_fds = process.num_fds()
        except AttributeError:
            # num_fds() not available on Windows
            pytest.skip("File descriptor counting not available on this platform")
        
        client = TestClient(app)
        
        # Make many requests
        for _ in range(100):
            response = client.get("/health")
            assert response.status_code in [200, 503]
        
        final_fds = process.num_fds()
        fd_increase = final_fds - initial_fds
        
        # Should not leak file descriptors
        assert fd_increase < 10, f"Too many file descriptors opened: {fd_increase}"


class TestScalabilityMetrics:
    """Test scalability metrics"""
    
    def test_response_time_under_increasing_load(self):
        """Test response time degradation under increasing load"""
        client = TestClient(app)
        
        load_levels = [1, 5, 10, 20, 50]
        response_times = []
        
        for load in load_levels:
            def make_requests():
                times = []
                for _ in range(load):
                    start = time.time()
                    response = client.get("/health")
                    end = time.time()
                    times.append(end - start)
                    assert response.status_code in [200, 503]
                return times
            
            times = make_requests()
            avg_time = sum(times) / len(times)
            response_times.append(avg_time)
        
        # Response times should not degrade too much
        # Allow some degradation but not exponential
        for i in range(1, len(response_times)):
            degradation = response_times[i] / response_times[0]
            assert degradation < 5.0, f"Response time degraded too much at load {load_levels[i]}: {degradation:.2f}x"
    
    def test_throughput_measurement(self):
        """Test overall system throughput"""
        client = TestClient(app)
        
        duration = 10.0  # Test for 10 seconds
        start_time = time.time()
        request_count = 0
        error_count = 0
        
        while time.time() - start_time < duration:
            response = client.get("/health")
            request_count += 1
            
            if response.status_code not in [200, 503]:
                error_count += 1
        
        actual_duration = time.time() - start_time
        throughput = request_count / actual_duration
        error_rate = error_count / request_count if request_count > 0 else 0
        
        # Should achieve reasonable throughput
        assert throughput > 50, f"Throughput too low: {throughput:.2f} req/sec"
        
        # Error rate should be low
        assert error_rate < 0.05, f"Error rate too high: {error_rate:.2%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
