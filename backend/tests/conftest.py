"""
Test configuration and fixtures
"""

import pytest
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Set environment variables for testing
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")  # Use test database
os.environ.setdefault("LOG_LEVEL", "WARNING")  # Reduce log noise in tests

import asyncio
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    class MockSettings:
        data_path = "/tmp/test_exoplanetai"
        redis_url = "redis://localhost:6379/1"  # Test database
        
    return MockSettings()
