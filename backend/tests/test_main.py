"""
Tests for main application
"""

import pytest
from fastapi.testclient import TestClient

from main import app


class TestMainApp:
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns correct response"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "ExoplanetAI Backend v2.0" in data["message"]
        assert data["version"] == "2.0.0"
        assert "features" in data
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code in [200, 503]  # May be degraded
        data = response.json()
        assert "status" in data
        assert "components" in data
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
