import pytest
from fastapi.testclient import TestClient
from main import app

class TestMainAPI:
    def setup_method(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get('/health')
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'

    def test_root_endpoint(self):
        response = self.client.get('/')
        assert response.status_code == 200
        assert 'Exoplanet AI API' in response.json()['message']

    def test_get_nasa_stats(self):
        response = self.client.get('/api/nasa/stats')
        assert response.status_code == 200
        stats = response.json()
        assert 'totalPlanets' in stats
        assert 'totalHosts' in stats

    def test_load_tic_data(self):
        response = self.client.post('/load-tic', json={'tic_id': 'TIC 307210830'})
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        assert 'data' in data

    def test_pro_analysis(self):
        lightcurve_data = {
            'times': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'fluxes': [1.0, 1.1, 0.9, 1.0, 1.1, 0.95, 1.05, 1.0, 0.9, 1.1]
        }

        response = self.client.post('/pro/analyze', json={
            'lightcurve_data': lightcurve_data,
            'model_type': 'detector',
            'tic_id': 'TIC 307210830'
        })

        assert response.status_code == 200
        result = response.json()
        assert result['success'] == True
        assert 'candidates' in result
        assert 'detailed_analysis' in result
        assert 'processing_time' in result
