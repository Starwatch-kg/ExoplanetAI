import pytest
import asyncio
from nasa_api import RealNASAService

class TestNASAApi:
    @pytest.fixture
    async def nasa_service(self):
        async with RealNASAService() as service:
            yield service

    @pytest.mark.asyncio
    async def test_get_real_exoplanet_statistics(self, nasa_service):
        stats = await nasa_service.get_real_exoplanet_statistics()

        assert 'totalPlanets' in stats
        assert 'totalHosts' in stats
        assert 'source' in stats
        assert stats['totalPlanets'] > 0
        assert stats['totalHosts'] > 0

    @pytest.mark.asyncio
    async def test_search_tess_observations(self, nasa_service):
        tess_data = await nasa_service.search_tess_observations('TIC 307210830')

        assert isinstance(tess_data, list)
        if tess_data:
            assert 'tic_id' in tess_data[0]
            assert 'tess_mag' in tess_data[0]

    @pytest.mark.asyncio
    async def test_search_simbad_data(self, nasa_service):
        simbad_data = await nasa_service.search_simbad_data('TIC 307210830')

        assert isinstance(simbad_data, list)
        if simbad_data:
            assert 'tic_id' in simbad_data[0]
            assert 'spectral_type' in simbad_data[0]

    @pytest.mark.asyncio
    async def test_search_real_nasa_planets(self, nasa_service):
        planets = await nasa_service.search_real_nasa_planets('TIC 307210830')

        assert isinstance(planets, list)
        # TOI-700 d должен быть найден
        planet_names = [p['planet_name'] for p in planets]
        assert 'TOI-700 d' in planet_names
