"""
Data source registry for plug-and-play architecture
Реестр источников данных для модульной архитектуры
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from .base import BaseDataSource, DataSourceError, DataSourceType

logger = logging.getLogger(__name__)


class DataSourceRegistry:
    """
    Registry for managing multiple data sources

    Provides plug-and-play architecture where new data sources
    can be registered without modifying existing code.
    """

    def __init__(self):
        self._sources: Dict[str, BaseDataSource] = {}
        self._source_types: Dict[DataSourceType, List[BaseDataSource]] = {}
        self._initialized = False

    def register_source(self, source: BaseDataSource) -> None:
        """
        Register a new data source

        Args:
            source: Data source instance to register
        """
        if source.name in self._sources:
            logger.warning(f"Data source '{source.name}' already registered, replacing")

        self._sources[source.name] = source

        # Group by type
        if source.source_type not in self._source_types:
            self._source_types[source.source_type] = []

        # Remove existing source of same type and name
        self._source_types[source.source_type] = [
            s for s in self._source_types[source.source_type] if s.name != source.name
        ]
        self._source_types[source.source_type].append(source)

        logger.info(f"Registered data source: {source}")

    def unregister_source(self, name: str) -> bool:
        """
        Unregister a data source

        Args:
            name: Name of source to unregister

        Returns:
            bool: True if source was found and removed
        """
        if name not in self._sources:
            return False

        source = self._sources[name]
        del self._sources[name]

        # Remove from type grouping
        if source.source_type in self._source_types:
            self._source_types[source.source_type] = [
                s for s in self._source_types[source.source_type] if s.name != name
            ]

        logger.info(f"Unregistered data source: {name}")
        return True

    def get_source(self, name: str) -> Optional[BaseDataSource]:
        """Get data source by name"""
        return self._sources.get(name)

    def get_sources_by_type(self, source_type: DataSourceType) -> List[BaseDataSource]:
        """Get all sources of a specific type"""
        return self._source_types.get(source_type, [])

    def get_all_sources(self) -> List[BaseDataSource]:
        """Get all registered sources"""
        return list(self._sources.values())

    def get_available_sources(self) -> List[BaseDataSource]:
        """Get all initialized and healthy sources"""
        return [s for s in self._sources.values() if s.is_initialized]

    async def initialize_all(self) -> Dict[str, bool]:
        """
        Initialize all registered data sources

        Returns:
            Dict mapping source names to initialization success
        """
        results = {}

        for name, source in self._sources.items():
            try:
                logger.info(f"Initializing data source: {name}")
                success = await source.initialize()
                results[name] = success

                if success:
                    logger.info(f"✅ {name} initialized successfully")
                else:
                    logger.error(f"❌ {name} failed to initialize")

            except Exception as e:
                logger.error(f"❌ {name} initialization error: {e}")
                results[name] = False

        self._initialized = True
        return results

    async def cleanup_all(self):
        """Cleanup all data sources"""
        cleanup_tasks = []

        for source in self._sources.values():
            if source.is_initialized:
                cleanup_tasks.append(source.cleanup())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        logger.info("All data sources cleaned up")

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform health check on all sources

        Returns:
            Dict mapping source names to health status
        """
        results = {}

        health_tasks = []
        source_names = []

        for name, source in self._sources.items():
            if source.is_initialized:
                health_tasks.append(source.health_check())
                source_names.append(name)

        if health_tasks:
            health_results = await asyncio.gather(*health_tasks, return_exceptions=True)

            for name, result in zip(source_names, health_results):
                if isinstance(result, Exception):
                    results[name] = {
                        "status": "error",
                        "error": str(result),
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    results[name] = result

        return results

    async def search_all_sources(
        self,
        query: str,
        limit: int = 100,
        source_types: Optional[List[DataSourceType]] = None,
    ) -> Dict[str, Any]:
        """
        Search across multiple data sources

        Args:
            query: Search query
            limit: Results per source
            source_types: Limit to specific source types

        Returns:
            Aggregated search results
        """
        sources_to_search = []

        if source_types:
            for source_type in source_types:
                sources_to_search.extend(self.get_sources_by_type(source_type))
        else:
            sources_to_search = self.get_available_sources()

        if not sources_to_search:
            return {
                "query": query,
                "total_sources": 0,
                "results": {},
                "errors": [],
                "search_time_ms": 0,
            }

        start_time = datetime.now()

        # Execute searches in parallel
        search_tasks = []
        source_names = []

        for source in sources_to_search:
            search_tasks.append(source.search_planets(query, limit))
            source_names.append(source.name)

        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Process results
        results = {}
        errors = []
        total_planets = 0

        for name, result in zip(source_names, search_results):
            if isinstance(result, Exception):
                errors.append(
                    {
                        "source": name,
                        "error": str(result),
                        "type": type(result).__name__,
                    }
                )
            else:
                results[name] = {
                    "total_found": result.total_found,
                    "planets": [p.__dict__ for p in result.planets],
                    "search_time_ms": result.search_time_ms,
                    "cached": result.cached,
                }
                total_planets += result.total_found

        end_time = datetime.now()
        total_time_ms = (end_time - start_time).total_seconds() * 1000

        return {
            "query": query,
            "total_sources": len(sources_to_search),
            "total_planets_found": total_planets,
            "results": results,
            "errors": errors,
            "search_time_ms": total_time_ms,
        }

    async def get_aggregated_statistics(self) -> Dict[str, Any]:
        """Get aggregated statistics from all sources"""
        sources = self.get_available_sources()

        if not sources:
            return {"error": "No available data sources"}

        # Get statistics from all sources
        stats_tasks = []
        source_names = []

        for source in sources:
            stats_tasks.append(source.get_statistics())
            source_names.append(source.name)

        stats_results = await asyncio.gather(*stats_tasks, return_exceptions=True)

        # Aggregate results
        aggregated = {
            "sources": {},
            "total_planets": 0,
            "by_discovery_method": {},
            "by_discovery_year": {},
            "by_status": {},
            "timestamp": datetime.now().isoformat(),
        }

        for name, stats in zip(source_names, stats_results):
            if isinstance(stats, Exception):
                aggregated["sources"][name] = {"error": str(stats)}
            else:
                aggregated["sources"][name] = stats

                # Aggregate totals
                if "total_planets" in stats:
                    aggregated["total_planets"] += stats.get("total_planets", 0)

        return aggregated

    def get_registry_info(self) -> Dict[str, Any]:
        """Get information about the registry"""
        return {
            "total_sources": len(self._sources),
            "initialized_sources": len(self.get_available_sources()),
            "sources_by_type": {
                source_type.value: len(sources)
                for source_type, sources in self._source_types.items()
            },
            "source_names": list(self._sources.keys()),
            "initialized": self._initialized,
        }


# Global registry instance
_registry = DataSourceRegistry()


def get_registry() -> DataSourceRegistry:
    """Get the global data source registry"""
    return _registry


def register_source(source: BaseDataSource) -> None:
    """Convenience function to register a source"""
    _registry.register_source(source)


async def initialize_default_sources():
    """Initialize default data sources"""
    from .esa import ESADataSource
    from .kepler import KeplerDataSource
    from .nasa import NASADataSource
    from .tess import TESSDataSource

    # Register default sources
    register_source(NASADataSource())
    register_source(ESADataSource())
    register_source(KeplerDataSource())
    register_source(TESSDataSource())

    # Initialize all
    results = await _registry.initialize_all()

    successful = sum(1 for success in results.values() if success)
    total = len(results)

    logger.info(f"Initialized {successful}/{total} data sources")

    return results
