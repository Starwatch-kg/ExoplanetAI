"""
Redis-based caching system for ExoplanetAI
Система кэширования на основе Redis
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from core.config import config

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Base cache error"""

    pass


class CacheManager:
    """
    Redis-based cache manager with TTL support

    Features:
    - Automatic serialization/deserialization
    - TTL (Time To Live) support
    - Prefetch for popular queries
    - Cache statistics
    - Fallback to in-memory cache if Redis unavailable
    """

    def __init__(
        self, redis_url: Optional[str] = None, default_ttl: int = 21600
    ):  # 6 hours
        self.redis_url = redis_url or config.cache.redis_url or "redis://localhost:6379"
        self.default_ttl = default_ttl
        self.redis_client: Optional[redis.Redis] = None
        self.is_connected = False

        # Fallback in-memory cache
        self._memory_cache: Dict[str, Dict] = {}
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
        }

        # Popular queries for prefetch
        self.popular_queries = [
            "TOI-715",
            "Kepler-452b",
            "TRAPPIST-1",
            "Proxima Centauri b",
            "K2-18b",
            "WASP-12b",
            "HD 209458b",
            "51 Eridani b",
            "HR 8799",
            "Kepler-186f",
        ]

    async def initialize(self) -> bool:
        """Initialize Redis connection"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using in-memory cache fallback")
            return True

        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We handle binary data
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )

            # Test connection
            await self.redis_client.ping()
            self.is_connected = True

            logger.info(f"✅ Redis cache connected: {self.redis_url}")
            return True

        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            logger.info("Falling back to in-memory cache")
            self.redis_client = None
            self.is_connected = False
            return True  # Still functional with memory cache

    async def cleanup(self):
        """Cleanup Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
            self.is_connected = False

    def _generate_key(self, namespace: str, key: str) -> str:
        """Generate cache key with namespace"""
        # Create hash for long keys
        if len(key) > 200:
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            return f"exoplanet:{namespace}:{key_hash}"
        else:
            # Clean key for Redis
            clean_key = key.replace(" ", "_").replace(":", "_")
            return f"exoplanet:{namespace}:{clean_key}"

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage using JSON only"""
        try:
            # Convert complex objects to serializable format
            if hasattr(value, "__dict__"):
                value = value.__dict__
            elif hasattr(value, "to_dict"):
                value = value.to_dict()

            return json.dumps(value, default=str).encode("utf-8")
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise CacheError(f"Failed to serialize value: {e}")

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage using JSON only"""
        try:
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise CacheError(f"Failed to deserialize value: {e}")

    async def get(self, namespace: str, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            namespace: Cache namespace (e.g., 'planets', 'lightcurves')
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        cache_key = self._generate_key(namespace, key)

        try:
            if self.is_connected and self.redis_client:
                # Try Redis first
                data = await self.redis_client.get(cache_key)
                if data is not None:
                    self._cache_stats["hits"] += 1
                    return self._deserialize_value(data)

            # Fallback to memory cache
            if cache_key in self._memory_cache:
                cache_entry = self._memory_cache[cache_key]

                # Check TTL
                if datetime.now() < cache_entry["expires"]:
                    self._cache_stats["hits"] += 1
                    return cache_entry["value"]
                else:
                    # Expired, remove from memory cache
                    del self._memory_cache[cache_key]

            self._cache_stats["misses"] += 1
            return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self._cache_stats["errors"] += 1
            return None

    async def set(
        self, namespace: str, key: str, value: Any, ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache

        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: 6 hours)

        Returns:
            True if successful
        """
        cache_key = self._generate_key(namespace, key)
        ttl = ttl or self.default_ttl

        try:
            serialized_value = self._serialize_value(value)

            if self.is_connected and self.redis_client:
                # Set in Redis with TTL
                await self.redis_client.setex(cache_key, ttl, serialized_value)

            # Also set in memory cache as backup
            expires = datetime.now() + timedelta(seconds=ttl)
            self._memory_cache[cache_key] = {"value": value, "expires": expires}

            # Limit memory cache size
            if len(self._memory_cache) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(
                    self._memory_cache.keys(),
                    key=lambda k: self._memory_cache[k]["expires"],
                )[:100]
                for old_key in oldest_keys:
                    del self._memory_cache[old_key]

            self._cache_stats["sets"] += 1
            return True

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self._cache_stats["errors"] += 1
            return False

    async def delete(self, namespace: str, key: str) -> bool:
        """Delete value from cache"""
        cache_key = self._generate_key(namespace, key)

        try:
            deleted = False

            if self.is_connected and self.redis_client:
                result = await self.redis_client.delete(cache_key)
                deleted = result > 0

            # Also delete from memory cache
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
                deleted = True

            if deleted:
                self._cache_stats["deletes"] += 1

            return deleted

        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            self._cache_stats["errors"] += 1
            return False

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace"""
        try:
            count = 0

            if self.is_connected and self.redis_client:
                # Find all keys with namespace prefix
                pattern = f"exoplanet:{namespace}:*"
                keys = await self.redis_client.keys(pattern)

                if keys:
                    count = await self.redis_client.delete(*keys)

            # Clear from memory cache
            memory_keys_to_delete = [
                k
                for k in self._memory_cache.keys()
                if k.startswith(f"exoplanet:{namespace}:")
            ]

            for key in memory_keys_to_delete:
                del self._memory_cache[key]
                count += 1

            logger.info(f"Cleared {count} keys from namespace '{namespace}'")
            return count

        except Exception as e:
            logger.error(f"Cache clear namespace error: {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = dict(self._cache_stats)

        # Calculate hit rate
        total_requests = stats["hits"] + stats["misses"]
        stats["hit_rate"] = (
            stats["hits"] / total_requests if total_requests > 0 else 0.0
        )
        stats["total_requests"] = total_requests

        # Memory cache info
        stats["memory_cache_size"] = len(self._memory_cache)

        # Redis info
        stats["redis_connected"] = self.is_connected

        if self.is_connected and self.redis_client:
            try:
                redis_info = await self.redis_client.info("memory")
                stats["redis_memory_used"] = redis_info.get(
                    "used_memory_human", "unknown"
                )
                stats["redis_keys"] = await self.redis_client.dbsize()
            except Exception as e:
                logger.error(f"Error getting Redis stats: {e}")
                stats["redis_error"] = str(e)

        return stats

    async def prefetch_popular_queries(self):
        """Prefetch popular planet queries"""
        logger.info("Starting prefetch of popular queries...")

        # Import here to avoid circular imports
        from data_sources.registry import get_registry

        registry = get_registry()
        sources = registry.get_available_sources()

        if not sources:
            logger.warning("No data sources available for prefetch")
            return

        prefetch_count = 0

        for query in self.popular_queries:
            try:
                # Check if already cached
                cached = await self.get("planets", query)
                if cached is not None:
                    continue

                # Fetch from first available source
                for source in sources:
                    try:
                        planet_info = await source.fetch_planet_info(query)
                        if planet_info:
                            # Cache for 12 hours (longer than default)
                            await self.set(
                                "planets", query, planet_info.__dict__, ttl=43200
                            )
                            prefetch_count += 1
                            logger.debug(f"Prefetched: {query}")
                            break
                    except Exception as e:
                        logger.debug(
                            f"Prefetch failed for {query} from {source.name}: {e}"
                        )
                        continue

                # Small delay to avoid overwhelming APIs
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Prefetch error for {query}: {e}")

        logger.info(f"Prefetch completed: {prefetch_count} queries cached")

    async def health_check(self) -> Dict[str, Any]:
        """Check cache health"""
        health = {
            "status": "healthy",
            "redis_connected": self.is_connected,
            "memory_cache_size": len(self._memory_cache),
            "timestamp": datetime.now().isoformat(),
        }

        if self.is_connected and self.redis_client:
            try:
                # Test Redis with a ping
                await self.redis_client.ping()
                health["redis_ping"] = "success"
            except Exception as e:
                health["status"] = "degraded"
                health["redis_error"] = str(e)

        return health


# Global cache instance
_cache_manager = CacheManager()


def get_cache() -> CacheManager:
    """Get the global cache manager"""
    return _cache_manager


def get_cache_manager() -> CacheManager:
    """Get the global cache manager (alias for compatibility)"""
    return _cache_manager


async def initialize_cache() -> bool:
    """Initialize the global cache"""
    success = await _cache_manager.initialize()

    if success:
        # Start prefetch in background
        asyncio.create_task(_cache_manager.prefetch_popular_queries())

    return success


async def cleanup_cache():
    """Cleanup the global cache"""
    await _cache_manager.cleanup()
