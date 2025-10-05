"""
Enterprise-grade rate limiting system for ExoplanetAI
Система ограничения скорости запросов уровня enterprise для ExoplanetAI
"""

import asyncio
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import redis.asyncio as redis
import logging

from .config import get_settings
from auth.models import UserRole

logger = logging.getLogger(__name__)


class RateLimitType(str, Enum):
    """Types of rate limits"""
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"
    BURST = "burst"


@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""
    limit: int
    window_seconds: int
    limit_type: RateLimitType
    description: str


class RateLimitConfig:
    """Enterprise rate limiting configuration based on user roles and endpoints"""
    
    # Default limits for different user roles
    ROLE_LIMITS = {
        UserRole.GUEST: {
            RateLimitType.PER_MINUTE: RateLimitRule(10, 60, RateLimitType.PER_MINUTE, "Guest users: 10 requests per minute"),
            RateLimitType.PER_HOUR: RateLimitRule(100, 3600, RateLimitType.PER_HOUR, "Guest users: 100 requests per hour"),
            RateLimitType.BURST: RateLimitRule(3, 10, RateLimitType.BURST, "Guest users: max 3 requests in 10 seconds")
        },
        UserRole.USER: {
            RateLimitType.PER_MINUTE: RateLimitRule(30, 60, RateLimitType.PER_MINUTE, "Regular users: 30 requests per minute"),
            RateLimitType.PER_HOUR: RateLimitRule(500, 3600, RateLimitType.PER_HOUR, "Regular users: 500 requests per hour"),
            RateLimitType.BURST: RateLimitRule(10, 10, RateLimitType.BURST, "Regular users: max 10 requests in 10 seconds")
        },
        UserRole.RESEARCHER: {
            RateLimitType.PER_MINUTE: RateLimitRule(100, 60, RateLimitType.PER_MINUTE, "Researchers: 100 requests per minute"),
            RateLimitType.PER_HOUR: RateLimitRule(2000, 3600, RateLimitType.PER_HOUR, "Researchers: 2000 requests per hour"),
            RateLimitType.BURST: RateLimitRule(20, 10, RateLimitType.BURST, "Researchers: max 20 requests in 10 seconds")
        },
        UserRole.ADMIN: {
            RateLimitType.PER_MINUTE: RateLimitRule(500, 60, RateLimitType.PER_MINUTE, "Admins: 500 requests per minute"),
            RateLimitType.PER_HOUR: RateLimitRule(10000, 3600, RateLimitType.PER_HOUR, "Admins: 10000 requests per hour"),
            RateLimitType.BURST: RateLimitRule(50, 10, RateLimitType.BURST, "Admins: max 50 requests in 10 seconds")
        }
    }
    
    # Endpoint-specific limits (more restrictive for resource-intensive operations)
    ENDPOINT_LIMITS = {
        "/api/v1/data/ingest/table": {
            UserRole.GUEST: RateLimitRule(2, 3600, RateLimitType.PER_HOUR, "Table ingestion: 2 per hour for guests"),
            UserRole.USER: RateLimitRule(10, 3600, RateLimitType.PER_HOUR, "Table ingestion: 10 per hour for users"),
            UserRole.RESEARCHER: RateLimitRule(50, 3600, RateLimitType.PER_HOUR, "Table ingestion: 50 per hour for researchers"),
            UserRole.ADMIN: RateLimitRule(200, 3600, RateLimitType.PER_HOUR, "Table ingestion: 200 per hour for admins")
        },
        "/api/v1/data/ingest/batch": {
            UserRole.GUEST: RateLimitRule(1, 86400, RateLimitType.PER_DAY, "Batch ingestion: 1 per day for guests"),
            UserRole.USER: RateLimitRule(3, 86400, RateLimitType.PER_DAY, "Batch ingestion: 3 per day for users"),
            UserRole.RESEARCHER: RateLimitRule(10, 86400, RateLimitType.PER_DAY, "Batch ingestion: 10 per day for researchers"),
            UserRole.ADMIN: RateLimitRule(50, 86400, RateLimitType.PER_DAY, "Batch ingestion: 50 per day for admins")
        },
        "/api/v1/data/preprocess/lightcurve": {
            UserRole.GUEST: RateLimitRule(5, 3600, RateLimitType.PER_HOUR, "Light curve processing: 5 per hour for guests"),
            UserRole.USER: RateLimitRule(20, 3600, RateLimitType.PER_HOUR, "Light curve processing: 20 per hour for users"),
            UserRole.RESEARCHER: RateLimitRule(100, 3600, RateLimitType.PER_HOUR, "Light curve processing: 100 per hour for researchers"),
            UserRole.ADMIN: RateLimitRule(500, 3600, RateLimitType.PER_HOUR, "Light curve processing: 500 per hour for admins")
        }
    }


class EnterpriseRateLimiter:
    """
    Enterprise-grade rate limiter with Redis backend and fallback to memory
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache: Dict[str, List[float]] = defaultdict(list)
        self.config = RateLimitConfig()
        
    async def initialize(self) -> bool:
        """Initialize Redis connection"""
        try:
            if hasattr(self.settings, 'redis_url') and self.settings.redis_url:
                self.redis_client = redis.from_url(self.settings.redis_url)
                # Test connection
                await self.redis_client.ping()
                logger.info("✅ Rate limiter initialized with Redis backend")
                return True
            else:
                logger.warning("⚠️ Redis not configured, using memory-based rate limiting")
                return True
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}, falling back to memory")
            self.redis_client = None
            return True
    
    async def check_rate_limit(
        self, 
        identifier: str, 
        endpoint: str, 
        user_role: Optional[UserRole] = None
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check if request is within rate limits
        
        Returns:
            Tuple[bool, dict]: (is_allowed, limit_info)
        """
        if user_role is None:
            user_role = UserRole.GUEST
            
        # Get applicable limits
        limits_to_check = []
        
        # Add role-based limits
        if user_role in self.config.ROLE_LIMITS:
            limits_to_check.extend(self.config.ROLE_LIMITS[user_role].values())
        
        # Add endpoint-specific limits
        if endpoint in self.config.ENDPOINT_LIMITS:
            if user_role in self.config.ENDPOINT_LIMITS[endpoint]:
                limits_to_check.append(self.config.ENDPOINT_LIMITS[endpoint][user_role])
        
        # Check each limit
        limit_info = {
            "allowed": True,
            "limits_checked": len(limits_to_check),
            "violations": [],
            "remaining": {},
            "reset_times": {}
        }
        
        for rule in limits_to_check:
            is_allowed, remaining, reset_time = await self._check_single_limit(
                identifier, endpoint, rule
            )
            
            if not is_allowed:
                limit_info["allowed"] = False
                limit_info["violations"].append({
                    "rule": rule.description,
                    "limit": rule.limit,
                    "window_seconds": rule.window_seconds,
                    "remaining": remaining,
                    "reset_time": reset_time
                })
            
            limit_info["remaining"][rule.limit_type.value] = remaining
            limit_info["reset_times"][rule.limit_type.value] = reset_time
        
        return limit_info["allowed"], limit_info
    
    async def _check_single_limit(
        self, 
        identifier: str, 
        endpoint: str, 
        rule: RateLimitRule
    ) -> Tuple[bool, int, float]:
        """
        Check a single rate limit rule
        
        Returns:
            Tuple[bool, int, float]: (is_allowed, remaining_requests, reset_timestamp)
        """
        now = time.time()
        window_start = now - rule.window_seconds
        key = f"rate_limit:{identifier}:{endpoint}:{rule.limit_type.value}"
        
        if self.redis_client:
            return await self._check_redis_limit(key, rule, now, window_start)
        else:
            return await self._check_memory_limit(key, rule, now, window_start)
    
    async def _check_redis_limit(
        self, 
        key: str, 
        rule: RateLimitRule, 
        now: float, 
        window_start: float
    ) -> Tuple[bool, int, float]:
        """Check rate limit using Redis backend"""
        try:
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current entries
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(now): now})
            
            # Set expiration
            pipe.expire(key, rule.window_seconds + 60)  # Extra buffer
            
            results = await pipe.execute()
            current_count = results[1]
            
            remaining = max(0, rule.limit - current_count - 1)  # -1 for current request
            reset_time = now + rule.window_seconds
            is_allowed = current_count < rule.limit
            
            if not is_allowed:
                # Remove the request we just added since it's not allowed
                await self.redis_client.zrem(key, str(now))
            
            return is_allowed, remaining, reset_time
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}, falling back to memory")
            return await self._check_memory_limit(key, rule, now, window_start)
    
    async def _check_memory_limit(
        self, 
        key: str, 
        rule: RateLimitRule, 
        now: float, 
        window_start: float
    ) -> Tuple[bool, int, float]:
        """Check rate limit using memory backend"""
        # Clean old entries
        self.memory_cache[key] = [
            timestamp for timestamp in self.memory_cache[key] 
            if timestamp > window_start
        ]
        
        current_count = len(self.memory_cache[key])
        remaining = max(0, rule.limit - current_count - 1)
        reset_time = now + rule.window_seconds
        is_allowed = current_count < rule.limit
        
        if is_allowed:
            self.memory_cache[key].append(now)
        
        return is_allowed, remaining, reset_time
    
    async def get_rate_limit_status(
        self, 
        identifier: str, 
        endpoint: str, 
        user_role: Optional[UserRole] = None
    ) -> Dict[str, any]:
        """Get current rate limit status without making a request"""
        if user_role is None:
            user_role = UserRole.GUEST
            
        status = {
            "identifier": identifier,
            "endpoint": endpoint,
            "user_role": user_role.value,
            "limits": {}
        }
        
        # Get role-based limits status
        if user_role in self.config.ROLE_LIMITS:
            for limit_type, rule in self.config.ROLE_LIMITS[user_role].items():
                key = f"rate_limit:{identifier}:{endpoint}:{limit_type.value}"
                now = time.time()
                window_start = now - rule.window_seconds
                
                if self.redis_client:
                    try:
                        await self.redis_client.zremrangebyscore(key, 0, window_start)
                        current_count = await self.redis_client.zcard(key)
                    except:
                        current_count = len([
                            t for t in self.memory_cache[key] if t > window_start
                        ])
                else:
                    current_count = len([
                        t for t in self.memory_cache[key] if t > window_start
                    ])
                
                status["limits"][limit_type.value] = {
                    "limit": rule.limit,
                    "used": current_count,
                    "remaining": max(0, rule.limit - current_count),
                    "reset_time": now + rule.window_seconds,
                    "description": rule.description
                }
        
        return status
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()
        self.memory_cache.clear()


# Global rate limiter instance
_rate_limiter: Optional[EnterpriseRateLimiter] = None


async def get_rate_limiter() -> EnterpriseRateLimiter:
    """Get or create global rate limiter instance"""
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = EnterpriseRateLimiter()
        await _rate_limiter.initialize()
    
    return _rate_limiter


async def cleanup_rate_limiter():
    """Cleanup global rate limiter"""
    global _rate_limiter
    
    if _rate_limiter:
        await _rate_limiter.cleanup()
        _rate_limiter = None
