"""
Enhanced Rate Limiter for the Medical Research Synthesizer.

This module provides a global rate limiter for API requests to prevent
any single user from consuming too many resources.
"""

import time
import asyncio
import logging
import json
import os
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedRateLimiter:
    """
    Enhanced rate limiter for API requests.

    This class provides a global rate limiter that can be used to limit
    the rate of API requests across multiple instances of the application.
    It uses Redis for distributed rate limiting.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """
        Create a singleton instance of the enhanced rate limiter.

        Returns:
            EnhancedRateLimiter: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(EnhancedRateLimiter, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_rate: int = 60,  # 60 requests per minute
        default_burst: int = 10,  # 10 requests in a burst
        default_window: int = 60,  # 1 minute window
        namespace: str = "asf:medical:rate_limit:"
    ):
        """
        Initialize the enhanced rate limiter.

        Args:
            redis_url: Redis URL for distributed rate limiting (default: from env var REDIS_URL)
            default_rate: Default rate limit in requests per window (default: 60)
            default_burst: Default burst limit in requests (default: 10)
            default_window: Default window size in seconds (default: 60)
            namespace: Cache namespace prefix (default: "asf:medical:rate_limit:")
        """
        if self._initialized:
            return

        # Get Redis URL from environment variable if not provided
        self.redis_url = redis_url or os.environ.get("REDIS_URL")

        self.default_rate = default_rate
        self.default_burst = default_burst
        self.default_window = default_window
        self.namespace = namespace
        self.redis = None

        # Local rate limiting (fallback if Redis is not available)
        self.local_limits = {}
        self.local_tokens = {}
        self.local_last_refill = {}
        self.local_counters = {}
        self.lock = asyncio.Lock()

        # Initialize Redis if URL is provided
        if self.redis_url:
            try:
                import redis.asyncio as aioredis
                self.redis = aioredis.from_url(
                    self.redis_url,
                    decode_responses=True,  # Automatically decode responses to strings
                    socket_timeout=5.0,     # Socket timeout
                    socket_connect_timeout=5.0,  # Connection timeout
                    retry_on_timeout=True,  # Retry on timeout
                    health_check_interval=30  # Health check interval
                )
                logger.info(f"Redis rate limiter initialized: {self.redis_url}")
            except ImportError:
                logger.error("redis-py package not installed. Install with: pip install redis")
                logger.warning("Falling back to local rate limiting")
            except Exception as e:
                logger.error(f"Failed to initialize Redis rate limiter: {str(e)}")
                logger.warning("Falling back to local rate limiting")
        else:
            logger.warning("No Redis URL provided. Falling back to local rate limiting")

        self._initialized = True
        logger.info("Enhanced rate limiter initialized")

    async def is_rate_limited(
        self,
        key: str,
        rate: Optional[int] = None,
        burst: Optional[int] = None,
        window: Optional[int] = None,
        cost: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a key is rate limited.

        Args:
            key: Rate limit key (e.g., user ID, IP address)
            rate: Rate limit in requests per window (default: self.default_rate)
            burst: Burst limit in requests (default: self.default_burst)
            window: Window size in seconds (default: self.default_window)
            cost: Cost of the request (default: 1)

        Returns:
            Tuple of (is_limited, limit_info)
        """
        # Use default values if not provided
        rate = rate or self.default_rate
        burst = burst or self.default_burst
        window = window or self.default_window

        # Apply namespace
        namespaced_key = f"{self.namespace}{key}"

        # Try Redis first if available
        if self.redis:
            try:
                # Use Redis for distributed rate limiting
                return await self._check_redis_rate_limit(namespaced_key, rate, burst, window, cost)
            except Exception as e:
                logger.error(f"Error checking Redis rate limit: {str(e)}")
                logger.warning("Falling back to local rate limiting")

        # Fall back to local rate limiting
        return await self._check_local_rate_limit(namespaced_key, rate, burst, window, cost)

    async def _check_redis_rate_limit(
        self,
        key: str,
        rate: int,
        burst: int,
        window: int,
        cost: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a key is rate limited using Redis.

        Args:
            key: Rate limit key
            rate: Rate limit in requests per window
            burst: Burst limit in requests
            window: Window size in seconds
            cost: Cost of the request

        Returns:
            Tuple of (is_limited, limit_info)
        """
        # Get current time
        now = time.time()

        # Get current tokens
        tokens_key = f"{key}:tokens"
        last_refill_key = f"{key}:last_refill"

        # Get current tokens and last refill time
        async with self.redis.pipeline() as pipe:
            await pipe.get(tokens_key)
            await pipe.get(last_refill_key)
            tokens_str, last_refill_str = await pipe.execute()

        # Parse values
        tokens = float(tokens_str) if tokens_str else burst
        last_refill = float(last_refill_str) if last_refill_str else now

        # Refill tokens based on time elapsed
        time_elapsed = now - last_refill
        new_tokens = time_elapsed * (rate / window)
        tokens = min(tokens + new_tokens, burst)

        # Check if enough tokens are available
        is_limited = tokens < cost

        if not is_limited:
            # Consume tokens
            tokens -= cost

        # Update tokens and last refill time
        async with self.redis.pipeline() as pipe:
            await pipe.set(tokens_key, tokens)
            await pipe.set(last_refill_key, now)
            await pipe.expire(tokens_key, window * 2)  # Set expiry to 2x window
            await pipe.expire(last_refill_key, window * 2)  # Set expiry to 2x window
            await pipe.execute()

        # Calculate reset time
        if tokens < cost:
            # Calculate time until enough tokens are available
            time_until_refill = (cost - tokens) * (window / rate)
            reset_time = now + time_until_refill
        else:
            # Calculate time until full refill
            time_until_full = (burst - tokens) * (window / rate)
            reset_time = now + time_until_full

        # Create limit info
        limit_info = {
            "limit": rate,
            "remaining": int(tokens / cost),
            "reset": int(reset_time),
            "window": window,
            "burst": burst
        }

        return is_limited, limit_info

    async def _check_local_rate_limit(
        self,
        key: str,
        rate: int,
        burst: int,
        window: int,
        cost: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a key is rate limited using local storage.

        Args:
            key: Rate limit key
            rate: Rate limit in requests per window
            burst: Burst limit in requests
            window: Window size in seconds
            cost: Cost of the request

        Returns:
            Tuple of (is_limited, limit_info)
        """
        async with self.lock:
            # Get current time
            now = time.time()

            # Initialize if key doesn't exist
            if key not in self.local_tokens:
                self.local_limits[key] = {"rate": rate, "burst": burst, "window": window}
                self.local_tokens[key] = burst
                self.local_last_refill[key] = now

            # Refill tokens based on time elapsed
            last_refill = self.local_last_refill[key]
            time_elapsed = now - last_refill
            new_tokens = time_elapsed * (rate / window)
            self.local_tokens[key] = min(self.local_tokens[key] + new_tokens, burst)
            self.local_last_refill[key] = now

            # Check if enough tokens are available
            is_limited = self.local_tokens[key] < cost

            if not is_limited:
                # Consume tokens
                self.local_tokens[key] -= cost

            # Calculate reset time
            if self.local_tokens[key] < cost:
                # Calculate time until enough tokens are available
                time_until_refill = (cost - self.local_tokens[key]) * (window / rate)
                reset_time = now + time_until_refill
            else:
                # Calculate time until full refill
                time_until_full = (burst - self.local_tokens[key]) * (window / rate)
                reset_time = now + time_until_full

            # Create limit info
            limit_info = {
                "limit": rate,
                "remaining": int(self.local_tokens[key] / cost),
                "reset": int(reset_time),
                "window": window,
                "burst": burst
            }

            return is_limited, limit_info

    async def get_limits(self, key: str) -> Dict[str, Any]:
        """
        Get rate limit information for a key.

        Args:
            key: Rate limit key (e.g., user ID, IP address)

        Returns:
            Dict[str, Any]: Rate limit information
        """
        # Apply namespace
        namespaced_key = f"{self.namespace}{key}"

        # Try Redis first if available
        if self.redis:
            try:
                # Get tokens and last refill time
                tokens_key = f"{namespaced_key}:tokens"
                last_refill_key = f"{namespaced_key}:last_refill"

                async with self.redis.pipeline() as pipe:
                    await pipe.get(tokens_key)
                    await pipe.get(last_refill_key)
                    tokens_str, last_refill_str = await pipe.execute()

                if tokens_str is None:
                    # Key doesn't exist
                    return {
                        "limit": self.default_rate,
                        "remaining": self.default_rate,
                        "reset": int(time.time() + self.default_window),
                        "window": self.default_window,
                        "burst": self.default_burst
                    }

                # Parse values
                tokens = float(tokens_str) if tokens_str else self.default_burst
                last_refill = float(last_refill_str) if last_refill_str else time.time()

                # Calculate reset time
                now = time.time()
                time_until_full = (self.default_burst - tokens) * (self.default_window / self.default_rate)
                reset_time = now + time_until_full

                return {
                    "limit": self.default_rate,
                    "remaining": int(tokens),
                    "reset": int(reset_time),
                    "window": self.default_window,
                    "burst": self.default_burst
                }
            except Exception as e:
                logger.error(f"Error getting Redis rate limit info: {str(e)}")
                logger.warning("Falling back to local rate limit info")

        # Fall back to local rate limit info
        async with self.lock:
            if namespaced_key not in self.local_tokens:
                # Key doesn't exist
                return {
                    "limit": self.default_rate,
                    "remaining": self.default_rate,
                    "reset": int(time.time() + self.default_window),
                    "window": self.default_window,
                    "burst": self.default_burst
                }

            # Calculate reset time
            now = time.time()
            time_until_full = (self.default_burst - self.local_tokens[namespaced_key]) * (self.default_window / self.default_rate)
            reset_time = now + time_until_full

            return {
                "limit": self.local_limits[namespaced_key]["rate"],
                "remaining": int(self.local_tokens[namespaced_key]),
                "reset": int(reset_time),
                "window": self.local_limits[namespaced_key]["window"],
                "burst": self.local_limits[namespaced_key]["burst"]
            }

    async def reset_limits(self, key: str) -> bool:
        """
        Reset rate limit for a key.

        Args:
            key: Rate limit key (e.g., user ID, IP address)

        Returns:
            bool: True if the rate limit was reset, False otherwise
        """
        # Apply namespace
        namespaced_key = f"{self.namespace}{key}"

        # Try Redis first if available
        if self.redis:
            try:
                # Delete tokens and last refill time
                tokens_key = f"{namespaced_key}:tokens"
                last_refill_key = f"{namespaced_key}:last_refill"

                async with self.redis.pipeline() as pipe:
                    await pipe.delete(tokens_key)
                    await pipe.delete(last_refill_key)
                    await pipe.execute()

                return True
            except Exception as e:
                logger.error(f"Error resetting Redis rate limit: {str(e)}")
                logger.warning("Falling back to local rate limit reset")

        # Fall back to local rate limit reset
        async with self.lock:
            if namespaced_key in self.local_tokens:
                del self.local_tokens[namespaced_key]
                del self.local_last_refill[namespaced_key]
                if namespaced_key in self.local_limits:
                    del self.local_limits[namespaced_key]
                return True
            return False

    async def increment_counter(
        self,
        key: str,
        window: int = 60,
        increment: int = 1
    ) -> int:
        """
        Increment a counter and return the new value.

        Args:
            key: Counter key
            window: Window size in seconds
            increment: Increment value

        Returns:
            int: New counter value
        """
        # Apply namespace
        namespaced_key = f"{self.namespace}counter:{key}"

        # Try Redis first if available
        if self.redis:
            try:
                # Use Redis for distributed counter
                value = await self.redis.incrby(namespaced_key, increment)
                await self.redis.expire(namespaced_key, window)
                return int(value)
            except Exception as e:
                logger.error(f"Error incrementing Redis counter: {str(e)}")
                logger.warning("Falling back to local counter")

        # Fall back to local counter
        async with self.lock:
            if namespaced_key not in self.local_counters:
                self.local_counters[namespaced_key] = 0

            self.local_counters[namespaced_key] += increment
            return self.local_counters[namespaced_key]

    async def get_counter(
        self,
        key: str
    ) -> int:
        """
        Get a counter value.

        Args:
            key: Counter key

        Returns:
            int: Counter value
        """
        # Apply namespace
        namespaced_key = f"{self.namespace}counter:{key}"

        # Try Redis first if available
        if self.redis:
            try:
                # Use Redis for distributed counter
                value = await self.redis.get(namespaced_key)
                return int(value) if value else 0
            except Exception as e:
                logger.error(f"Error getting Redis counter: {str(e)}")
                logger.warning("Falling back to local counter")

        # Fall back to local counter
        async with self.lock:
            return self.local_counters.get(namespaced_key, 0)

    async def reset_counter(
        self,
        key: str
    ) -> None:
        """
        Reset a counter to zero.

        Args:
            key: Counter key
        """
        # Apply namespace
        namespaced_key = f"{self.namespace}counter:{key}"

        # Try Redis first if available
        if self.redis:
            try:
                # Use Redis for distributed counter
                await self.redis.delete(namespaced_key)
            except Exception as e:
                logger.error(f"Error resetting Redis counter: {str(e)}")
                logger.warning("Falling back to local counter")

        # Fall back to local counter
        async with self.lock:
            if namespaced_key in self.local_counters:
                del self.local_counters[namespaced_key]

    async def update_limits(
        self,
        key: str,
        rate: Optional[int] = None,
        burst: Optional[int] = None,
        window: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Update rate limit for a key.

        Args:
            key: Rate limit key (e.g., user ID, IP address)
            rate: Rate limit in requests per window (default: unchanged)
            burst: Burst limit in requests (default: unchanged)
            window: Window size in seconds (default: unchanged)

        Returns:
            Dict[str, Any]: Updated rate limit information
        """
        # Apply namespace
        namespaced_key = f"{self.namespace}{key}"

        # Get current limits
        current_limits = await self.get_limits(key)

        # Update limits
        new_rate = rate if rate is not None else current_limits["limit"]
        new_burst = burst if burst is not None else current_limits["burst"]
        new_window = window if window is not None else current_limits["window"]

        # Try Redis first if available
        if self.redis:
            try:
                # Store limits in Redis
                limits_key = f"{namespaced_key}:limits"
                limits = {
                    "rate": new_rate,
                    "burst": new_burst,
                    "window": new_window
                }

                await self.redis.set(limits_key, json.dumps(limits))
                await self.redis.expire(limits_key, new_window * 2)  # Set expiry to 2x window

                # Reset tokens to burst
                tokens_key = f"{namespaced_key}:tokens"
                last_refill_key = f"{namespaced_key}:last_refill"

                async with self.redis.pipeline() as pipe:
                    await pipe.set(tokens_key, new_burst)
                    await pipe.set(last_refill_key, time.time())
                    await pipe.expire(tokens_key, new_window * 2)  # Set expiry to 2x window
                    await pipe.expire(last_refill_key, new_window * 2)  # Set expiry to 2x window
                    await pipe.execute()

                return {
                    "limit": new_rate,
                    "remaining": new_burst,
                    "reset": int(time.time() + new_window),
                    "window": new_window,
                    "burst": new_burst
                }
            except Exception as e:
                logger.error(f"Error updating Redis rate limit: {str(e)}")
                logger.warning("Falling back to local rate limit update")

        # Fall back to local rate limit update
        async with self.lock:
            self.local_limits[namespaced_key] = {
                "rate": new_rate,
                "burst": new_burst,
                "window": new_window
            }
            self.local_tokens[namespaced_key] = new_burst
            self.local_last_refill[namespaced_key] = time.time()

            return {
                "limit": new_rate,
                "remaining": new_burst,
                "reset": int(time.time() + new_window),
                "window": new_window,
                "burst": new_burst
            }

# Create a singleton instance
enhanced_rate_limiter = EnhancedRateLimiter()
