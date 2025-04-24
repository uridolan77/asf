"""
Rate Limiter module for the Medical Research Synthesizer.

This module provides a unified rate limiting system with both local and distributed (Redis) 
implementations. The rate limiters control the frequency of requests or operations
within the application to prevent abuse and ensure system stability.

Classes:
    RateLimitExceededError: Exception raised when a rate limit is exceeded.
    TokenBucketRateLimiter: Synchronous token bucket rate limiter.
    AsyncTokenBucketRateLimiter: Asynchronous token bucket rate limiter.
    DistributedRateLimiter: Redis-based distributed rate limiter.
    RateLimiterManager: Unified manager supporting both local and distributed rate limiting.

Functions:
    rate_limit: Decorator for applying rate limiting to functions.
    global_rate_limit: Decorator for applying distributed rate limiting across app instances.
"""

import os
import time
import json
import asyncio
import logging
import threading
from functools import wraps
from typing import Optional, Callable, Dict, Any, Tuple, List, Union

logger = logging.getLogger(__name__)

class RateLimitExceededError(Exception):
    """
    Exception raised when a rate limit is exceeded.

    Attributes:
        message (str): The error message.
        key (str): The rate limit key.
        limit (int): The rate limit.
        reset_time (float): Time when the rate limit will reset.
    """

    def __init__(self, message: str, key: str = None, limit: int = None, reset_time: float = None):
        """
        Initialize a RateLimitExceededError.

        Args:
            message (str): The error message.
            key (str, optional): The rate limit key. Defaults to None.
            limit (int, optional): The rate limit. Defaults to None.
            reset_time (float, optional): Time when the rate limit will reset. Defaults to None.
        """
        self.message = message
        self.key = key
        self.limit = limit
        self.reset_time = reset_time
        super().__init__(message)

class TokenBucketRateLimiter:
    """
    Token bucket rate limiter implementation.
    
    This class provides methods to enforce rate limits for different operations 
    or resources, with support for varying rates based on keys.

    Attributes:
        default_rate (int): Default number of tokens per minute.
        default_burst (int): Default maximum token capacity.
        buckets (Dict[str, Dict[str, float]]): Map of keys to token buckets.
        lock (threading.RLock): Lock for thread-safe access to buckets.
    """

    def __init__(self, default_rate: int = 60, default_burst: int = 10):
        """
        Initialize the TokenBucketRateLimiter instance.

        Args:
            default_rate (int, optional): Default number of tokens per minute. Defaults to 60.
            default_burst (int, optional): Default maximum token capacity. Defaults to 10.
        """
        self.default_rate = default_rate
        self.default_burst = default_burst
        self.buckets = {}
        self.last_refill = {}
        self.lock = threading.RLock()

    def allow(self, key: str, cost: float = 1.0, rate: int = None, burst: int = None) -> bool:
        """
        Check if an operation should be allowed based on rate limits.

        Args:
            key (str): Key to identify the rate limit bucket.
            cost (float, optional): Cost of the operation in tokens. Defaults to 1.0.
            rate (int, optional): Tokens per minute for this key. Defaults to None.
            burst (int, optional): Maximum token capacity for this key. Defaults to None.

        Returns:
            bool: True if the operation is allowed, False otherwise.
        """
        with self.lock:
            return self._check_limit(key, cost, rate, burst)

    def try_acquire(self, key: str, cost: float = 1.0, rate: int = None, burst: int = None) -> bool:
        """
        Try to acquire tokens for an operation.

        Args:
            key (str): Key to identify the rate limit bucket.
            cost (float, optional): Cost of the operation in tokens. Defaults to 1.0.
            rate (int, optional): Tokens per minute for this key. Defaults to None.
            burst (int, optional): Maximum token capacity for this key. Defaults to None.

        Returns:
            bool: True if tokens were acquired, False otherwise.
        """
        with self.lock:
            allowed = self._check_limit(key, cost, rate, burst)
            if allowed:
                self._update_tokens(key, -cost)
            return allowed
            
    def get_token_count(self, key: str) -> float:
        """
        Get the current token count for a specific key.

        Args:
            key (str): Key to identify the rate limit bucket.

        Returns:
            float: Current token count.
        """
        with self.lock:
            self._refill(key)
            return self.buckets.get(key, self.default_burst)
            
    def update_limits(self, key: str, rate: int, burst: int) -> None:
        """
        Update rate limits for a specific key.

        Args:
            key (str): Key to identify the rate limit bucket.
            rate (int): New tokens per minute rate.
            burst (int): New maximum token capacity.
        """
        with self.lock:
            self.buckets[key] = burst
            self._rate_for_key(key, rate=rate)
            self._burst_for_key(key, burst=burst)

    def reset(self, key: str = None) -> None:
        """
        Reset rate limits for a specific key or all keys.

        Args:
            key (str, optional): Key to reset limits for. Defaults to None (reset all).
        """
        with self.lock:
            if key:
                if key in self.buckets:
                    del self.buckets[key]
                if key in self.last_refill:
                    del self.last_refill[key]
            else:
                self.buckets.clear()
                self.last_refill.clear()

    def _check_limit(self, key: str, cost: float, rate: int = None, burst: int = None) -> bool:
        """
        Check if an operation is allowed under rate limit and refill tokens.
        
        Args:
            key: Rate limit key
            cost: Operation cost in tokens
            rate: Optional rate override
            burst: Optional burst size override
            
        Returns:
            True if allowed, False otherwise
        """
        self._refill(key)
        tokens = self.buckets.get(key, self._burst_for_key(key, burst))
        return tokens >= cost

    def _update_tokens(self, key: str, amount: float) -> None:
        """
        Update the token count for a key.
        
        Args:
            key: Rate limit key
            amount: Amount to adjust tokens by (negative for consuming)
        """
        if key not in self.buckets:
            self.buckets[key] = self._burst_for_key(key)
        
        self.buckets[key] = max(0, min(
            self.buckets[key] + amount,
            self._burst_for_key(key)
        ))

    def _refill(self, key: str) -> None:
        """
        Refill tokens for a key based on elapsed time.
        
        Args:
            key: Rate limit key
        """
        now = time.time()
        last = self.last_refill.get(key, now)
        rate = self._rate_for_key(key)
        
        # Calculate tokens to add (tokens per second * elapsed seconds)
        elapsed = now - last
        to_add = elapsed * (rate / 60.0)
        
        if to_add > 0:
            self.last_refill[key] = now
            if key not in self.buckets:
                self.buckets[key] = self._burst_for_key(key)
            self.buckets[key] = min(
                self.buckets[key] + to_add,
                self._burst_for_key(key)
            )
            
    def _rate_for_key(self, key: str, rate: int = None) -> int:
        """Get the rate for a key."""
        return rate if rate is not None else self.default_rate
        
    def _burst_for_key(self, key: str, burst: int = None) -> int:
        """Get the burst size for a key."""
        return burst if burst is not None else self.default_burst

class AsyncTokenBucketRateLimiter:
    """
    Asynchronous token bucket rate limiter for API clients.
    
    This class provides a rate limiter that can be used to limit the rate of API requests.
    It uses a token bucket algorithm to limit the rate of requests.

    Attributes:
        requests_per_second (float): Number of requests allowed per second.
        burst_size (int): Maximum number of tokens in the bucket.
        tokens (float): Current number of tokens in the bucket.
        last_refill_time (float): Last time the bucket was refilled.
        lock (asyncio.Lock): Lock for thread-safe access to the bucket.
    """
    
    def __init__(
        self,
        requests_per_second: float,
        burst_size: Optional[int] = None
    ):
        """
        Initialize the AsyncTokenBucketRateLimiter instance.

        Args:
            requests_per_second (float): Number of requests allowed per second.
            burst_size (int, optional): Maximum number of tokens in the bucket. Defaults to 2 * requests_per_second.
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size or int(2 * requests_per_second)
        self.tokens = self.burst_size
        self.last_refill_time = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1, wait: bool = True) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens (int, optional): Number of tokens to acquire. Defaults to 1.
            wait (bool, optional): Whether to wait for tokens to become available. Defaults to True.

        Returns:
            bool: True if tokens were acquired, False if not acquired and wait is False.

        Raises:
            ValueError: If the number of tokens requested exceeds the burst size.
        """
        if tokens > self.burst_size:
            raise ValueError(f"Cannot acquire more tokens than burst size ({self.burst_size})")
        
        async with self.lock:
            await self._refill()
            
            if not wait and self.tokens < tokens:
                return False
                
            while self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.requests_per_second
                
                self.lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self.lock.acquire()
                
                await self._refill()
            
            self.tokens -= tokens
            return True
    
    async def _refill(self) -> None:
        """
        Refill the token bucket based on the elapsed time.
        """
        now = time.time()
        time_elapsed = now - self.last_refill_time
        
        new_tokens = time_elapsed * self.requests_per_second
        
        if new_tokens > 0:
            self.tokens = min(self.tokens + new_tokens, self.burst_size)
            self.last_refill_time = now
            
    async def get_token_count(self) -> float:
        """
        Get the current number of tokens in the bucket.
        
        Returns:
            float: Current number of tokens.
        """
        async with self.lock:
            await self._refill()
            return self.tokens
            
    async def wait_time_for_tokens(self, tokens: int = 1) -> float:
        """
        Calculate wait time for tokens to become available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            float: Wait time in seconds, 0 if tokens are available now
        """
        async with self.lock:
            await self._refill()
            if self.tokens >= tokens:
                return 0
            return (tokens - self.tokens) / self.requests_per_second

class DistributedRateLimiter:
    """
    Distributed rate limiter using Redis.

    This class provides methods to enforce rate limits across multiple instances
    of the application using Redis as a shared storage.

    Attributes:
        redis_url (str): URL for the Redis server.
        default_rate (int): Default maximum number of requests allowed per minute.
        default_burst (int): Default burst capacity for requests.
        default_window (int): Default time window in seconds for rate limiting.
        namespace (str): Namespace for rate limiting keys.
        redis (Optional[Redis]): Redis client instance.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_rate: int = 60,  # 60 requests per minute
        default_burst: int = 10,  # 10 requests in a burst
        default_window: int = 60,  # 1 minute window
        namespace: str = "asf:medical:rate_limit:"
    ):
        """
        Initialize the DistributedRateLimiter instance.

        Args:
            redis_url (Optional[str]): URL for the Redis server. Defaults to None (from env).
            default_rate (int): Default max number of requests per minute. Defaults to 60.
            default_burst (int): Default burst capacity. Defaults to 10.
            default_window (int): Default time window in seconds. Defaults to 60.
            namespace (str): Namespace for rate limiting keys. Defaults to "asf:medical:rate_limit:".
        """
        self.redis_url = redis_url or os.environ.get("REDIS_URL")
        
        self.default_rate = default_rate
        self.default_burst = default_burst
        self.default_window = default_window
        self.namespace = namespace
        self.redis = None
        
        # Local fallback mechanisms
        self.local_limits = {}  # Key -> {rate, burst, window}
        self.local_tokens = {}  # Key -> current token count
        self.local_last_refill = {}  # Key -> timestamp of last refill
        self.lock = asyncio.Lock()
        
        if self.redis_url:
            try:
                import redis.asyncio as aioredis
                self.redis = aioredis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                logger.info(f"Redis rate limiter initialized: {self.redis_url}")
            except ImportError:
                logger.error("redis-py package not installed. Install with: pip install redis")
                logger.warning("Falling back to local rate limiting")
            except Exception as e:
                logger.error(f"Failed to initialize Redis rate limiter: {str(e)}")
                logger.warning("Falling back to local rate limiting")
        else:
            logger.warning("No Redis URL provided. Using local rate limiting only.")
    
    async def is_rate_limited(
        self,
        key: str,
        rate: Optional[int] = None,
        burst: Optional[int] = None,
        window: Optional[int] = None,
        cost: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a request is rate-limited.

        Args:
            key (str): Unique key to identify the request source.
            rate (Optional[int]): Maximum number of requests allowed per minute.
            burst (Optional[int]): Burst capacity for requests.
            window (Optional[int]): Time window in seconds for rate limiting.
            cost (int): Cost of the request in terms of tokens.

        Returns:
            Tuple[bool, Dict[str, Any]]: A tuple containing:
                - boolean indicating if the request is rate-limited (True = limited)
                - dictionary with rate limit information
        """
        rate = rate or self.default_rate
        burst = burst or self.default_burst
        window = window or self.default_window

        namespaced_key = f"{self.namespace}{key}"

        if self.redis:
            try:
                return await self._check_redis_rate_limit(namespaced_key, rate, burst, window, cost)
            except Exception as e:
                logger.error(f"Error checking Redis rate limit: {str(e)}")
                logger.warning("Falling back to local rate limiting")

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
        Check rate limit using Redis.

        Args:
            key (str): Unique key to identify the request source.
            rate (int): Maximum number of requests allowed per minute.
            burst (int): Burst capacity for requests.
            window (int): Time window in seconds for rate limiting.
            cost (int): Cost of the request in terms of tokens.

        Returns:
            Tuple[bool, Dict[str, Any]]: A tuple of (is_limited, limit_info)
        """
        now = time.time()

        tokens_key = f"{key}:tokens"
        last_refill_key = f"{key}:last_refill"

        async with self.redis.pipeline() as pipe:
            await pipe.get(tokens_key)
            await pipe.get(last_refill_key)
            tokens_str, last_refill_str = await pipe.execute()

        tokens = float(tokens_str) if tokens_str else burst
        last_refill = float(last_refill_str) if last_refill_str else now

        time_elapsed = now - last_refill
        new_tokens = time_elapsed * (rate / window)
        tokens = min(tokens + new_tokens, burst)

        is_limited = tokens < cost

        if not is_limited:
            tokens -= cost

        async with self.redis.pipeline() as pipe:
            await pipe.set(tokens_key, tokens)
            await pipe.set(last_refill_key, now)
            await pipe.expire(tokens_key, window * 2)  # Set expiry to 2x window
            await pipe.expire(last_refill_key, window * 2)  # Set expiry to 2x window
            await pipe.execute()

        if tokens < cost:
            time_until_refill = (cost - tokens) * (window / rate)
            reset_time = now + time_until_refill
        else:
            time_until_full = (burst - tokens) * (window / rate)
            reset_time = now + time_until_full

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
        Check rate limit using local in-memory storage (fallback).

        Args:
            key (str): Unique key to identify the request source.
            rate (int): Maximum number of requests allowed per minute.
            burst (int): Burst capacity for requests.
            window (int): Time window in seconds for rate limiting.
            cost (int): Cost of the request in terms of tokens.

        Returns:
            Tuple[bool, Dict[str, Any]]: A tuple of (is_limited, limit_info)
        """
        async with self.lock:
            now = time.time()

            if key not in self.local_tokens:
                self.local_limits[key] = {"rate": rate, "burst": burst, "window": window}
                self.local_tokens[key] = burst
                self.local_last_refill[key] = now

            last_refill = self.local_last_refill[key]
            time_elapsed = now - last_refill
            new_tokens = time_elapsed * (rate / window)
            self.local_tokens[key] = min(self.local_tokens[key] + new_tokens, burst)
            self.local_last_refill[key] = now

            is_limited = self.local_tokens[key] < cost

            if not is_limited:
                self.local_tokens[key] -= cost

            if self.local_tokens[key] < cost:
                time_until_refill = (cost - self.local_tokens[key]) * (window / rate)
                reset_time = now + time_until_refill
            else:
                time_until_full = (burst - self.local_tokens[key]) * (window / rate)
                reset_time = now + time_until_full

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
        Get the current rate limit information for a key.

        Args:
            key (str): The rate limit key.

        Returns:
            Dict[str, Any]: Rate limit information.
        """
        namespaced_key = f"{self.namespace}{key}"
        
        if self.redis:
            try:
                tokens_key = f"{namespaced_key}:tokens"
                last_refill_key = f"{namespaced_key}:last_refill"
                
                async with self.redis.pipeline() as pipe:
                    await pipe.get(tokens_key)
                    await pipe.get(last_refill_key)
                    tokens_str, last_refill_str = await pipe.execute()
                
                if tokens_str is None:
                    return {
                        "limit": self.default_rate,
                        "remaining": self.default_rate,
                        "reset": int(time.time() + self.default_window),
                        "window": self.default_window,
                        "burst": self.default_burst
                    }
                
                tokens = float(tokens_str) if tokens_str else self.default_burst
                
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
        
        # Local fallback
        async with self.lock:
            if namespaced_key not in self.local_tokens:
                return {
                    "limit": self.default_rate,
                    "remaining": self.default_rate,
                    "reset": int(time.time() + self.default_window),
                    "window": self.default_window,
                    "burst": self.default_burst
                }
            
            now = time.time()
            time_until_full = (self.default_burst - self.local_tokens[namespaced_key]) * (
                self.default_window / self.default_rate)
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
        Reset the rate limit for a specific key.

        Args:
            key (str): The rate limit key to reset.

        Returns:
            bool: True if successfully reset.
        """
        namespaced_key = f"{self.namespace}{key}"
        success = False
        
        if self.redis:
            try:
                tokens_key = f"{namespaced_key}:tokens"
                last_refill_key = f"{namespaced_key}:last_refill"
                
                async with self.redis.pipeline() as pipe:
                    await pipe.delete(tokens_key)
                    await pipe.delete(last_refill_key)
                    await pipe.execute()
                success = True
            except Exception as e:
                logger.error(f"Error resetting Redis rate limit: {str(e)}")
        
        # Also reset local data
        async with self.lock:
            if namespaced_key in self.local_tokens:
                del self.local_tokens[namespaced_key]
                del self.local_last_refill[namespaced_key]
                if namespaced_key in self.local_limits:
                    del self.local_limits[namespaced_key]
                success = True
                
        return success
    
    async def update_limits(
        self,
        key: str,
        rate: Optional[int] = None,
        burst: Optional[int] = None,
        window: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Update rate limit parameters for a key.

        Args:
            key (str): The rate limit key to update.
            rate (int, optional): New request rate per minute.
            burst (int, optional): New burst capacity.
            window (int, optional): New window in seconds.

        Returns:
            Dict[str, Any]: Updated limit information.
        """
        namespaced_key = f"{self.namespace}{key}"
        
        current_limits = await self.get_limits(key)
        
        new_rate = rate if rate is not None else current_limits["limit"]
        new_burst = burst if burst is not None else current_limits["burst"]
        new_window = window if window is not None else current_limits["window"]
        
        if self.redis:
            try:
                limits_key = f"{namespaced_key}:limits"
                limits = {
                    "rate": new_rate,
                    "burst": new_burst,
                    "window": new_window
                }
                
                await self.redis.set(limits_key, json.dumps(limits))
                await self.redis.expire(limits_key, new_window * 2)
                
                tokens_key = f"{namespaced_key}:tokens"
                last_refill_key = f"{namespaced_key}:last_refill"
                
                async with self.redis.pipeline() as pipe:
                    await pipe.set(tokens_key, new_burst)
                    await pipe.set(last_refill_key, time.time())
                    await pipe.expire(tokens_key, new_window * 2)
                    await pipe.expire(last_refill_key, new_window * 2)
                    await pipe.execute()
            except Exception as e:
                logger.error(f"Error updating Redis rate limit: {str(e)}")
        
        # Update local data
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
    
    async def ping_redis(self) -> bool:
        """
        Check if Redis connection is active.
        
        Returns:
            bool: True if connected, False otherwise
        """
        if not self.redis:
            return False
        
        try:
            return await self.redis.ping()
        except Exception:
            return False

class RateLimiterManager:
    """
    Unified rate limiter manager for the application.
    
    This class provides a singleton manager that creates and manages rate limiters
    for different parts of the application.
    
    Attributes:
        distributed_limiter: Redis-based distributed rate limiter
        limiters: Dictionary of named AsyncTokenBucketRateLimiter instances
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Create a singleton instance of rate limiter manager."""
        if cls._instance is None:
            cls._instance = super(RateLimiterManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_rate: int = 60,
        default_burst: int = 10,
        default_window: int = 60,
        namespace: str = "asf:medical:rate_limit:"
    ):
        """Initialize the rate limiter manager."""
        if self._initialized:
            return
            
        self.default_rate = default_rate
        self.default_burst = default_burst 
        self.default_window = default_window
        
        # Create the distributed rate limiter
        self.distributed_limiter = DistributedRateLimiter(
            redis_url=redis_url,
            default_rate=default_rate,
            default_burst=default_burst,
            default_window=default_window,
            namespace=namespace
        )
        
        # Dictionary of rate limiters
        self.limiters = {}
        
        self._initialized = True
        logger.info("Rate limiter manager initialized")
    
    def get_limiter(self, name: str, rps: float = None) -> AsyncTokenBucketRateLimiter:
        """
        Get or create an async token bucket limiter by name.
        
        Args:
            name: Limiter name
            rps: Requests per second (default: default_rate/60)
            
        Returns:
            AsyncTokenBucketRateLimiter instance
        """
        if name not in self.limiters:
            rps = rps or (self.default_rate / 60.0)
            self.limiters[name] = AsyncTokenBucketRateLimiter(
                requests_per_second=rps,
                burst_size=self.default_burst
            )
        return self.limiters[name]
    
    async def is_rate_limited(
        self,
        key: str,
        rate: Optional[int] = None,
        burst: Optional[int] = None,
        window: Optional[int] = None,
        cost: int = 1,
        distributed: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a request would exceed the rate limit.
        
        Args:
            key: Rate limit key
            rate: Requests per minute
            burst: Burst capacity
            window: Time window in seconds
            cost: Cost of the request
            distributed: Whether to use distributed limiting
            
        Returns:
            Tuple of (is_limited, limit_info)
        """
        if distributed:
            return await self.distributed_limiter.is_rate_limited(key, rate, burst, window, cost)
        else:
            # Use a named limiter for local limiting
            limiter = self.get_limiter(key, (rate or self.default_rate) / 60.0)
            tokens_needed = cost
            
            # Check if rate limited
            current_tokens = await limiter.get_token_count()
            is_limited = current_tokens < tokens_needed
            
            # Calculate reset time and other limit info
            now = time.time()
            if is_limited:
                time_until_refill = await limiter.wait_time_for_tokens(tokens_needed)
                reset_time = now + time_until_refill
            else:
                burst_size = burst or self.default_burst
                time_until_full = (burst_size - current_tokens) * 60.0 / (rate or self.default_rate)
                reset_time = now + time_until_full
                
                # Consume tokens if not limited
                await limiter.acquire(tokens_needed)
            
            limit_info = {
                "limit": rate or self.default_rate,
                "remaining": int(current_tokens / cost),
                "reset": int(reset_time),
                "window": window or self.default_window,
                "burst": burst or self.default_burst
            }
            
            return is_limited, limit_info
    
    async def reset_all_limiters(self) -> None:
        """Reset all rate limiters."""
        self.limiters.clear()

# Singleton instance of the rate limiter manager
_rate_limiter_manager = None

def get_rate_limiter_manager(**kwargs) -> RateLimiterManager:
    """
    Get the singleton rate limiter manager.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        RateLimiterManager instance
    """
    global _rate_limiter_manager
    if _rate_limiter_manager is None:
        _rate_limiter_manager = RateLimiterManager(**kwargs)
    return _rate_limiter_manager

def rate_limit(
    key_func: Callable[..., str],
    rate: Optional[int] = None,
    burst: Optional[int] = None,
    window: Optional[int] = None,
    distributed: bool = False,
    raise_on_limit: bool = True
):
    """
    Decorator to apply rate limiting to a function.
    
    Args:
        key_func: Function to generate the rate limit key
        rate: Requests per minute limit
        burst: Burst capacity
        window: Time window in seconds
        distributed: Whether to use distributed limiting
        raise_on_limit: Whether to raise an exception when limited
        
    Returns:
        Decorated function
    """
    manager = get_rate_limiter_manager()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = key_func(*args, **kwargs)
            
            is_limited, limit_info = await manager.is_rate_limited(
                key=key,
                rate=rate,
                burst=burst,
                window=window,
                distributed=distributed
            )
            
            if is_limited and raise_on_limit:
                raise RateLimitExceededError(
                    f"Rate limit exceeded for key '{key}'",
                    key=key,
                    limit=limit_info["limit"],
                    reset_time=limit_info["reset"]
                )
                
            if is_limited:
                logger.warning(f"Rate limit exceeded for {key} (limit: {limit_info['limit']}/min)")
                return None
                
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def global_rate_limit(
    key_func: Callable[..., str],
    rate: Optional[int] = None,
    burst: Optional[int] = None,
    window: Optional[int] = None
):
    """
    Decorator to apply distributed rate limiting across instances.
    
    This is an alias for rate_limit with distributed=True.
    
    Args:
        key_func: Function to generate the rate limit key
        rate: Requests per minute limit
        burst: Burst capacity
        window: Time window in seconds
        
    Returns:
        Decorated function
    """
    return rate_limit(
        key_func=key_func,
        rate=rate,
        burst=burst,
        window=window,
        distributed=True,
        raise_on_limit=True
    )

# Initialize singleton manager
rate_limiter_manager = get_rate_limiter_manager()
