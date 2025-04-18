"""
Rate limiter for MCP transport.

This module provides rate limiting for MCP transport,
with support for token bucket, sliding window, and adaptive rate limiting.
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategy."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    ADAPTIVE = "adaptive"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    requests_per_minute: int = 60
    burst_size: int = 10
    window_size_seconds: int = 60
    adaptive_factor: float = 0.5
    adaptive_min_rate: float = 0.1
    adaptive_max_rate: float = 2.0


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.
    
    This class implements a token bucket rate limiter,
    which allows bursts of requests up to a certain size.
    """
    
    def __init__(
        self,
        requests_per_minute: int,
        burst_size: int
    ):
        """
        Initialize the token bucket rate limiter.
        
        Args:
            requests_per_minute: Requests per minute
            burst_size: Burst size
        """
        self.requests_per_second = requests_per_minute / 60.0
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_refill_time = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> Tuple[bool, float]:
        """
        Acquire a token.
        
        Returns:
            Tuple of (success, wait_time)
        """
        async with self._lock:
            # Refill tokens
            now = time.time()
            elapsed = now - self.last_refill_time
            self.last_refill_time = now
            
            # Calculate new tokens
            new_tokens = elapsed * self.requests_per_second
            self.tokens = min(self.burst_size, self.tokens + new_tokens)
            
            # Check if we have a token
            if self.tokens >= 1:
                self.tokens -= 1
                return True, 0.0
            
            # Calculate wait time
            wait_time = (1.0 - self.tokens) / self.requests_per_second
            return False, wait_time


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.
    
    This class implements a sliding window rate limiter,
    which tracks requests over a sliding time window.
    """
    
    def __init__(
        self,
        requests_per_minute: int,
        window_size_seconds: int
    ):
        """
        Initialize the sliding window rate limiter.
        
        Args:
            requests_per_minute: Requests per minute
            window_size_seconds: Window size in seconds
        """
        self.requests_per_window = requests_per_minute * window_size_seconds / 60.0
        self.window_size_seconds = window_size_seconds
        self.request_timestamps = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> Tuple[bool, float]:
        """
        Acquire a token.
        
        Returns:
            Tuple of (success, wait_time)
        """
        async with self._lock:
            # Remove old timestamps
            now = time.time()
            window_start = now - self.window_size_seconds
            self.request_timestamps = [ts for ts in self.request_timestamps if ts >= window_start]
            
            # Check if we can make a request
            if len(self.request_timestamps) < self.requests_per_window:
                self.request_timestamps.append(now)
                return True, 0.0
            
            # Calculate wait time
            if self.request_timestamps:
                oldest = self.request_timestamps[0]
                wait_time = oldest + self.window_size_seconds - now
                return False, max(0.0, wait_time)
            
            return False, 0.0


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter.
    
    This class implements an adaptive rate limiter,
    which adjusts the rate based on success and failure.
    """
    
    def __init__(
        self,
        requests_per_minute: int,
        adaptive_factor: float = 0.5,
        adaptive_min_rate: float = 0.1,
        adaptive_max_rate: float = 2.0
    ):
        """
        Initialize the adaptive rate limiter.
        
        Args:
            requests_per_minute: Requests per minute
            adaptive_factor: Adaptive factor
            adaptive_min_rate: Minimum rate factor
            adaptive_max_rate: Maximum rate factor
        """
        self.base_rate = requests_per_minute / 60.0
        self.current_rate = self.base_rate
        self.adaptive_factor = adaptive_factor
        self.adaptive_min_rate = adaptive_min_rate
        self.adaptive_max_rate = adaptive_max_rate
        self.success_count = 0
        self.failure_count = 0
        self.last_request_time = time.time()
        self.tokens = 1.0
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> Tuple[bool, float]:
        """
        Acquire a token.
        
        Returns:
            Tuple of (success, wait_time)
        """
        async with self._lock:
            # Refill tokens
            now = time.time()
            elapsed = now - self.last_request_time
            self.last_request_time = now
            
            # Calculate new tokens
            new_tokens = elapsed * self.current_rate
            self.tokens = min(1.0, self.tokens + new_tokens)
            
            # Check if we have a token
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True, 0.0
            
            # Calculate wait time
            wait_time = (1.0 - self.tokens) / self.current_rate
            return False, wait_time
    
    async def record_success(self) -> None:
        """
        Record a successful request.
        """
        async with self._lock:
            self.success_count += 1
            self._adjust_rate()
    
    async def record_failure(self) -> None:
        """
        Record a failed request.
        """
        async with self._lock:
            self.failure_count += 1
            self._adjust_rate()
    
    def _adjust_rate(self) -> None:
        """
        Adjust the rate based on success and failure.
        """
        total = self.success_count + self.failure_count
        
        if total < 10:
            # Not enough data
            return
        
        # Calculate success rate
        success_rate = self.success_count / total
        
        # Adjust rate
        if success_rate > 0.95:
            # Increase rate
            factor = 1.0 + (self.adaptive_factor * (success_rate - 0.95) * 20.0)
        elif success_rate < 0.9:
            # Decrease rate
            factor = 1.0 - (self.adaptive_factor * (0.9 - success_rate) * 10.0)
        else:
            # Keep rate
            factor = 1.0
        
        # Apply factor
        factor = max(self.adaptive_min_rate, min(self.adaptive_max_rate, factor))
        self.current_rate = self.base_rate * factor
        
        # Reset counters periodically
        if total > 100:
            self.success_count = int(self.success_count * 0.5)
            self.failure_count = int(self.failure_count * 0.5)


class RateLimiter:
    """
    Rate limiter for MCP transport.
    
    This class provides rate limiting for MCP transport,
    with support for token bucket, sliding window, and adaptive rate limiting.
    """
    
    def __init__(
        self,
        provider_id: str,
        config: RateLimitConfig
    ):
        """
        Initialize the rate limiter.
        
        Args:
            provider_id: Provider ID
            config: Rate limit configuration
        """
        self.provider_id = provider_id
        self.config = config
        
        # Create rate limiter based on strategy
        if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            self.limiter = TokenBucketRateLimiter(
                requests_per_minute=config.requests_per_minute,
                burst_size=config.burst_size
            )
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            self.limiter = SlidingWindowRateLimiter(
                requests_per_minute=config.requests_per_minute,
                window_size_seconds=config.window_size_seconds
            )
        elif config.strategy == RateLimitStrategy.ADAPTIVE:
            self.limiter = AdaptiveRateLimiter(
                requests_per_minute=config.requests_per_minute,
                adaptive_factor=config.adaptive_factor,
                adaptive_min_rate=config.adaptive_min_rate,
                adaptive_max_rate=config.adaptive_max_rate
            )
        else:
            raise ValueError(f"Unknown rate limit strategy: {config.strategy}")
    
    async def acquire(self) -> Tuple[bool, float]:
        """
        Acquire a token.
        
        Returns:
            Tuple of (success, wait_time)
        """
        return await self.limiter.acquire()
    
    async def wait(self) -> None:
        """
        Wait for a token.
        """
        while True:
            success, wait_time = await self.acquire()
            
            if success:
                return
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
    
    async def record_success(self) -> None:
        """
        Record a successful request.
        """
        if hasattr(self.limiter, "record_success"):
            await self.limiter.record_success()
    
    async def record_failure(self) -> None:
        """
        Record a failed request.
        """
        if hasattr(self.limiter, "record_failure"):
            await self.limiter.record_failure()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.
        
        Returns:
            Rate limiter statistics
        """
        stats = {
            "provider_id": self.provider_id,
            "strategy": self.config.strategy,
            "requests_per_minute": self.config.requests_per_minute
        }
        
        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            stats.update({
                "burst_size": self.config.burst_size,
                "current_tokens": self.limiter.tokens
            })
        elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            stats.update({
                "window_size_seconds": self.config.window_size_seconds,
                "current_requests": len(self.limiter.request_timestamps)
            })
        elif self.config.strategy == RateLimitStrategy.ADAPTIVE:
            stats.update({
                "adaptive_factor": self.config.adaptive_factor,
                "adaptive_min_rate": self.config.adaptive_min_rate,
                "adaptive_max_rate": self.config.adaptive_max_rate,
                "current_rate": self.limiter.current_rate,
                "base_rate": self.limiter.base_rate,
                "success_count": self.limiter.success_count,
                "failure_count": self.limiter.failure_count
            })
        
        return stats
