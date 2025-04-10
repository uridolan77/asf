"""
Rate limiter for the Medical Research Synthesizer.

This module provides a rate limiter for API clients.
"""

import asyncio
import time
from typing import Dict, Optional, Any

class AsyncRateLimiter:
    """
    Asynchronous rate limiter for API clients.
    
    This class provides a rate limiter that can be used to limit the rate of API requests.
    It uses a token bucket algorithm to limit the rate of requests.
    """
    
    def __init__(
        self,
        requests_per_second: float,
        burst_size: Optional[int] = None
    ):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_second: Maximum number of requests per second
            burst_size: Maximum number of requests that can be made in a burst (default: 2 * requests_per_second)
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size or int(2 * requests_per_second)
        self.tokens = self.burst_size
        self.last_refill_time = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens from the rate limiter.
        
        This method blocks until the requested number of tokens are available.
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
        """
        if tokens > self.burst_size:
            raise ValueError(f"Cannot acquire more tokens than burst size ({self.burst_size})")
        
        async with self.lock:
            # Refill tokens based on time elapsed since last refill
            await self._refill()
            
            # Wait until enough tokens are available
            while self.tokens < tokens:
                # Calculate how long to wait for the next token
                wait_time = (tokens - self.tokens) / self.requests_per_second
                
                # Release the lock while waiting
                self.lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    # Reacquire the lock
                    await self.lock.acquire()
                
                # Refill tokens again after waiting
                await self._refill()
            
            # Consume tokens
            self.tokens -= tokens
    
    async def _refill(self) -> None:
        """
        Refill tokens based on time elapsed since last refill.
        
        This method should be called with the lock held.
        """
        now = time.time()
        time_elapsed = now - self.last_refill_time
        
        # Calculate how many tokens to add
        new_tokens = time_elapsed * self.requests_per_second
        
        if new_tokens > 0:
            self.tokens = min(self.tokens + new_tokens, self.burst_size)
            self.last_refill_time = now
