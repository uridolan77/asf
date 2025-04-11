"""
Rate limiter for the Medical Research Synthesizer.

This module provides a rate limiter for API clients.
"""

import asyncio
import time

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
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size or int(2 * requests_per_second)
        self.tokens = self.burst_size
        self.last_refill_time = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> None:
        if tokens > self.burst_size:
            raise ValueError(f"Cannot acquire more tokens than burst size ({self.burst_size})")
        
        async with self.lock:
            await self._refill()
            
            while self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.requests_per_second
                
                self.lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self.lock.acquire()
                
                await self._refill()
            
            self.tokens -= tokens
    
    async def _refill(self) -> None:
        now = time.time()
        time_elapsed = now - self.last_refill_time
        
        new_tokens = time_elapsed * self.requests_per_second
        
        if new_tokens > 0:
            self.tokens = min(self.tokens + new_tokens, self.burst_size)
            self.last_refill_time = now
