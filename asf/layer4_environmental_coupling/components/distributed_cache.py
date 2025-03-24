# === FILE: asf/environmental_coupling/components/distributed_cache.py ===
import asyncio
import time
import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Set

class DistributedCouplingCache:
    """
    Provides distributed caching for coupling data.
    Optimizes performance through multi-level caching with consistency guarantees.
    """
    
    def __init__(self, cache_config: Dict = None):
        self.config = cache_config or {
            'max_local_size': 10000,
            'max_shared_size': 100000,
            'default_ttl': 300,  # 5 minutes
            'consistency_check_interval': 60  # 1 minute
        }
        
        # Local cache (per-instance)
        self.local_cache = {}
        self.local_expiry = {}
        
        # Shared cache (simulated)
        # In a real implementation, this would use Redis, Memcached, etc.
        self.shared_cache = {}
        self.shared_expiry = {}
        
        # Tracking for eviction
        self.access_history = {}
        self.update_timestamps = {}
        
        # Lock for cache operations
        self.lock = asyncio.Lock()
        
        # Background task for maintenance
        self.maintenance_task = None
        self.running = False
        
        self.logger = logging.getLogger("ASF.Layer4.DistributedCouplingCache")
        
    async def initialize(self):
        """Initialize the cache service."""
        self.running = True
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        self.logger.info(f"Initialized with max local size {self.config['max_local_size']}")
        return True
    
    async def get(self, key: str, use_shared: bool = True) -> Optional[Any]:
        """
        Get a value from the cache.
        Checks local cache first, then shared if enabled.
        """
        current_time = time.time()
        
        # Check local cache first
        if key in self.local_cache:
            expiry = self.local_expiry.get(key, 0)
            
            if expiry > current_time:
                # Update access record
                self.access_history[key] = current_time
                return self.local_cache[key]
            else:
                # Expired, remove from local cache
                await self._remove_from_local(key)
        
        # If not in local cache and shared cache is enabled, check there
        if use_shared and key in self.shared_cache:
            expiry = self.shared_expiry.get(key, 0)
            
            if expiry > current_time:
                # Found in shared cache, update local cache
                value = self.shared_cache[key]
                await self._add_to_local(key, value, expiry - current_time)
                return value
        
        # Not found or expired
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                use_shared: bool = True) -> bool:
        """
        Set a value in the cache with optional time-to-live.
        Updates both local and shared caches if enabled.
        """
        if ttl is None:
            ttl = self.config['default_ttl']
            
        current_time = time.time()
        expiry = current_time + ttl
        
        # Update local cache
        result = await self._add_to_local(key, value, ttl)
        
        # Update shared cache if enabled
        if use_shared:
            self.shared_cache[key] = value
            self.shared_expiry[key] = expiry
            self.update_timestamps[key] = current_time
            
            # Check shared cache size limits
            if len(self.shared_cache) > self.config['max_shared_size']:
                await self._evict_from_shared(10)  # Evict 10 items
        
        return result
    
    async def invalidate(self, key: str, use_shared: bool = True) -> bool:
        """
        Invalidate a cache entry.
        Removes from both local and shared caches if enabled.
        """
        result = await self._remove_from_local(key)
        
        # Also remove from shared cache if enabled
        if use_shared and key in self.shared_cache:
            del self.shared_cache[key]
            if key in self.shared_expiry:
                del self.shared_expiry[key]
            if key in self.update_timestamps:
                del self.update_timestamps[key]
                
            result = True
        
        return result
    
    async def invalidate_pattern(self, pattern: str, use_shared: bool = True) -> int:
        """
        Invalidate all cache entries matching a pattern.
        Returns the number of entries invalidated.
        """
        count = 0
        
        # Find matching keys
        local_matches = [k for k in self.local_cache if pattern in k]
        
        # Remove from local cache
        for key in local_matches:
            if await self._remove_from_local(key):
                count += 1
        
        # Also remove from shared cache if enabled
        if use_shared:
            shared_matches = [k for k in self.shared_cache if pattern in k]
            
            for key in shared_matches:
                if key in self.shared_cache:
                    del self.shared_cache[key]
                    if key in self.shared_expiry:
                        del self.shared_expiry[key]
                    if key in self.update_timestamps:
                        del self.update_timestamps[key]
                    count += 1
        
        return count
    
    async def _add_to_local(self, key: str, value: Any, ttl: int) -> bool:
        """Add a value to the local cache with expiration."""
        async with self.lock:
            # Check cache size limit
            if key not in self.local_cache and len(self.local_cache) >= self.config['max_local_size']:
                await self._evict_from_local(1)  # Make room
            
            current_time = time.time()
            self.local_cache[key] = value
            self.local_expiry[key] = current_time + ttl
            self.access_history[key] = current_time
            
            return True
    
    async def _remove_from_local(self, key: str) -> bool:
        """Remove a value from the local cache."""
        async with self.lock:
            if key in self.local_cache:
                del self.local_cache[key]
                if key in self.local_expiry:
                    del self.local_expiry[key]
                if key in self.access_history:
                    del self.access_history[key]
                return True
            return False
    
    async def _evict_from_local(self, count: int) -> int:
        """
        Evict entries from local cache using LRU policy.
        Returns the number of entries evicted.
        """
        if not self.access_history:
            return 0
            
        # Sort by last access time (oldest first)
        sorted_keys = sorted(
            self.access_history.keys(),
            key=lambda k: self.access_history[k]
        )
        
        # Take the oldest entries up to count
        to_evict = sorted_keys[:count]
        
        # Remove them
        evicted = 0
        for key in to_evict:
            if await self._remove_from_local(key):
                evicted += 1
                
        return evicted
    
    async def _evict_from_shared(self, count: int) -> int:
        """
        Evict entries from shared cache.
        In a real implementation, this might coordinate with other instances.
        """
        if not self.update_timestamps:
            return 0
            
        # Sort by last update time (oldest first)
        sorted_keys = sorted(
            self.update_timestamps.keys(),
            key=lambda k: self.update_timestamps[k]
        )
        
        # Take the oldest entries up to count
        to_evict = sorted_keys[:count]
        
        # Remove them
        evicted = 0
        for key in to_evict:
            if key in self.shared_cache:
                del self.shared_cache[key]
                if key in self.shared_expiry:
                    del self.shared_expiry[key]
                if key in self.update_timestamps:
                    del self.update_timestamps[key]
                evicted += 1
                
        return evicted
    
    async def _maintenance_loop(self):
        """Background task for cache maintenance."""
        try:
            while self.running:
                try:
                    # Clean expired entries from local cache
                    await self._clean_expired_local()
                    
                    # Clean expired entries from shared cache
                    await self._clean_expired_shared()
                    
                    # Check consistency with shared cache
                    await self._check_consistency()
                    
                    # Wait for next maintenance cycle
                    await asyncio.sleep(self.config['consistency_check_interval'])
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in cache maintenance: {str(e)}")
                    await asyncio.sleep(5)  # Wait before retry
                    
        finally:
            self.running = False
    
    async def _clean_expired_local(self) -> int:
        """Clean expired entries from local cache."""
        current_time = time.time()
        to_remove = []
        
        async with self.lock:
            for key, expiry in self.local_expiry.items():
                if expiry <= current_time:
                    to_remove.append(key)
                    
            # Remove expired entries
            for key in to_remove:
                await self._remove_from_local(key)
                
            return len(to_remove)
    
    async def _clean_expired_shared(self) -> int:
        """Clean expired entries from shared cache."""
        current_time = time.time()
        to_remove = []
        
        for key, expiry in self.shared_expiry.items():
            if expiry <= current_time:
                to_remove.append(key)
                
        # Remove expired entries
        for key in to_remove:
            if key in self.shared_cache:
                del self.shared_cache[key]
            if key in self.shared_expiry:
                del self.shared_expiry[key]
            if key in self.update_timestamps:
                del self.update_timestamps[key]
                
        return len(to_remove)
    
    async def _check_consistency(self) -> Dict:
        """
        Check consistency between local and shared caches.
        Synchronizes when needed.
        """
        # In a real implementation, this would check with a distributed
        # cache system like Redis for newer versions of locally cached items
        
        # For this simulation, we'll just check a few random keys
        if not self.local_cache or not self.shared_cache:
            return {'checked': 0, 'updated': 0}
            
        # Select random keys from local cache
        sample_size = min(10, len(self.local_cache))
        sample_keys = random.sample(list(self.local_cache.keys()), sample_size)
        
        updated = 0
        
        for key in sample_keys:
            # Check if key exists in shared cache and is newer
            if (key in self.shared_cache and key in self.update_timestamps and
                key in self.local_expiry and 
                self.update_timestamps[key] > self.access_history.get(key, 0)):
                
                # Shared version is newer, update local
                self.local_cache[key] = self.shared_cache[key]
                self.local_expiry[key] = self.shared_expiry[key]
                self.access_history[key] = time.time()
                updated += 1
        
        return {'checked': sample_size, 'updated': updated}
    
    async def stop(self):
        """Stop the cache service."""
        self.running = False
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass
                
        self.logger.info("Cache service stopped")
        return True
    
    async def get_metrics(self) -> Dict:
        """Get cache metrics."""
        return {
            'local_entries': len(self.local_cache),
            'shared_entries': len(self.shared_cache),
            'local_hit_ratio': self.local_hit_ratio if hasattr(self, 'local_hit_ratio') else 0,
            'memory_usage': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of the cache in bytes."""
        # This is a very rough estimation
        # In a real implementation, you would use sys.getsizeof or similar
        
        # Assume average key size of 50 bytes and value size of 500 bytes
        local_estimate = len(self.local_cache) * 550
        
        return local_estimate
