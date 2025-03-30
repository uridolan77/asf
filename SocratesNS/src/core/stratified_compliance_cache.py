import time
import hashlib
from typing import Dict, Any, Tuple, Optional, List, Union
from functools import lru_cache
from dataclasses import dataclass, field

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    last_cleanup: float = field(default_factory=time.time)

class StratifiedComplianceCache:
    """
    Stratified cache for compliance results with different time-to-live periods 
    based on content type and compliance frameworks.
    """
    
    def __init__(self, 
                max_size: int = 10000, 
                cleanup_interval: int = 3600,
                ttl_config: Dict[str, int] = None):
        """
        Initialize the cache
        
        Args:
            max_size: Maximum number of entries in the cache
            cleanup_interval: Seconds between cleanup runs
            ttl_config: Configuration for TTL (time-to-live) in seconds by content type
        """
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self.ttl_config = ttl_config or {
            # Default TTL values by content and framework type
            "default": 3600,  # 1 hour default
            "text": 7200,     # 2 hours for text content
            "prompt": 14400,  # 4 hours for prompts
            "GDPR": 43200,    # 12 hours for GDPR-only content
            "HIPAA": 21600    # 6 hours for HIPAA-only content
        }
        
        # Initialize cache storage with timestamps
        self.cache: Dict[str, Tuple[Any, float, str]] = {}  # key -> (value, timestamp, category)
        self.stats = CacheStats()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if it exists and is not expired
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self.cache:
            self.stats.misses += 1
            return None
            
        value, timestamp, category = self.cache[key]
        current_time = time.time()
        
        # Get TTL for this category
        ttl = self.ttl_config.get(category, self.ttl_config["default"])
        
        # Check if entry is expired
        if current_time - timestamp > ttl:
            # Remove expired entry
            del self.cache[key]
            self.stats.evictions += 1
            self.stats.misses += 1
            return None
            
        self.stats.hits += 1
        return value
    
    def put(self, key: str, value: Any, category: str = "default") -> None:
        """
        Add or update value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            category: Category for TTL determination
        """
        # Check if cleanup is needed
        self._maybe_cleanup()
        
        # If at max capacity, remove oldest entry
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
            del self.cache[oldest_key]
            self.stats.evictions += 1
        
        # Add or update entry
        self.cache[key] = (value, time.time(), category)
    
    def invalidate(self, pattern: str = None) -> int:
        """
        Invalidate cache entries matching pattern
        
        Args:
            pattern: Pattern to match keys against (None to invalidate all)
            
        Returns:
            Number of invalidated entries
        """
        if pattern is None:
            count = len(self.cache)
            self.cache.clear()
            return count
        
        # Find keys matching pattern
        keys_to_remove = [k for k in self.cache if pattern in k]
        for key in keys_to_remove:
            del self.cache[key]
            
        return len(keys_to_remove)
    
    def _maybe_cleanup(self) -> None:
        """Perform cleanup if the interval has elapsed"""
        current_time = time.time()
        if current_time - self.stats.last_cleanup > self.cleanup_interval:
            self._cleanup()
            self.stats.last_cleanup = current_time
    
    def _cleanup(self) -> None:
        """Remove expired entries from cache"""
        current_time = time.time()
        keys_to_remove = []
        
        for key, (_, timestamp, category) in self.cache.items():
            ttl = self.ttl_config.get(category, self.ttl_config["default"])
            if current_time - timestamp > ttl:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.cache[key]
            self.stats.evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_ratio": self.stats.hits / (self.stats.hits + self.stats.misses) if (self.stats.hits + self.stats.misses) > 0 else 0,
            "evictions": self.stats.evictions,
            "categories": self._count_by_category()
        }
    
    def _count_by_category(self) -> Dict[str, int]:
        """Count cache entries by category"""
        counts = {}
        for _, (_, _, category) in self.cache.items():
            if category not in counts:
                counts[category] = 0
            counts[category] += 1
        return counts
    
    def _generate_cache_key(self, content: Union[str, Dict], content_type: str, frameworks: List[str]) -> str:
        """Generate a unique cache key for compliance content"""
        # Sort frameworks for consistent keys
        sorted_frameworks = sorted(frameworks)
        
        # Convert content to string for hashing
        if isinstance(content, dict):
            content_str = str(sorted(content.items()))
        else:
            content_str = str(content)
            
        # Create a hash of the content
        content_hash = hashlib.md5(content_str.encode()).hexdigest()
        
        # Combine components into a key
        return f"{content_type}:{content_hash}:{'-'.join(sorted_frameworks)}"

# Function wrapper for easy caching of compliance checks
def cached_compliance_check(cache: StratifiedComplianceCache):
    """Decorator for caching compliance check results"""
    def decorator(func):
        def wrapper(content, content_type="text", frameworks=None, **kwargs):
            frameworks = frameworks or ["default"]
            cache_key = cache._generate_cache_key(content, content_type, frameworks)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
                
            # Calculate result
            result = func(content, content_type, frameworks, **kwargs)
            
            # Determine category for TTL
            if len(frameworks) == 1:
                category = frameworks[0]
            else:
                category = content_type
                
            # Cache result
            cache.put(cache_key, result, category)
            
            return result
        return wrapper
    return decorator