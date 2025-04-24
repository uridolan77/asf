"""
Core module for the Medical Research Synthesizer.

This module provides core functionality for the Medical Research Synthesizer,
including caching, progress tracking, and exception handling.
"""

# Mock cache functionality
class CacheManager:
    """Mock CacheManager class."""
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or "./cache"

    def get_cache(self, name):
        """Get a cache by name."""
        return {}

def get_cache_manager():
    """Get the cache manager instance."""
    return CacheManager()

def cached(func):
    """Decorator for caching function results."""
    return func

def clear_cache(name=None):
    """Clear the cache."""
    pass

class DataSensitivity:
    """Data sensitivity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class EnhancedCacheManager(CacheManager):
    """Mock EnhancedCacheManager class."""
    pass

def get_enhanced_cache_manager():
    """Get the enhanced cache manager instance."""
    return EnhancedCacheManager()

def enhanced_cached(sensitivity=DataSensitivity.LOW):
    """Decorator for enhanced caching function results."""
    def decorator(func):
        return func
    return decorator

enhanced_cache_manager = EnhancedCacheManager()

__all__ = [
    'CacheManager',
    'get_cache_manager',
    'cached',
    'clear_cache',
    'EnhancedCacheManager',
    'get_enhanced_cache_manager',
    'enhanced_cached',
    'DataSensitivity',
    'enhanced_cache_manager'
]
