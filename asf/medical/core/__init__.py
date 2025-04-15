"""
Core module for the Medical Research Synthesizer.

This module provides core functionality for the Medical Research Synthesizer,
including caching, progress tracking, and exception handling.
"""

# Import cache functionality
from asf.medical.core.cache_init import (
    CacheManager,
    get_cache_manager,
    cached,
    clear_cache,
    EnhancedCacheManager,
    get_enhanced_cache_manager,
    enhanced_cached,
    DataSensitivity,
    enhanced_cache_manager
)

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
