"""
Cache initialization module for the Medical Research Synthesizer.

This module initializes the cache system and provides access to both the base
and enhanced cache functionality.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import base cache functionality
from asf.medical.core.cache import (
    CacheManager,
    get_cache_manager,
    cached,
    clear_cache
)

# Import enhanced cache functionality
from asf.medical.core.enhanced_cache import (
    EnhancedCacheManager,
    get_enhanced_cache_manager,
    enhanced_cached,
    DataSensitivity
)

# Create enhanced cache manager instance
enhanced_cache_manager = get_enhanced_cache_manager()

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
