"""
Test script for the enhanced caching system.

This script tests the enhanced caching system with Redis support.
"""

import asyncio
import logging
import time
import os
import sys
from typing import Dict, Any

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from asf.medical.core.cache import cache_manager, cached
from asf.medical.core.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Test data
TEST_DATA = {
    "key1": "value1",
    "key2": {
        "nested_key": "nested_value",
        "numbers": [1, 2, 3, 4, 5]
    },
    "key3": [
        {"name": "item1", "value": 100},
        {"name": "item2", "value": 200},
        {"name": "item3", "value": 300}
    ]
}

# Test function with caching
@cached(prefix="test_function", ttl=10)
async def test_function(arg1: str, arg2: int) -> Dict[str, Any]:
    """
    Test function with caching.
    
    Args:
        arg1: First argument
        arg2: Second argument
        
    Returns:
        Dictionary with the arguments and a timestamp
    """
    # Simulate a slow operation
    await asyncio.sleep(1)
    
    return {
        "arg1": arg1,
        "arg2": arg2,
        "timestamp": time.time()
    }

async def test_basic_operations():
    """Test basic cache operations."""
    logger.info("Testing basic cache operations...")
    
    # Set a value
    await cache_manager.set("test_key", TEST_DATA, ttl=60)
    logger.info("Set test_key in cache")
    
    # Get the value
    value = await cache_manager.get("test_key")
    logger.info(f"Got value from cache: {value is not None}")
    
    # Delete the value
    deleted = await cache_manager.delete("test_key")
    logger.info(f"Deleted test_key from cache: {deleted}")
    
    # Get the value again (should be None)
    value = await cache_manager.get("test_key")
    logger.info(f"Got value from cache after deletion: {value is not None}")

async def test_cached_decorator():
    """Test the cached decorator."""
    logger.info("Testing cached decorator...")
    
    # Call the function for the first time (should be slow)
    start_time = time.time()
    result1 = await test_function("test", 123)
    elapsed_time = time.time() - start_time
    logger.info(f"First call took {elapsed_time:.2f} seconds")
    logger.info(f"Result: {result1}")
    
    # Call the function again with the same arguments (should be fast)
    start_time = time.time()
    result2 = await test_function("test", 123)
    elapsed_time = time.time() - start_time
    logger.info(f"Second call took {elapsed_time:.2f} seconds")
    logger.info(f"Result: {result2}")
    
    # Call the function with different arguments (should be slow)
    start_time = time.time()
    result3 = await test_function("test", 456)
    elapsed_time = time.time() - start_time
    logger.info(f"Third call (different args) took {elapsed_time:.2f} seconds")
    logger.info(f"Result: {result3}")

async def test_namespaces():
    """Test cache namespaces."""
    logger.info("Testing cache namespaces...")
    
    # Set values in different namespaces
    await cache_manager.set("test_key", "value1", namespace="namespace1:")
    await cache_manager.set("test_key", "value2", namespace="namespace2:")
    
    # Get values from different namespaces
    value1 = await cache_manager.get("test_key", namespace="namespace1:")
    value2 = await cache_manager.get("test_key", namespace="namespace2:")
    
    logger.info(f"Value from namespace1: {value1}")
    logger.info(f"Value from namespace2: {value2}")
    
    # Clear namespace1
    await cache_manager.clear(namespace="namespace1:")
    
    # Get values again
    value1 = await cache_manager.get("test_key", namespace="namespace1:")
    value2 = await cache_manager.get("test_key", namespace="namespace2:")
    
    logger.info(f"Value from namespace1 after clear: {value1}")
    logger.info(f"Value from namespace2 after clear: {value2}")

async def test_pattern_operations():
    """Test pattern-based cache operations."""
    logger.info("Testing pattern-based cache operations...")
    
    # Set multiple values
    await cache_manager.set("pattern:key1", "value1")
    await cache_manager.set("pattern:key2", "value2")
    await cache_manager.set("pattern:key3", "value3")
    await cache_manager.set("other:key1", "value4")
    
    # Delete keys matching a pattern
    count = await cache_manager.delete_pattern("pattern:*")
    logger.info(f"Deleted {count} keys matching pattern:*")
    
    # Get values
    value1 = await cache_manager.get("pattern:key1")
    value2 = await cache_manager.get("pattern:key2")
    value3 = await cache_manager.get("pattern:key3")
    value4 = await cache_manager.get("other:key1")
    
    logger.info(f"pattern:key1 after delete: {value1}")
    logger.info(f"pattern:key2 after delete: {value2}")
    logger.info(f"pattern:key3 after delete: {value3}")
    logger.info(f"other:key1 after delete: {value4}")

async def test_cache_stats():
    """Test cache statistics."""
    logger.info("Testing cache statistics...")
    
    # Clear cache
    await cache_manager.clear()
    
    # Get initial stats
    stats = await cache_manager.get_stats()
    logger.info(f"Initial stats: {stats}")
    
    # Set and get some values
    await cache_manager.set("stats:key1", "value1")
    await cache_manager.set("stats:key2", "value2")
    
    value1 = await cache_manager.get("stats:key1")
    value2 = await cache_manager.get("stats:key2")
    value3 = await cache_manager.get("stats:key3")  # This should be a miss
    
    # Get updated stats
    stats = await cache_manager.get_stats()
    logger.info(f"Updated stats: {stats}")

async def main():
    """Main function."""
    logger.info("Starting cache test...")
    
    # Initialize cache manager with Redis if configured
    if settings.REDIS_URL:
        # Re-initialize cache manager with Redis URL
        cache_manager.__init__(
            max_size=1000,
            redis_url=settings.REDIS_URL,
            default_ttl=settings.CACHE_TTL,
            namespace="test:"
        )
        logger.info(f"Cache manager initialized with Redis: {settings.REDIS_URL}")
    else:
        logger.info("Cache manager initialized with local LRU cache only")
    
    # Run tests
    await test_basic_operations()
    await test_cached_decorator()
    await test_namespaces()
    await test_pattern_operations()
    await test_cache_stats()
    
    # Clear cache
    await cache_manager.clear()
    logger.info("Cache test completed")

if __name__ == "__main__":
    asyncio.run(main())
