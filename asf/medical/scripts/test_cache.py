Test script for the enhanced caching system.

This script tests the enhanced caching system with Redis support.

import asyncio
import logging
import time
import os
import sys
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from asf.medical.core.enhanced_cache import enhanced_cache_manager as cache_manager, enhanced_cached as cached, cached
from asf.medical.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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

@enhanced_cached(prefix="test_function", ttl=10)
async def test_function(arg1: str, arg2: int) -> Dict[str, Any]:
    await asyncio.sleep(1)
    
    return {
        "arg1": arg1,
        "arg2": arg2,
        "timestamp": time.time()
    }

async def test_basic_operations():
    logger.info("Testing cached decorator...")
    
    start_time = time.time()
    result1 = await test_function("test", 123)
    elapsed_time = time.time() - start_time
    logger.info(f"First call took {elapsed_time:.2f} seconds")
    logger.info(f"Result: {result1}")
    
    start_time = time.time()
    result2 = await test_function("test", 123)
    elapsed_time = time.time() - start_time
    logger.info(f"Second call took {elapsed_time:.2f} seconds")
    logger.info(f"Result: {result2}")
    
    start_time = time.time()
    result3 = await test_function("test", 456)
    elapsed_time = time.time() - start_time
    logger.info(f"Third call (different args) took {elapsed_time:.2f} seconds")
    logger.info(f"Result: {result3}")

async def test_namespaces():
    logger.info("Testing pattern-based cache operations...")
    
    await enhanced_cache_manager.set("pattern:key1", "value1")
    await enhanced_cache_manager.set("pattern:key2", "value2")
    await enhanced_cache_manager.set("pattern:key3", "value3")
    await enhanced_cache_manager.set("other:key1", "value4")
    
    count = await cache_manager.delete_pattern("pattern:*")
    logger.info(f"Deleted {count} keys matching pattern:*")
    
    value1 = await enhanced_cache_manager.get("pattern:key1")
    value2 = await enhanced_cache_manager.get("pattern:key2")
    value3 = await enhanced_cache_manager.get("pattern:key3")
    value4 = await enhanced_cache_manager.get("other:key1")
    
    logger.info(f"pattern:key1 after delete: {value1}")
    logger.info(f"pattern:key2 after delete: {value2}")
    logger.info(f"pattern:key3 after delete: {value3}")
    logger.info(f"other:key1 after delete: {value4}")

async def test_cache_stats():
    logger.info("Starting cache test...")
    
    if settings.REDIS_URL:
        cache_manager.__init__(
            max_size=1000,
            redis_url=settings.REDIS_URL,
            default_ttl=settings.CACHE_TTL,
            namespace="test:"
        )
        logger.info(f"Cache manager initialized with Redis: {settings.REDIS_URL}")
    else:
        logger.info("Cache manager initialized with local LRU cache only")
    
    await test_basic_operations()
    await test_cached_decorator()
    await test_namespaces()
    await test_pattern_operations()
    await test_cache_stats()
    
    await enhanced_cache_manager.clear()
    logger.info("Cache test completed")

if __name__ == "__main__":
    asyncio.run(main())
