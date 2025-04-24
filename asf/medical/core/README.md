# ASF Medical Research Synthesizer Core

This directory contains the core components of the ASF Medical Research Synthesizer, including configuration, caching, and exception handling.

## Caching System

The caching system is implemented in `cache.py` and provides a flexible and efficient way to cache data in the application. It supports both in-memory LRU caching and Redis-based distributed caching.

### Features

- **Dual-layer caching**: Local LRU cache for fast access and Redis for distributed caching
- **TTL-based expiration**: Different TTLs for different types of data
- **Namespace support**: Organize cache keys by namespace
- **Pattern-based operations**: Delete keys matching a pattern
- **Cache statistics**: Track cache hits, misses, and other metrics
- **Decorator-based caching**: Easy to use with async functions

### Usage

#### Basic Usage

```python
from asf.medical.core.cache import cache_manager

# Set a value in the cache
await cache_manager.set("my_key", "my_value", ttl=3600)

# Get a value from the cache
value = await cache_manager.get("my_key")

# Delete a value from the cache
await cache_manager.delete("my_key")
```

#### Using the Decorator

```python
from asf.medical.core.cache import cached

@cached(prefix="my_function", ttl=3600)
async def my_function(arg1, arg2):
    # Expensive operation
    return result
```

#### Using Data Types for TTL Selection

```python
# Set a value with a data type
await cache_manager.set("my_key", "my_value", data_type="search")

# The TTL will be selected based on the data type
# search: 3600 seconds (1 hour)
# analysis: 7200 seconds (2 hours)
# knowledge_base: 86400 seconds (24 hours)
# user: 1800 seconds (30 minutes)
```

#### Using Namespaces

```python
# Set a value with a namespace
await cache_manager.set("my_key", "my_value", namespace="my_namespace:")

# Get a value with a namespace
value = await cache_manager.get("my_key", namespace="my_namespace:")
```

#### Pattern-based Operations

```python
# Delete all keys matching a pattern
count = await cache_manager.delete_pattern("my_namespace:*")
```

#### Cache Statistics

```python
# Get cache statistics
stats = await cache_manager.get_stats()
```

### Configuration

The cache manager can be configured in the `.env` file:

```
# Cache settings
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600  # 1 hour
```

## Exception Handling

The exception handling system is implemented in `exceptions.py` and provides a set of custom exceptions for the application.

### Features

- **Base exception**: `ASFException` is the base exception for all custom exceptions
- **Specific exceptions**: Specific exceptions for different error scenarios
- **Consistent error messages**: All exceptions provide consistent error messages

### Usage

```python
from asf.medical.core.exceptions import ValidationError, ResourceNotFoundError

# Raise a validation error
raise ValidationError("Invalid input")

# Raise a resource not found error
raise ResourceNotFoundError("User", "123")
```

## Configuration

The configuration system is implemented in `config.py` and provides a centralized configuration using Pydantic.

### Features

- **Environment variables**: Load configuration from environment variables
- **Validation**: Validate configuration values
- **Default values**: Provide default values for configuration
- **Type safety**: Type-safe configuration values

### Usage

```python
from asf.medical.core.config import settings

# Access configuration values
database_url = settings.DATABASE_URL
redis_url = settings.REDIS_URL
```

### Configuration

The configuration can be set in the `.env` file:

```
# Database settings
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/medical_research_synthesizer

# Cache settings
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600  # 1 hour
```
