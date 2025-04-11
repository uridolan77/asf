# Scalability Improvements for ASF Medical Research Synthesizer

This document outlines the scalability improvements made to the ASF Medical Research Synthesizer to address the identified weaknesses and implement the recommended solutions.

## 1. Enhanced Cache System

### Problem
The original caching system used in-memory data structures that were local to a process. In a multi-instance deployment, this could lead to cache misses or inconsistent views of task status.

### Solution
We implemented an enhanced cache system that uses Redis as the primary cache store, with a local LRU cache as a fallback. This ensures consistent state across multiple instances of the application.

### Implementation
- `asf/medical/core/enhanced_cache.py`: Enhanced cache manager with Redis support
- Key features:
  - Mandatory Redis support in production environments
  - Local LRU cache as a fallback
  - Configurable TTLs for different types of data
  - Cache namespaces and pattern-based operations
  - Cache statistics and monitoring

### Usage
```python
from asf.medical.core.enhanced_cache import enhanced_cache_manager, enhanced_cached

# Set a value in the cache
await enhanced_cache_manager.set("my_key", "my_value", ttl=3600, data_type="search")

# Get a value from the cache
value = await enhanced_cache_manager.get("my_key")

# Use the decorator for caching function results
@enhanced_cached(prefix="my_function", data_type="search")
async def my_function(arg1, arg2):
    # Expensive operation
    return result
```

## 2. Batch Database Operations

### Problem
The original implementation performed database operations one by one, which could be a bottleneck when dealing with large datasets.

### Solution
We implemented batch database operations to optimize database operations when dealing with large datasets.

### Implementation
- `asf/medical/storage/batch_operations.py`: Batch database operations utility
- Key features:
  - Batch insert, update, delete, and fetch operations
  - Configurable batch size
  - Error handling and transaction management
  - Support for upsert operations
  - Concurrency control for batch operations

### Usage
```python
from asf.medical.storage.batch_operations import BatchOperations

# Batch insert
results = await BatchOperations.batch_insert(
    db=db,
    model=User,
    items=[{"name": "User 1"}, {"name": "User 2"}],
    batch_size=100
)

# Batch update
count = await BatchOperations.batch_update(
    db=db,
    model=User,
    items=[{"id": 1, "name": "Updated User 1"}, {"id": 2, "name": "Updated User 2"}],
    primary_key="id",
    batch_size=100
)

# Batch delete
count = await BatchOperations.batch_delete(
    db=db,
    model=User,
    ids=[1, 2, 3],
    primary_key="id",
    batch_size=100
)

# Execute a function in batches with concurrency control
results = await BatchOperations.execute_in_batches(
    func=process_batch,
    items=items,
    batch_size=100,
    max_concurrency=5
)
```

## 3. Global Rate Limiter

### Problem
The original implementation did not have a global rate limit, which could allow a single user to consume too many resources.

### Solution
We implemented a global rate limiter that uses Redis for distributed rate limiting, with a local fallback.

### Implementation
- `asf/medical/core/enhanced_rate_limiter.py`: Enhanced rate limiter with Redis support
- `asf/medical/api/middleware/rate_limit_middleware.py`: FastAPI middleware for rate limiting
- Key features:
  - Token bucket algorithm for rate limiting
  - Distributed rate limiting with Redis
  - Local fallback if Redis is not available
  - Configurable rate limits per user/IP
  - Rate limit headers in responses

### Usage
```python
from asf.medical.core.enhanced_rate_limiter import enhanced_rate_limiter

# Check if rate limited
is_limited, limit_info = await enhanced_rate_limiter.is_rate_limited(
    key="user:123",
    rate=60,  # 60 requests per minute
    burst=10,  # 10 requests in a burst
    window=60  # 1 minute window
)

# Add rate limit middleware to FastAPI app
from asf.medical.api.middleware.rate_limit_middleware import add_rate_limit_middleware

add_rate_limit_middleware(
    app=app,
    default_rate=60,
    default_burst=10,
    default_window=60,
    exempt_paths=["/docs", "/redoc", "/openapi.json", "/health"]
)
```

## 4. Optimized Model Loading

### Problem
The original implementation loaded ML models in each worker process, which could exhaust memory if the number of workers is high.

### Solution
We implemented an optimized model loading system that efficiently manages ML models in memory and supports sharing models across processes.

### Implementation
- `asf/medical/ml/optimized_model_loader.py`: Optimized model loader with shared memory support
- Key features:
  - Lazy loading of models
  - Automatic unloading of unused models
  - Shared memory support for sharing models across processes
  - Memory usage tracking
  - Factory registry for model creation

### Usage
```python
from asf.medical.ml.optimized_model_loader import optimized_model_loader, lazy_load_model

# Get or create a model
model = optimized_model_loader.get_or_create(
    model_id="biomedlm",
    factory=lambda: create_model(),
    metadata={"memory_mb": 1024}
)

# Use the decorator for lazy loading models
class MyService:
    @lazy_load_model(model_id="biomedlm", metadata={"memory_mb": 1024})
    def create_model(self):
        # Create and return the model
        return model
```

## 5. Enhanced Task Storage

### Problem
The original implementation stored task results in memory, which could lead to inconsistent views of task status across multiple instances.

### Solution
We implemented an enhanced task storage system that uses Redis for persistent storage, with a local fallback.

### Implementation
- `asf/medical/core/enhanced_task_storage.py`: Enhanced task storage with Redis support
- Key features:
  - Persistent storage of task results
  - Task status and progress tracking
  - Waiting for task completion
  - Automatic cleanup of expired tasks
  - Local fallback if Redis is not available

### Usage
```python
from asf.medical.core.enhanced_task_storage import enhanced_task_storage

# Set a task result
await enhanced_task_storage.set_task_result(
    task_id="task-123",
    result={"data": "result"},
    metadata={"status": "completed"}
)

# Get a task result
task_data = await enhanced_task_storage.get_task_result("task-123")

# Update task status
await enhanced_task_storage.update_task_status(
    task_id="task-123",
    status="in_progress",
    progress=0.5,
    message="Processing data"
)

# Wait for a task to complete
result = await enhanced_task_storage.wait_for_task(
    task_id="task-123",
    timeout=300.0,
    poll_interval=1.0
)
```

## 6. Integration with Existing Code

To integrate these improvements with the existing codebase, you can:

1. Replace the existing cache manager with the enhanced cache manager:
   ```python
   from asf.medical.core.enhanced_cache import enhanced_cache_manager as cache_manager
   ```

2. Use batch operations in repositories:
   ```python
   from asf.medical.storage.batch_operations import BatchOperations
   
   class UserRepository:
       async def create_many(self, db: AsyncSession, users: List[Dict[str, Any]]) -> List[User]:
           return await BatchOperations.batch_insert(db, User, users)
   ```

3. Add rate limit middleware to the FastAPI app:
   ```python
   from asf.medical.api.middleware.rate_limit_middleware import add_rate_limit_middleware
   
   add_rate_limit_middleware(app)
   ```

4. Use the optimized model loader in ML services:
   ```python
   from asf.medical.ml.optimized_model_loader import optimized_model_loader
   
   class BioMedLMService:
       @property
       def model(self):
           return optimized_model_loader.get_or_create(
               model_id=f"biomedlm:{self.model_name}",
               factory=self._create_model,
               metadata={"memory_mb": 1024}
           )
   ```

5. Use the enhanced task storage in background tasks:
   ```python
   from asf.medical.core.enhanced_task_storage import enhanced_task_storage
   
   @dramatiq.actor
   async def process_task(task_id: str, data: Dict[str, Any]):
       try:
           # Update task status
           await enhanced_task_storage.update_task_status(
               task_id=task_id,
               status="in_progress",
               progress=0.0,
               message="Starting task"
           )
           
           # Process task
           result = await process_data(data)
           
           # Set task result
           await enhanced_task_storage.set_task_result(
               task_id=task_id,
               result=result,
               metadata={"status": "completed"}
           )
       except Exception as e:
           # Set task error
           await enhanced_task_storage.set_task_result(
               task_id=task_id,
               result=None,
               metadata={"status": "error", "error": str(e)}
           )
   ```

## 7. Configuration

These improvements can be configured using environment variables:

### Enhanced Cache
- `REDIS_URL`: Redis URL for distributed caching
- `ENVIRONMENT`: Environment (development or production)
- `CACHE_TTL`: Default cache TTL in seconds

### Rate Limiter
- `REDIS_URL`: Redis URL for distributed rate limiting
- `DEFAULT_RATE`: Default rate limit in requests per minute
- `DEFAULT_BURST`: Default burst limit in requests
- `DEFAULT_WINDOW`: Default window size in seconds

### Optimized Model Loader
- `MAX_CACHED_MODELS`: Maximum number of models to keep in memory
- `MODEL_CACHE_TTL`: Time to live in seconds
- `MODEL_CACHE_CHECK_INTERVAL`: Interval in seconds for checking expired models
- `SHARED_MEMORY_DIR`: Directory for shared memory files
- `USE_SHARED_MEMORY`: Whether to use shared memory

### Enhanced Task Storage
- `REDIS_URL`: Redis URL for persistent storage
- `TASK_TTL`: Time to live in seconds for task results

## 8. Deployment Considerations

When deploying the ASF Medical Research Synthesizer with these improvements, consider the following:

1. **Redis**: Set up a Redis server or cluster for distributed caching, rate limiting, and task storage.

2. **Database**: Use a PostgreSQL database with connection pooling for optimal performance.

3. **Workers**: Configure the number of worker processes and threads based on the available resources.

4. **Memory**: Monitor memory usage, especially for ML models, and adjust the `MAX_CACHED_MODELS` accordingly.

5. **Rate Limits**: Set appropriate rate limits based on the expected usage patterns.

6. **Monitoring**: Implement monitoring for key performance metrics (through the observability module with Prometheus, etc.).

7. **Scaling**: Use horizontal scaling (multiple instances) for handling increased load.

## 9. Future Improvements

While these improvements address the identified weaknesses, there are still opportunities for further optimization:

1. **Circuit Breakers**: Implement circuit breakers for external API calls to prevent cascading failures.

2. **Distributed Tracing**: Add distributed tracing to identify bottlenecks in the request flow.

3. **Caching Strategies**: Implement more sophisticated caching strategies, such as cache warming and cache invalidation.

4. **Model Sharing**: Explore more efficient ways to share ML models across processes, such as using TensorRT or ONNX.

5. **Database Sharding**: Implement database sharding for handling very large datasets.

6. **Async Database Driver**: Use an async database driver for PostgreSQL to maximize concurrency.

7. **Load Testing**: Conduct load testing to identify any remaining bottlenecks.

## 10. Conclusion

These scalability improvements address the identified weaknesses in the ASF Medical Research Synthesizer and implement the recommended solutions. By using Redis for shared state, optimizing database operations, implementing rate limiting, and improving model loading, the system is now better equipped to handle high concurrency and large data volumes.
