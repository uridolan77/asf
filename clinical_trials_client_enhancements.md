# Clinical Trials Client Enhancements

## Overview
The ClinicalTrials.gov client has been enhanced with improved error handling, retry logic, caching, and rate limiting to provide a more robust and efficient interface to the ClinicalTrials.gov API.

## Key Enhancements

### 1. Improved Error Handling
- Custom `ClinicalTrialsClientError` exception class with status code tracking
- Detailed error messages with context
- Specific handling for different HTTP status codes
- Proper handling of JSON parsing errors

### 2. Retry Logic
- Exponential backoff with jitter for failed requests
- Configurable maximum number of retries
- Special handling for rate limiting (HTTP 429)
- Detailed logging of retry attempts

### 3. Caching
- In-memory LRU cache for API responses
- Configurable TTL (Time-To-Live) for different types of requests
- Cache key generation based on method name and parameters
- Cache hit/miss logging

### 4. Rate Limiting
- Token bucket algorithm for rate limiting
- Configurable requests per second and burst size
- Asynchronous implementation with proper locking
- Prevention of API throttling

### 5. Batch Operations
- Parallel fetching of multiple studies
- Configurable concurrency limit
- Error handling for individual study fetches
- Aggregation of results

## Usage Examples

### Basic Search
```python
client = ClinicalTrialsClient()
studies = await client.search("cancer", max_results=10)
```

### Get Study Details
```python
client = ClinicalTrialsClient()
study = await client.get_study("NCT04280705")
```

### Batch Get Studies
```python
client = ClinicalTrialsClient()
studies = await client.batch_get_studies(["NCT04280705", "NCT04368728", "NCT04380688"], max_concurrent=3)
```

## Configuration Options

### Client Initialization
```python
client = ClinicalTrialsClient(
    base_url="https://clinicaltrials.gov/api/v2",
    timeout=30.0,
    max_retries=3,
    requests_per_second=5.0,
    burst_size=10,
    cache_ttl=3600  # 1 hour
)
```

### Cache Configuration
The `clinical_trials_cache` decorator can be configured with:
- `ttl`: Time-to-live in seconds (default: 3600 = 1 hour)
- `prefix`: Cache key prefix (default: "ct")
- `data_type`: Type of data being cached (default: "search")

## Error Handling

The client now provides more detailed error information:

```python
try:
    studies = await client.search("cancer")
except ClinicalTrialsClientError as e:
    print(f"Error: {e.message}")
    print(f"Status code: {e.status_code}")
```

## Performance Considerations

### Caching
- Search results are cached for 1 hour by default
- Study details are cached for 24 hours by default
- Batch operations are cached for 1 hour by default

### Rate Limiting
- Default rate limit is 5 requests per second
- Burst size of 10 requests allows for short bursts of activity
- Rate limiting is applied at the request level, not the method level

## Implementation Details

### Rate Limiter
The client uses the `AsyncRateLimiter` class from the core module, which implements a token bucket algorithm for rate limiting.

### Cache
The client uses the `cache_manager` from the core module, which provides an LRU cache implementation.

### Retry Logic
The client implements custom retry logic with exponential backoff and jitter to handle transient errors.

### Error Handling
The client provides detailed error messages and proper handling of different types of errors, including HTTP errors, JSON parsing errors, and timeouts.

### Batch Operations
The client uses `asyncio.gather` with a semaphore to limit concurrency when fetching multiple studies in parallel.
