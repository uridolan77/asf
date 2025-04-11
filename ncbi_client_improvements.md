# NCBI Client Improvements

## Overview
The NCBI client has been enhanced with the following improvements:

1. **Error Handling**
   - Added custom `NCBIClientError` exception class
   - Improved error messages with context
   - Added proper error handling for network issues, rate limits, and server errors
   - Added validation for input parameters

2. **Caching**
   - Added caching for all API requests
   - Implemented cache key generation based on request parameters
   - Added methods to manage cache (clear, count, get stats)
   - Added configurable cache TTL

3. **Retry Logic**
   - Implemented retry logic with exponential backoff
   - Added configurable max retries
   - Added different retry strategies for different error types

4. **Rate Limiting**
   - Improved rate limiting with configurable limits
   - Added methods to update rate limits
   - Added proper handling of rate limit errors

5. **Logging**
   - Added detailed logging for all operations
   - Added logging of cache hits/misses
   - Added logging of rate limit information
   - Added logging of errors with context

6. **Batch Processing**
   - Improved batch processing with better error handling
   - Added validation for batch parameters
   - Added logging of batch progress

7. **Configuration**
   - Added methods to get and update client configuration
   - Added methods to check API status
   - Added methods to get rate limit and cache information

## New Methods

### Error Handling
- `NCBIClientError` - Custom exception class for NCBI client errors

### Caching
- `clear_cache()` - Clear the cache for this client
- `get_cache_stats()` - Get cache statistics for this client
- `count_cached_items()` - Count the number of cached items for this client
- `update_cache_settings()` - Update the cache settings

### Rate Limiting
- `get_rate_limit()` - Get the current rate limit settings
- `update_rate_limit()` - Update the rate limit settings

### Configuration
- `get_config()` - Get the current configuration of the client
- `check_api_status()` - Check the status of the NCBI API

## Improved Methods

### API Methods
- `search_pubmed()` - Added caching, validation, and better error handling
- `fetch_article_details()` - Added caching, validation, and better error handling
- `fetch_article_abstracts()` - Added caching, validation, and better error handling
- `fetch_pubmed_abstracts()` - Added caching, validation, and better error handling
- `search_and_fetch_pubmed()` - Added caching, validation, and better error handling

### Helper Methods
- `_make_request()` - Added retry logic, caching, and better error handling
- `_batch_pmids()` - Added validation and better error handling

## Usage Examples

### Basic Usage
```python
# Initialize the client
client = NCBIClient(email="example@example.com")

# Search PubMed
results = await client.search_pubmed("cancer", max_results=10)

# Fetch article details
articles = await client.fetch_article_details(results["esearchresult"]["idlist"])

# Search and fetch in one step
articles = await client.search_and_fetch_pubmed("cancer", max_results=10)
```

### Cache Management
```python
# Clear the cache
await client.clear_cache()

# Get cache statistics
stats = await client.get_cache_stats()

# Update cache settings
client.update_cache_settings(use_cache=True, cache_ttl=7200)
```

### Rate Limit Management
```python
# Get rate limit settings
rate_limit = client.get_rate_limit()

# Update rate limit
client.update_rate_limit(requests_per_second=5)
```

### API Status
```python
# Check API status
status = await client.check_api_status()
```

## Configuration Options

### Client Initialization
- `email` - Email address for NCBI API (required)
- `api_key` - API key for NCBI API (optional)
- `tool` - Tool name for NCBI API (default: "MedicalResearchSynthesizer")
- `base_url` - Base URL for NCBI API (default: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/")
- `max_batch_size` - Maximum batch size for requests (default: 200)
- `timeout` - Request timeout in seconds (default: 30.0)
- `max_retries` - Maximum number of retries for failed requests (default: 3)
- `cache_ttl` - Cache TTL in seconds (default: 3600)
- `use_cache` - Whether to use caching (default: True)
