# Cochrane Library Client

This module provides a client for interacting with the Cochrane Library API, which contains high-quality systematic reviews, meta-analyses, and clinical guidelines.

## Features

- Search the Cochrane Library for systematic reviews and clinical trials
- Retrieve complete review information including metadata, abstracts, and full text
- Extract PICO elements (Population, Intervention, Comparison, Outcome) from reviews
- Extract evidence ratings and GRADE assessments
- Search using the PICO framework
- List top cited and recent reviews
- Browse by topics/categories
- Robust error handling, caching, and rate limiting

## Installation

The Cochrane client is part of the ASF Medical Research Synthesizer. No separate installation is needed.

## Usage

### Basic Usage

```python
import asyncio
from asf.medical.clients.cochrane import CochraneClient

async def main():
    # Initialize the client
    client = CochraneClient(
        requests_per_second=2.0,  # Rate limit: 2 requests per second
        burst_size=5,             # Allow bursts of up to 5 requests
        timeout=30.0,             # 30 second timeout for requests
        use_cache=True,           # Enable caching
        cache_ttl=86400          # Cache for 24 hours
    )
    
    try:
        # Check API status
        status = await client.check_api_status()
        print(f"API Status: {status['status']}")
        
        # Search for reviews
        results = await client.search("diabetes mellitus type 2", max_results=5)
        print(f"Found {results['total_results']} results")
        
        for result in results['results']:
            print(f"- {result.get('title')} (ID: {result.get('id')})")
            
    finally:
        # Close the client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Features

#### Search by PICO Elements

```python
# Search using PICO framework
results = await client.search_by_pico(
    population="adults",
    intervention="statins",
    comparison="placebo",
    outcome="cardiovascular disease",
    max_results=10
)
```

#### Get Review Details

```python
# Get a specific review
review = await client.get_review("CD012345")  # Replace with actual review ID

# Extract PICO elements from a review
pico = await client.extract_pico("CD012345")

# Extract evidence ratings
evidence = await client.extract_evidence("CD012345")

# Get full content including HTML sections
html_content = await client.get_review_html_content("CD012345")

# Get complete review with all information
full_review = await client.get_review_with_full_content("CD012345")
```

#### Browse Reviews

```python
# Get recent reviews
recent = await client.get_recent_reviews(count=10)

# Get top cited reviews
top_cited = await client.get_top_cited_reviews(count=10)

# Get list of topics
topics = await client.get_topics()
```

### Configuration

#### Cache Management

```python
# Update cache settings
client.update_cache_settings(use_cache=True, cache_ttl=7200)  # 2 hours

# Count cached items
stats = await client.count_cached_items()
print(f"Cached items: {stats['count']}")

# Clear cache
cleared = await client.clear_cache()
print(f"Cleared {cleared} items from cache")
```

#### Rate Limiting

```python
# Get current rate limit
rate_limit = client.get_rate_limit()
print(f"Rate limit: {rate_limit['requests_per_second']} requests/second")

# Update rate limit
client.update_rate_limit(requests_per_second=3.0)
```

## Example Script

An example script is included that demonstrates the most common use cases. Run it with:

```bash
python -m asf.medical.clients.cochrane.example
```

## API Reference

For complete details on all available methods and parameters, see the docstrings in the code.

## Notes on Rate Limiting

The Cochrane Library may have its own rate limits. This client implements conservative rate limiting by default but you may need to adjust based on the requirements of the API. Always respect the rate limits of the service you are connecting to.

## Dependencies

- `httpx`: For making HTTP requests
- `beautifulsoup4`: For parsing HTML content
- `asf.medical.core`: For rate limiting and caching functionality