# Pagination Implementation

This document describes the implementation of pagination for the search endpoints in the Medical Research Synthesizer API.

## Overview

Pagination has been implemented for the search endpoints to improve performance and user experience when dealing with large result sets. The implementation follows RESTful API best practices and provides a consistent interface for clients.

## Pagination Parameters

The following pagination parameters are supported:

- `page`: The page number (1-based)
- `page_size`: The number of results per page (default: 20, max: 100)

These parameters are encapsulated in a `PaginationParams` model that is included in the request models.

## Request Models

The `QueryRequest` and `PICORequest` models have been updated to include pagination parameters:

```python
class PaginationParams(BaseModel):
    """Pagination parameters for search requests."""
    page: int = Field(1, description="Page number (1-based)", ge=1)
    page_size: int = Field(20, description="Number of results per page", ge=1, le=100)

class QueryRequest(BaseModel):
    """Request model for the search endpoint."""
    query: str = Field(..., description="The search query")
    max_results: int = Field(100, description="Maximum number of results to return", ge=1, le=500)
    pagination: PaginationParams = Field(default_factory=PaginationParams, description="Pagination parameters")
```

## Response Format

The search response includes pagination metadata:

```python
class PaginationMetadata(BaseModel):
    """Pagination metadata for search responses."""
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of results per page")
    total_pages: int = Field(..., description="Total number of pages")
    total_count: int = Field(..., description="Total number of results")
    has_previous: bool = Field(..., description="Whether there is a previous page")
    has_next: bool = Field(..., description="Whether there is a next page")

class SearchResponse(BaseModel):
    """Response model for the search endpoint."""
    query: str = Field(..., description="The search query")
    total_count: int = Field(..., description="Total number of results")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    result_id: str = Field(..., description="Unique ID for this search result")
    pagination: PaginationMetadata = Field(..., description="Pagination metadata")
```

## Implementation Details

### Search Service

The search service has been updated to support pagination:

1. The `search` method now accepts `page` and `page_size` parameters
2. The method validates these parameters to ensure they are within acceptable ranges
3. After retrieving and enriching the search results, the method applies pagination:
   - Calculates the total number of pages
   - Slices the results based on the requested page and page size
   - Adds pagination metadata to the response

### API Endpoints

The search endpoints (`/search` and `/search/pico`) have been updated to:

1. Accept pagination parameters from the request
2. Pass these parameters to the search service
3. Include pagination metadata in the response

## Example Usage

### Basic Search with Pagination

```http
POST /search
Content-Type: application/json

{
  "query": "covid-19 treatment",
  "max_results": 100,
  "pagination": {
    "page": 1,
    "page_size": 20
  }
}
```

### Response

```json
{
  "success": true,
  "message": "Search completed successfully",
  "data": {
    "query": "covid-19 treatment",
    "total_count": 150,
    "results": [...],
    "result_id": "abc123",
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total_pages": 8,
      "total_count": 150,
      "has_previous": false,
      "has_next": true
    }
  },
  "meta": {
    "query": "covid-19 treatment",
    "max_results": 100,
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total_pages": 8,
      "total_count": 150
    },
    "user_id": 123
  }
}
```

## Client Implementation

Clients should use the pagination metadata to navigate through the result set:

```javascript
// Example client implementation
async function fetchAllResults(query) {
  let allResults = [];
  let page = 1;
  let hasMore = true;
  
  while (hasMore) {
    const response = await fetch('/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        pagination: { page, page_size: 20 }
      })
    });
    
    const data = await response.json();
    allResults = allResults.concat(data.data.results);
    
    hasMore = data.data.pagination.has_next;
    page++;
  }
  
  return allResults;
}
```

## Performance Considerations

- The pagination implementation significantly reduces the amount of data transferred in each request
- It also reduces the processing time for each request, as only a subset of the results needs to be processed
- The implementation uses efficient slicing of the result set to avoid unnecessary processing
- The pagination metadata provides clients with the information they need to efficiently navigate through the result set

## Future Improvements

- Add cursor-based pagination for very large result sets
- Add support for sorting results
- Add support for filtering results
- Add support for field selection to further reduce the amount of data transferred
