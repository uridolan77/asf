# ASF Medical Research Synthesizer API Documentation

This document describes the API documentation solutions for the ASF Medical Research Synthesizer.

## API Documentation

The API documentation is provided using Swagger UI and ReDoc, which are both based on the OpenAPI specification.

## Accessing Documentation

The API documentation is available at the following endpoints:

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`

## Swagger UI

Swagger UI provides an interactive documentation interface that allows you to:

- Browse API endpoints
- View request and response schemas
- Try out API endpoints
- View authentication requirements

## ReDoc

ReDoc provides a more user-friendly documentation interface that is optimized for reading and navigation.

## OpenAPI Specification

The OpenAPI specification is available in JSON format at `/openapi.json`. This can be used to generate client libraries or import into API development tools.

## Custom Documentation

The API documentation is customized to provide a better user experience:

- Custom Swagger UI with improved styling
- Custom ReDoc with improved styling
- Custom OpenAPI schema with detailed descriptions and examples

## API Endpoints

The API endpoints are organized into the following categories:

- **Auth**: Authentication endpoints
- **Search**: Search endpoints for medical literature
- **Analysis**: Analysis endpoints for medical literature
- **Knowledge Base**: Knowledge base management endpoints
- **Export**: Export endpoints for data export
- **Screening**: PRISMA-guided screening and bias assessment endpoints
- **Contradiction**: Enhanced contradiction detection endpoints
- **Status**: Status endpoints
- **Admin**: Admin endpoints

## Authentication

The API uses JWT (JSON Web Token) authentication. To authenticate:

1. Call the `/token` endpoint with your username and password
2. Use the returned token in the `Authorization` header for subsequent requests

Example:

```http
POST /token
Content-Type: application/x-www-form-urlencoded

username=user&password=password
```

Response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

Then use the token in subsequent requests:

```http
GET /api/v1/search
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## Request and Response Formats

The API uses JSON for request and response bodies. All responses include appropriate HTTP status codes and content types.

### Success Response

```json
{
  "query": "statin therapy cardiovascular",
  "total_results": 100,
  "results": [
    {
      "pmid": "12345678",
      "title": "Statin therapy for cardiovascular disease prevention",
      "abstract": "...",
      "publication_date": "2020-01-01",
      "authors": ["Smith, J", "Jones, K"],
      "journal": "Journal of Cardiology",
      "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    }
  ]
}
```

### Error Response

```json
{
  "detail": "Invalid search query"
}
```

## API Versioning

The API is versioned using URL prefixes. The current version is `v1` and is accessed using the `/api/v1/` prefix.

## Rate Limiting

The API implements rate limiting to prevent abuse. Rate limits are specified in the response headers:

- `X-RateLimit-Limit`: Maximum number of requests allowed in the current time window
- `X-RateLimit-Remaining`: Number of requests remaining in the current time window
- `X-RateLimit-Reset`: Time when the current rate limit window resets (Unix timestamp)

## Pagination

API endpoints that return multiple items support pagination using the following query parameters:

- `skip`: Number of items to skip (default: 0)
- `limit`: Maximum number of items to return (default: 20, max: 100)

Example:

```http
GET /api/v1/search?query=statin&skip=20&limit=10
```

## Filtering and Sorting

API endpoints that return multiple items support filtering and sorting using query parameters:

- `sort`: Field to sort by (e.g., `publication_date`)
- `order`: Sort order (`asc` or `desc`)
- `filter`: Filter criteria (e.g., `journal=Lancet`)

Example:

```http
GET /api/v1/search?query=statin&sort=publication_date&order=desc&filter=journal=Lancet
```

## Examples

### Search for Medical Literature

```http
POST /api/v1/search
Content-Type: application/json
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

{
  "query": "statin therapy cardiovascular",
  "max_results": 10,
  "years": 5,
  "sort_by": "relevance"
}
```

### Analyze Contradictions

```http
POST /api/v1/contradiction/analyze
Content-Type: application/json
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

{
  "query": "statin therapy cardiovascular",
  "max_results": 10,
  "threshold": 0.7,
  "use_all_methods": true
}
```

### PRISMA Screening

```http
POST /api/v1/screening/prisma
Content-Type: application/json
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

{
  "query": "statin therapy cardiovascular",
  "max_results": 10,
  "stage": "screening",
  "criteria": {
    "include": ["randomized controlled trial", "cardiovascular outcomes"],
    "exclude": ["animal study", "in vitro"]
  }
}
```

### Bias Assessment

```http
POST /api/v1/screening/bias-assessment
Content-Type: application/json
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

{
  "query": "statin therapy cardiovascular",
  "max_results": 10,
  "domains": ["randomization", "blinding", "allocation_concealment", "sample_size", "attrition"]
}
```
