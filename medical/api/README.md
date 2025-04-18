# ASF Medical Research Synthesizer API

This directory contains the API implementation for the ASF Medical Research Synthesizer.

## Overview

The ASF Medical Research Synthesizer API provides a comprehensive set of endpoints for searching, analyzing, and synthesizing medical research literature. The API is built using FastAPI and follows modern best practices for API design.

## API Structure

The API is organized into the following components:

- **Main Application**: `main.py` - The main FastAPI application
- **Authentication**: `auth.py` - Unified authentication system
- **Routers**: `routers/` - API endpoints organized by functionality
- **Models**: `models/` - Request and response models
- **Dependencies**: `dependencies.py` - Dependency injection
- **Middleware**: `middleware.py` - Request/response middleware

## Authentication

The API uses JWT-based authentication with role-based access control. The authentication system is implemented in `auth.py` and provides the following features:

- User registration and management
- Token generation and validation
- Role-based access control
- Password hashing with bcrypt

## API Endpoints

The API provides the following endpoint categories:

- **Auth**: User authentication and management
- **Search**: Medical literature search
- **Analysis**: Analysis of medical literature
- **Knowledge Base**: Knowledge base management
- **Export**: Export of search results
- **Screening**: PRISMA-guided screening and bias assessment
- **Contradiction**: Enhanced contradiction detection
- **Status**: API status and health checks
- **Admin**: Administrative endpoints

## Running the API

To run the API, use the following command:

```bash
uvicorn asf.medical.api.main:app --reload
```

## API Documentation

The API documentation is available at the following endpoints:

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`

## Consolidation

The API has been consolidated to improve maintainability, security, and consistency. See `README_CONSOLIDATION.md` for details on the consolidation process and migration guide.

## Development

To contribute to the API, follow these guidelines:

1. Use the unified authentication system in `auth.py`
2. Add new endpoints to the appropriate router in `routers/`
3. Define request and response models in `models/`
4. Use dependency injection for services and repositories
5. Follow the consistent API response format defined in `models/base.py`
6. Add comprehensive error handling
7. Add logging for all operations
8. Add monitoring for performance-critical operations
9. Add tests for all new functionality

## Testing

To run the tests, use the following command:

```bash
python -m pytest asf/medical/tests
```

## Deployment

The API can be deployed using Docker. See `Dockerfile` and `docker-compose.yml` for details.
