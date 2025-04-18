# ASF Medical Research Synthesizer API Consolidation

This document describes the consolidation of the ASF Medical Research Synthesizer API.

## Overview

The ASF Medical Research Synthesizer API has been consolidated to improve maintainability, security, and consistency. The consolidation focuses on three main areas:

1. **Authentication**: Unified authentication system
2. **API Layer**: Standardized API entry point
3. **Interfaces**: Consistent API interfaces

## 1. Authentication Consolidation

The authentication system has been consolidated into a single, comprehensive system:

- **auth.py**: Unified authentication module with service-based approach
- **routers/auth.py**: Unified authentication router with consistent endpoints

### Key Features

- JWT-based authentication with proper token expiration
- Role-based access control (admin, user)
- Comprehensive user management (create, read, update, delete)
- Dependency injection for easy testing and maintenance
- Proper password hashing with bcrypt
- Consistent error handling

### Usage

```python
from asf.medical.api.auth import (
    get_current_active_user, get_admin_user, has_role, has_any_role
)

@router.get("/protected")
async def protected_endpoint(
    current_user: DBUser = Depends(get_current_active_user)
):
    return {"message": "Protected endpoint", "user": current_user.email}

@router.get("/admin-only")
async def admin_only_endpoint(
    admin_user: DBUser = Depends(get_admin_user)
):
    return {"message": "Admin only endpoint", "user": admin_user.email}

@router.get("/role-specific")
async def role_specific_endpoint(
    user: DBUser = Depends(has_role("researcher"))
):
    return {"message": "Researcher only endpoint", "user": user.email}

@router.get("/multi-role")
async def multi_role_endpoint(
    user: DBUser = Depends(has_any_role(["admin", "researcher"]))
):
    return {"message": "Admin or researcher endpoint", "user": user.email}
```

## 2. API Layer Standardization

The API layer has been standardized to use a single, comprehensive FastAPI application:

- **main.py**: Unified FastAPI application with all routers

### Key Features

- Modern lifespan management for startup and shutdown events
- Comprehensive middleware (CORS, monitoring)
- Consistent error handling
- Unified documentation (Swagger UI, ReDoc)
- Health check and metrics endpoints
- Cache management

### Usage

```bash
# Run the API
uvicorn asf.medical.api.main:app --reload

# Access the API
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

## 3. Interface Consistency

The API interfaces have been standardized to provide consistent request and response models:

- **models/base.py**: Base models for consistent API responses

### Key Features

- Consistent response structure
- Proper error handling
- Pagination support
- Metadata support

### Usage

```python
from asf.medical.api.models.base import APIResponse, PaginatedResponse, ErrorResponse

@router.get("/example", response_model=APIResponse[YourModel])
async def example_endpoint():
    try:
        data = your_service.get_data()
        return APIResponse(
            success=True,
            message="Data retrieved successfully",
            data=data,
            meta={"timestamp": datetime.now().isoformat()}
        )
    except Exception as e:
        return ErrorResponse(
            message="Failed to retrieve data",
            errors=[{"detail": str(e)}],
            code="DATA_RETRIEVAL_ERROR"
        )

@router.get("/paginated", response_model=PaginatedResponse[YourModel])
async def paginated_endpoint(page: int = 1, page_size: int = 10):
    data, total = your_service.get_paginated_data(page, page_size)
    total_pages = (total + page_size - 1) // page_size

    return PaginatedResponse(
        success=True,
        message="Data retrieved successfully",
        data=data,
        meta={"timestamp": datetime.now().isoformat()},
        page=page,
        page_size=page_size,
        total_items=total,
        total_pages=total_pages
    )
```

## Migration Guide

To migrate to the consolidated API:

1. Update imports to use the unified authentication module:
   ```python
   # Old
   from asf.medical.api.old_auth import get_current_user

   # New
   from asf.medical.api.auth import get_current_active_user
   ```

2. Update API entry point:
   ```bash
   # Old
   uvicorn asf.medical.api.old_main:app --reload

   # New
   uvicorn asf.medical.api.main:app --reload
   ```

3. Update response models:
   ```python
   # Old
   @router.get("/example")
   async def example_endpoint():
       return {"data": your_data}

   # New
   from asf.medical.api.models.base import APIResponse

   @router.get("/example", response_model=APIResponse[YourModel])
   async def example_endpoint():
       return APIResponse(
           success=True,
           message="Data retrieved successfully",
           data=your_data
       )
   ```

## Conclusion

The consolidation of the ASF Medical Research Synthesizer API improves maintainability, security, and consistency. The unified authentication system, standardized API layer, and consistent interfaces provide a solid foundation for future development.
