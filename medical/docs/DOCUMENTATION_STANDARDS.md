# Documentation Standards for Medical Research Synthesizer

This document outlines the documentation standards for the Medical Research Synthesizer codebase. Following these standards ensures consistency, improves code maintainability, and makes onboarding new developers easier.

## Docstring Format

We use **Google-style docstrings** throughout the codebase. This style is chosen for its readability and compatibility with documentation generation tools.

### Module Docstrings

Every Python module should have a module-level docstring that includes:

1. A brief one-line description
2. A more detailed description (if needed)
3. A list of key components or functions in the module
4. Usage examples (for important modules)

```python
"""
Export utilities for the Medical Research Synthesizer.

This module provides functions to export research data in various formats:
- JSON
- CSV
- Excel
- PDF

Each export function handles data validation, formatting, and error handling
consistently across all formats.

Example:
    ```python
    from asf.medical.api.export_utils_consolidated import export_to_json
    
    data = [{"title": "Research Paper", "authors": ["Smith, J."]}]
    response = export_to_json(data, query_text="cancer research")
    ```
"""
```

### Class Docstrings

Class docstrings should include:

1. A brief description of the class
2. A more detailed description of its purpose and behavior
3. Attributes (class variables and instance variables)
4. Examples of usage (for important classes)

```python
class ExportService:
    """
    Service for exporting research data in various formats.
    
    This service provides methods to export search results and analyses
    to different file formats with consistent error handling and logging.
    
    Attributes:
        export_dir (str): Directory where exported files are stored
        file_handlers (Dict[str, Callable]): Mapping of formats to handler functions
        max_file_size (int): Maximum file size in bytes
    
    Example:
        ```python
        service = ExportService(export_dir="/tmp/exports")
        file_path = await service.export_to_json(data)
        ```
    """
```

### Function/Method Docstrings

Function and method docstrings should include:

1. A brief description of what the function does
2. Args section listing all parameters with types and descriptions
3. Returns section describing the return value with type
4. Raises section listing exceptions that might be raised
5. Examples (for complex or important functions)

```python
def export_to_json(data: List[Dict[str, Any]], query_text: Optional[str] = None) -> JSONResponse:
    """
    Export data to JSON response.
    
    Args:
        data: List of dictionaries containing publication data
        query_text: Optional query text that produced the results
    
    Returns:
        JSONResponse object containing the formatted data
    
    Raises:
        ValueError: If data is not a list or contains invalid items
        ExportError: If there's an issue exporting the data
    
    Example:
        ```python
        data = [{"title": "Research Paper", "authors": ["Smith, J."]}]
        response = export_to_json(data, query_text="cancer research")
        ```
    """
```

### Property Docstrings

Properties should have docstrings that describe what the property represents:

```python
@property
def is_complete(self) -> bool:
    """
    Whether the export operation is complete.
    
    Returns:
        True if the export is complete, False otherwise
    """
    return self._status == "complete"
```

## Type Annotations

All function parameters and return values should include type annotations:

```python
def process_results(results: List[Dict[str, Any]], limit: int = 10) -> Dict[str, Any]:
    """Process search results."""
    # Function implementation
```

## Comments

- Use comments sparingly and only when necessary to explain complex logic
- Prefer descriptive variable names and well-structured code over excessive comments
- Avoid commented-out code; use version control instead
- TODO comments should include a description of what needs to be done and, ideally, a ticket reference

```python
# TODO: Implement caching for export operations (TICKET-123)
```

## Documentation Maintenance

- Update docstrings when changing function signatures or behavior
- Run the docstring checker regularly to identify missing or incomplete documentation:
  ```
  python -m asf.medical.scripts.check_docstrings
  ```
- Use the docstring updater to fix identified issues:
  ```
  python -m asf.medical.scripts.update_docstrings
  ```

## Examples

### Well-Documented Module

```python
"""
Authentication middleware for the Medical Research Synthesizer API.

This module provides middleware components for handling authentication
and authorization in the API. It includes JWT token validation, role-based
access control, and integration with the user service.

Key components:
- JWTAuthMiddleware: Middleware for validating JWT tokens
- RoleBasedAccessControl: Component for enforcing role-based permissions
- get_current_user: Dependency for retrieving the authenticated user

Example:
    ```python
    from fastapi import Depends
    from asf.medical.api.auth import get_current_user
    
    @app.get("/protected")
    async def protected_route(user = Depends(get_current_user)):
        return {"message": f"Hello, {user.username}!"}
    ```
"""

import logging
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from asf.medical.core.exceptions import AuthenticationError, AuthorizationError
from asf.medical.services.user_service import UserService
from asf.medical.storage.models import User

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    user_service: UserService = Depends(get_user_service)
) -> User:
    """
    Get the current authenticated user.
    
    This dependency validates the JWT token and returns the corresponding user.
    
    Args:
        token: JWT token from the Authorization header
        user_service: Service for user operations
    
    Returns:
        User object for the authenticated user
    
    Raises:
        HTTPException: If the token is invalid or the user is not found
    """
    try:
        user = await user_service.get_user_from_token(token)
        return user
    except AuthenticationError as e:
        logger.warning(f"Authentication failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
```

### Well-Documented Class

```python
class SearchService:
    """
    Service for searching medical research literature.
    
    This service provides methods for searching medical literature using
    various criteria and filters. It supports both simple keyword searches
    and advanced structured queries.
    
    Attributes:
        search_client: Client for the search engine
        cache_manager: Manager for caching search results
        max_results: Maximum number of results to return
    
    Example:
        ```python
        service = SearchService(search_client, cache_manager)
        results = await service.search("cancer treatment", limit=10)
        ```
    """
    
    def __init__(
        self,
        search_client: SearchClient,
        cache_manager: CacheManager,
        max_results: int = 100
    ):
        """
        Initialize the search service.
        
        Args:
            search_client: Client for the search engine
            cache_manager: Manager for caching search results
            max_results: Maximum number of results to return
        """
        self.search_client = search_client
        self.cache_manager = cache_manager
        self.max_results = max_results
```
