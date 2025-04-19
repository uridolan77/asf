import secrets
from typing import Optional, List
from fastapi import Security, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# API Key header
api_key_header = APIKeyHeader(name="X-API-KEY")


class Settings(BaseModel):
    """Application settings."""
    api_key: str
    allowed_origins: List[str] = ["*"]
    debug: bool = False


def get_settings() -> Settings:
    """Get the application settings.
    
    This would typically load from environment variables or a config file.
    For this example, we'll just return a hardcoded Settings object.
    """
    return Settings(
        api_key="test-api-key",
        allowed_origins=["http://localhost:3000"],
        debug=True
    )


async def validate_api_key(
    api_key: str = Security(api_key_header),
    config: Settings = Depends(get_settings)
):
    """Validate the API key.
    
    Args:
        api_key: The API key from the request header
        config: The application settings
        
    Raises:
        HTTPException: If the API key is invalid
    """
    if not secrets.compare_digest(api_key, config.api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    return api_key


class RBACMiddleware:
    """Role-Based Access Control middleware."""
    
    def __init__(self, app):
        """Initialize the RBAC middleware.
        
        Args:
            app: The FastAPI application
        """
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """Process an incoming request.
        
        Args:
            scope: The ASGI scope
            receive: The ASGI receive function
            send: The ASGI send function
        """
        if scope["type"] == "http":
            # In a real implementation, we would validate JWT and check permissions
            await self._check_permissions(scope)
        await self.app(scope, receive, send)
    
    async def _check_permissions(self, scope):
        """Check if the user has permission to access the endpoint.
        
        Args:
            scope: The ASGI scope
            
        Raises:
            HTTPException: If the user doesn't have permission
        """
        # This is a placeholder for actual permission checking
        # In a real implementation, we would extract the user from the JWT
        # and check if they have the required permissions
        pass
