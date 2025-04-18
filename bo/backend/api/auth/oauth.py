"""
OAuth 2.0/OIDC authentication for the API.

This module provides OAuth 2.0/OIDC authentication for the API,
including token validation, user info retrieval, and role mapping.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import httpx
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2AuthorizationCodeBearer, OAuth2PasswordBearer

from api.models.user import User
from api.auth.dependencies import get_current_user as get_jwt_user
from api.config import settings

logger = logging.getLogger(__name__)


# OAuth2 configuration
OAUTH2_ENABLED = settings.OAUTH2_ENABLED
OAUTH2_ISSUER = settings.OAUTH2_ISSUER
OAUTH2_CLIENT_ID = settings.OAUTH2_CLIENT_ID
OAUTH2_CLIENT_SECRET = settings.OAUTH2_CLIENT_SECRET
OAUTH2_AUDIENCE = settings.OAUTH2_AUDIENCE
OAUTH2_SCOPE = settings.OAUTH2_SCOPE
OAUTH2_JWKS_URI = settings.OAUTH2_JWKS_URI
OAUTH2_TOKEN_URI = settings.OAUTH2_TOKEN_URI
OAUTH2_USERINFO_URI = settings.OAUTH2_USERINFO_URI
OAUTH2_ROLE_MAPPING = settings.OAUTH2_ROLE_MAPPING


# OAuth2 security schemes
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=f"{OAUTH2_ISSUER}/protocol/openid-connect/auth",
    tokenUrl=f"{OAUTH2_ISSUER}/protocol/openid-connect/token",
    scopes={
        "openid": "OpenID Connect",
        "profile": "User profile",
        "email": "User email"
    }
) if OAUTH2_ENABLED else None

password_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


# JWKS cache
_jwks_cache = None
_jwks_cache_timestamp = 0
_jwks_cache_ttl = 3600  # 1 hour


async def get_jwks() -> Dict[str, Any]:
    """
    Get JSON Web Key Set (JWKS) from the OAuth2 provider.
    
    Returns:
        JWKS
    """
    global _jwks_cache, _jwks_cache_timestamp
    
    # Return cached JWKS if still valid
    if _jwks_cache and time.time() - _jwks_cache_timestamp < _jwks_cache_ttl:
        return _jwks_cache
    
    # Fetch JWKS from provider
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(OAUTH2_JWKS_URI)
            response.raise_for_status()
            
            # Update cache
            _jwks_cache = response.json()
            _jwks_cache_timestamp = time.time()
            
            return _jwks_cache
    except Exception as e:
        logger.error(f"Error fetching JWKS: {str(e)}")
        
        # Return cached JWKS if available, even if expired
        if _jwks_cache:
            return _jwks_cache
        
        # Raise exception if no cached JWKS
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch JWKS"
        )


async def get_public_key(kid: str) -> Optional[str]:
    """
    Get public key for a key ID.
    
    Args:
        kid: Key ID
        
    Returns:
        Public key or None if not found
    """
    jwks = await get_jwks()
    
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            # Construct public key from JWKS
            if key.get("kty") == "RSA":
                return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key))
    
    return None


async def validate_token(token: str) -> Dict[str, Any]:
    """
    Validate OAuth2 token.
    
    Args:
        token: OAuth2 token
        
    Returns:
        Token claims
    """
    try:
        # Get token header
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        
        if not kid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token header"
            )
        
        # Get public key
        public_key = await get_public_key(kid)
        
        if not public_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Key not found"
            )
        
        # Validate token
        claims = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=OAUTH2_AUDIENCE,
            issuer=OAUTH2_ISSUER
        )
        
        return claims
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error validating token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed"
        )


async def get_user_info(token: str) -> Dict[str, Any]:
    """
    Get user info from OAuth2 provider.
    
    Args:
        token: OAuth2 token
        
    Returns:
        User info
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                OAUTH2_USERINFO_URI,
                headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status()
            
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching user info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user info"
        )


async def map_roles(claims: Dict[str, Any]) -> List[str]:
    """
    Map OAuth2 roles to application roles.
    
    Args:
        claims: Token claims
        
    Returns:
        Application roles
    """
    # Get roles from claims
    oauth_roles = []
    
    # Check realm_access roles
    realm_access = claims.get("realm_access", {})
    if realm_access and isinstance(realm_access, dict):
        oauth_roles.extend(realm_access.get("roles", []))
    
    # Check resource_access roles
    resource_access = claims.get("resource_access", {})
    if resource_access and isinstance(resource_access, dict):
        client_access = resource_access.get(OAUTH2_CLIENT_ID, {})
        if client_access and isinstance(client_access, dict):
            oauth_roles.extend(client_access.get("roles", []))
    
    # Map roles
    app_roles = []
    
    for oauth_role in oauth_roles:
        if oauth_role in OAUTH2_ROLE_MAPPING:
            app_roles.append(OAUTH2_ROLE_MAPPING[oauth_role])
    
    return app_roles


async def get_current_user(token: str = Depends(oauth2_scheme or password_scheme)) -> User:
    """
    Get current user from OAuth2 token.
    
    Args:
        token: OAuth2 token
        
    Returns:
        User
    """
    if not OAUTH2_ENABLED:
        # Fall back to JWT authentication
        return await get_jwt_user(token)
    
    # Validate token
    claims = await validate_token(token)
    
    # Get user info
    user_info = await get_user_info(token)
    
    # Map roles
    roles = await map_roles(claims)
    
    # Create user
    user = User(
        id=user_info.get("sub"),
        username=user_info.get("preferred_username"),
        email=user_info.get("email"),
        first_name=user_info.get("given_name"),
        last_name=user_info.get("family_name"),
        roles=roles,
        is_active=True
    )
    
    return user
