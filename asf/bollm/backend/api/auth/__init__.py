"""
Authentication package for the BO backend.
"""

# Import from dependencies
from .dependencies import get_current_user, get_current_user_ws, get_db

# Import from auth_utils
from .auth_utils import (
    authenticate_user,
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    verify_password,
    oauth2_scheme,
    SECRET_KEY,
    ALGORITHM
)

# Import User model for convenience
from models.user import User
