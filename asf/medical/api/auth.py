Unified authentication module for the Medical Research Synthesizer API.

This module provides a comprehensive JWT-based authentication system for the FastAPI implementation,
including user management, token generation, validation, and role-based access control.


from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr, Field

from asf.medical.core.config import settings
from asf.medical.storage.database import get_db_session
from asf.medical.storage.repositories.user_repository import UserRepository
from asf.medical.storage.models import User as DBUser
from asf.medical.core.monitoring import log_error

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/token")

class Token(BaseModel):
    Token model for authentication responses.
    async def _has_any_role(current_user: DBUser = Depends(get_current_active_user)) -> DBUser:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of roles {roles} required"
            )
        return current_user

    return _has_any_role
