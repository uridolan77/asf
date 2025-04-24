from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging
# Use absolute imports instead of relative imports
from repositories.provider_repository import ProviderRepository
from utils.crypto import encrypt_value, decrypt_value

logger = logging.getLogger(__name__)

class UserProviderService:
    def __init__(self, db: Session, encryption_key: bytes = None, current_user_id: Optional[int] = None):
        self.db = db
        self.encryption_key = encryption_key
        self.current_user_id = current_user_id
        self.provider_repo = ProviderRepository(db, encryption_key)

    def assign_user_to_provider(self, provider_id: str, user_id: int, role: str = "user") -> bool:
        """Assign a user to a provider with a specific role."""
        try:
            # Check if the assignment already exists
            stmt = text("""
            SELECT * FROM users_providers
            WHERE user_id = :user_id AND provider_id = :provider_id
            """)
            result = self.db.execute(stmt, {"user_id": user_id, "provider_id": provider_id}).fetchone()

            if result:
                # Update the role if it exists
                stmt = text("""
                UPDATE users_providers
                SET role = :role
                WHERE user_id = :user_id AND provider_id = :provider_id
                """)
                self.db.execute(stmt, {"user_id": user_id, "provider_id": provider_id, "role": role})
            else:
                # Create a new assignment
                stmt = text("""
                INSERT INTO users_providers (user_id, provider_id, role)
                VALUES (:user_id, :provider_id, :role)
                """)
                self.db.execute(stmt, {"user_id": user_id, "provider_id": provider_id, "role": role})

            self.db.commit()

            # Log the assignment
            logger.info(f"User {user_id} assigned to provider {provider_id} with role {role}")

            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error assigning user to provider: {e}")
            return False

    def remove_user_from_provider(self, provider_id: str, user_id: int) -> bool:
        """Remove a user from a provider."""
        try:
            stmt = text("""
            DELETE FROM users_providers
            WHERE user_id = :user_id AND provider_id = :provider_id
            """)
            self.db.execute(stmt, {"user_id": user_id, "provider_id": provider_id})
            self.db.commit()

            # Log the removal
            logger.info(f"User {user_id} removed from provider {provider_id}")

            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error removing user from provider: {e}")
            return False

    def get_users_for_provider(self, provider_id: str) -> List[Dict[str, Any]]:
        """Get all users assigned to a provider."""
        try:
            stmt = text("""
            SELECT u.id, u.username, u.email, up.role, up.created_at
            FROM users u
            JOIN users_providers up ON u.id = up.user_id
            WHERE up.provider_id = :provider_id
            """)
            result = self.db.execute(stmt, {"provider_id": provider_id}).fetchall()

            users = []
            for row in result:
                users.append({
                    "user_id": row.id,
                    "username": row.username,
                    "email": row.email,
                    "role": row.role,
                    "assigned_at": row.created_at
                })

            return users
        except Exception as e:
            logger.error(f"Error getting users for provider: {e}")
            return []

    def get_providers_for_user(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all providers assigned to a user."""
        try:
            stmt = text("""
            SELECT p.provider_id, p.display_name, p.provider_type, p.enabled, up.role, up.created_at
            FROM providers p
            JOIN users_providers up ON p.provider_id = up.provider_id
            WHERE up.user_id = :user_id
            """)
            result = self.db.execute(stmt, {"user_id": user_id}).fetchall()

            providers = []
            for row in result:
                providers.append({
                    "provider_id": row.provider_id,
                    "display_name": row.display_name,
                    "provider_type": row.provider_type,
                    "enabled": row.enabled,
                    "role": row.role,
                    "assigned_at": row.created_at
                })

            return providers
        except Exception as e:
            logger.error(f"Error getting providers for user: {e}")
            return []

    def get_user_role_for_provider(self, provider_id: str, user_id: int) -> Optional[str]:
        """Get the role of a user for a specific provider."""
        try:
            stmt = text("""
            SELECT role FROM users_providers
            WHERE user_id = :user_id AND provider_id = :provider_id
            """)
            result = self.db.execute(stmt, {"user_id": user_id, "provider_id": provider_id}).fetchone()

            if result:
                return result.role
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting user role for provider: {e}")
            return None

    def check_user_has_access(self, provider_id: str, user_id: int, required_role: str = "user") -> bool:
        """Check if a user has access to a provider with a specific role."""
        try:
            role = self.get_user_role_for_provider(provider_id, user_id)

            if not role:
                return False

            # Simple role hierarchy: admin > editor > user
            if required_role == "admin":
                return role == "admin"
            elif required_role == "editor":
                return role in ["admin", "editor"]
            else:
                return role in ["admin", "editor", "user"]
        except Exception as e:
            logger.error(f"Error checking user access: {e}")
            return False
