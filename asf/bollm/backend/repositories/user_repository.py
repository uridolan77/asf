from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import logging
from asf.bollm.backend.models.user import User, Role
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all_users(self) -> List[User]:
        """Get all users."""
        return self.db.query(User).all()

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get a user by ID."""
        return self.db.query(User).filter(User.id == user_id).first()

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        return self.db.query(User).filter(User.email == email).first()

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        return self.db.query(User).filter(User.username == username).first()

    def authenticate_user(self, email: str, password: str, pwd_context: CryptContext) -> Optional[User]:
        """Authenticate a user with email and password."""
        user = self.get_user_by_email(email)
        if not user:
            return None
        if not pwd_context.verify(password, user.password_hash):
            return None
        return user

    def create_user(self, user_data: Dict[str, Any], pwd_context: CryptContext) -> User:
        """Create a new user."""
        try:
            # Hash the password
            hashed_password = pwd_context.hash(user_data["password"])
            
            # Create the user
            user = User(
                username=user_data["username"],
                email=user_data["email"],
                password_hash=hashed_password,
                role_id=user_data["role_id"]
            )
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            return user
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating user: {e}")
            raise

    def update_user(self, user_id: int, user_data: Dict[str, Any], pwd_context: CryptContext = None) -> Optional[User]:
        """Update a user."""
        try:
            user = self.get_user_by_id(user_id)
            if not user:
                return None

            # Update fields
            if "username" in user_data:
                user.username = user_data["username"]
            if "email" in user_data:
                user.email = user_data["email"]
            if "role_id" in user_data:
                user.role_id = user_data["role_id"]
            
            # Update password if provided
            if "password" in user_data and pwd_context:
                user.password_hash = pwd_context.hash(user_data["password"])

            self.db.commit()
            self.db.refresh(user)
            return user
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating user: {e}")
            raise

    def delete_user(self, user_id: int) -> bool:
        """Delete a user."""
        try:
            user = self.get_user_by_id(user_id)
            if not user:
                return False

            self.db.delete(user)
            self.db.commit()
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error deleting user: {e}")
            raise

    def get_all_roles(self) -> List[Role]:
        """Get all roles."""
        return self.db.query(Role).all()

    def get_role_by_id(self, role_id: int) -> Optional[Role]:
        """Get a role by ID."""
        return self.db.query(Role).filter(Role.id == role_id).first()

    def create_role(self, role_data: Dict[str, Any]) -> Role:
        """Create a new role."""
        try:
            role = Role(
                name=role_data["name"],
                description=role_data.get("description")
            )
            self.db.add(role)
            self.db.commit()
            self.db.refresh(role)
            return role
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating role: {e}")
            raise

    def update_role(self, role_id: int, role_data: Dict[str, Any]) -> Optional[Role]:
        """Update a role."""
        try:
            role = self.get_role_by_id(role_id)
            if not role:
                return None

            if "name" in role_data:
                role.name = role_data["name"]
            if "description" in role_data:
                role.description = role_data["description"]

            self.db.commit()
            self.db.refresh(role)
            return role
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating role: {e}")
            raise

    def delete_role(self, role_id: int) -> bool:
        """Delete a role."""
        try:
            role = self.get_role_by_id(role_id)
            if not role:
                return False

            self.db.delete(role)
            self.db.commit()
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error deleting role: {e}")
            raise
