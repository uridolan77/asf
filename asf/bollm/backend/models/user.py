from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from asf.bollm.backend.models.base import Base
from asf.bollm.backend.models.association import users_providers  # Import the association table

class Role(Base):
    __tablename__ = 'roles'

    # Define table arguments with extend_existing=True
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(String(255))
    users = relationship('asf.bollm.backend.models.user.BOLLMUser', back_populates='role')  # Use fully qualified path

class BOLLMUser(Base):  # Renamed from User to BOLLMUser
    __tablename__ = 'users'

    # Define table arguments with extend_existing=True
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    role_id = Column(Integer, ForeignKey('roles.id'))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    role = relationship('asf.bollm.backend.models.user.Role', back_populates='users')

    # Many-to-many relationship with providers
    providers = relationship("asf.bollm.backend.models.provider.Provider", secondary=users_providers, back_populates="users")

    # Other relationships
    api_keys = relationship("asf.bollm.backend.models.provider.ApiKey", foreign_keys="[asf.bollm.backend.models.provider.ApiKey.created_by_user_id]", back_populates="created_by", lazy="dynamic")
    configurations = relationship("asf.bollm.backend.models.configuration.Configuration", foreign_keys="[asf.bollm.backend.models.configuration.Configuration.created_by_user_id]", back_populates="created_by", lazy="dynamic")
    settings = relationship("asf.bollm.backend.models.configuration.UserSetting", back_populates="user", lazy="dynamic")
    audit_logs = relationship("asf.bollm.backend.models.audit.AuditLog", foreign_keys="[asf.bollm.backend.models.audit.AuditLog.changed_by_user_id]", back_populates="changed_by", lazy="dynamic")

# Add backward compatibility alias to maintain compatibility with existing imports
User = BOLLMUser
