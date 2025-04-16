from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import datetime

Base = declarative_base()

class Role(Base):
    __tablename__ = 'roles'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(String(255))
    users = relationship('User', back_populates='role')

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    role_id = Column(Integer, ForeignKey('roles.id'))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    role = relationship('Role', back_populates='users')

    # New relationships for API keys and configuration
    created_providers = relationship("Provider", foreign_keys="Provider.created_by_user_id", back_populates="created_by")
    api_keys = relationship("ApiKey", foreign_keys="ApiKey.created_by_user_id", back_populates="created_by")
    configurations = relationship("Configuration", foreign_keys="Configuration.created_by_user_id", back_populates="created_by")
    settings = relationship("UserSetting", back_populates="user")
    audit_logs = relationship("AuditLog", foreign_keys="AuditLog.changed_by_user_id", back_populates="changed_by")