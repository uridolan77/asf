from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Table
from sqlalchemy.sql import func
from asf.bollm.backend.models.base import Base

# Association table for many-to-many relationship between users and providers
users_providers = Table(
    'users_providers',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('provider_id', String(50), ForeignKey('llm_providers.provider_id'), primary_key=True),
    Column('role', String(50), default='user'),
    Column('created_at', DateTime, default=func.now()),
    extend_existing=True  # Add this to handle duplicate table definitions
)
