from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .base import Base

class Configuration(Base):
    __tablename__ = "configurations"
    
    config_id = Column(Integer, primary_key=True, autoincrement=True)
    config_key = Column(String(100), nullable=False)
    config_value = Column(Text)
    config_type = Column(String(20), default="string")
    description = Column(Text)
    environment = Column(String(20), default="development")
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by_user_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    created_by = relationship("User", foreign_keys=[created_by_user_id])
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('config_key', 'environment', name='uix_config_key_env'),
    )

class UserSetting(Base):
    __tablename__ = "user_settings"
    
    setting_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    setting_key = Column(String(100), nullable=False)
    setting_value = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="settings")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'setting_key', name='uix_user_setting'),
    )
