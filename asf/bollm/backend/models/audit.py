from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, JSON, Date, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from asf.bollm.backend.models.base import Base
from asf.bollm.backend.models.user import BOLLMUser  # Import the renamed BOLLMUser class
from asf.bollm.backend.models.provider import ApiKey  # Import the ApiKey class explicitly

class AuditLog(Base):
    __tablename__ = "audit_logs"

    # Define table arguments with extend_existing=True
    __table_args__ = {'extend_existing': True}

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    table_name = Column(String(50), nullable=False)
    record_id = Column(String(50), nullable=False)
    action = Column(String(20), nullable=False)
    changed_by_user_id = Column(Integer, ForeignKey("users.id"))
    changed_at = Column(DateTime, default=func.now())
    old_values = Column(JSON)
    new_values = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(Text)

    # Relationships with fully qualified paths
    changed_by = relationship("asf.bollm.backend.models.user.BOLLMUser", foreign_keys=[changed_by_user_id])

class ApiKeyUsage(Base):
    __tablename__ = "api_key_usage"

    # Define table arguments with extend_existing=True
    __table_args__ = (
        UniqueConstraint('key_id', 'user_id', 'usage_date', name='uix_api_key_usage'),
        {'extend_existing': True}
    )

    usage_id = Column(Integer, primary_key=True, autoincrement=True)
    key_id = Column(Integer, ForeignKey("llm_provider_api_keys.key_id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))
    request_count = Column(Integer, default=1)
    tokens_used = Column(Integer, default=0)
    usage_date = Column(Date, default=func.current_date())
    created_at = Column(DateTime, default=func.now())

    # Relationships with fully qualified paths
    api_key = relationship("asf.bollm.backend.models.provider.ApiKey", back_populates="usage_records")
    user = relationship("asf.bollm.backend.models.user.BOLLMUser")
