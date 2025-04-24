from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, Boolean, UniqueConstraint, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from asf.bollm.backend.models.base import Base

class LLMModel(Base):
    """
    LLM Model model for storing LLM model configurations.

    This model stores information about LLM models, including their capabilities,
    parameters, and configuration.
    """
    __tablename__ = "llm_models"

    # Primary keys
    model_id = Column(String(255), primary_key=True)
    provider_id = Column(String(255), ForeignKey("llm_providers.provider_id"), nullable=False, primary_key=True)
    
    # Model attributes
    display_name = Column(String(255), nullable=False)
    model_type = Column(String(50), default="chat")  # chat, completion, embedding, etc.
    context_window = Column(Integer)
    max_tokens = Column(Integer)  # Using max_tokens for consistency with ProviderModel
    max_output_tokens = Column(Integer)  # Keeping this for backward compatibility
    enabled = Column(Boolean, default=True)
    capabilities = Column(JSON)
    parameters = Column(JSON)

    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationship with Provider
    provider = relationship(
        "Provider",
        back_populates="models",
        foreign_keys=[provider_id]
    )

    # Constraints - only need extend_existing now since primary keys define uniqueness
    __table_args__ = {
        'extend_existing': True
    }

    def to_dict(self):
        """Convert the model to a dictionary for API responses."""
        return {
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "display_name": self.display_name,
            "model_type": self.model_type,
            "context_window": self.context_window,
            "max_tokens": self.max_tokens,
            "max_output_tokens": self.max_output_tokens,
            "enabled": self.enabled,
            "capabilities": self.capabilities or [],
            "parameters": self.parameters or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
