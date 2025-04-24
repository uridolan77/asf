"""
DSPy Module Models

This module provides database models for DSPy modules, parameters, executions, and related entities.
"""

import uuid
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, JSON, Enum, Text, Table
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from typing import Dict, Any, List, Optional

from ..db.database import Base


class ModuleParameterType(str, enum.Enum):
    """Enumeration of parameter types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ModuleExecutionStatus(str, enum.Enum):
    """Enumeration of execution status values"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class DSPyModuleParameter(Base):
    """DSPy module parameter model"""
    __tablename__ = "dspy_module_parameters"

    parameter_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    module_id = Column(String, ForeignKey("dspy_modules.module_id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    type = Column(String, default=ModuleParameterType.STRING)
    required = Column(Boolean, default=False)
    default_value = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Additional parameter metadata
    position = Column(Integer, nullable=True)  # For ordered parameters
    enum_values = Column(JSON, nullable=True)  # For parameters with enumerated values
    min_value = Column(Float, nullable=True)   # For numeric parameters
    max_value = Column(Float, nullable=True)   # For numeric parameters
    pattern = Column(String, nullable=True)    # Regex pattern for string validation
    
    # Relationships
    module = relationship("DSPyModule", backref="parameters")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return {
            "parameter_id": self.parameter_id,
            "module_id": self.module_id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "required": self.required,
            "default_value": self.default_value,
            "position": self.position,
            "enum_values": self.enum_values,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "pattern": self.pattern,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class DSPyModuleExecution(Base):
    """DSPy module execution model"""
    __tablename__ = "dspy_module_executions"

    execution_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    module_id = Column(String, ForeignKey("dspy_modules.module_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    status = Column(String, default=ModuleExecutionStatus.PENDING)
    inputs = Column(JSON, nullable=True)
    outputs = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)
    started_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime, nullable=True)
    
    # Request tracking
    request_id = Column(String, nullable=True)
    correlation_id = Column(String, nullable=True)
    
    # Performance metrics
    tokens_input = Column(Integer, nullable=True)
    tokens_output = Column(Integer, nullable=True)
    cache_hit = Column(Boolean, default=False)
    
    # Additional metadata
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    module = relationship("DSPyModule")
    user = relationship("User")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return {
            "execution_id": self.execution_id,
            "module_id": self.module_id,
            "user_id": self.user_id,
            "status": self.status,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "cache_hit": self.cache_hit,
            "metadata": self.metadata
        }


class DSPyModuleVersion(Base):
    """DSPy module version model"""
    __tablename__ = "dspy_module_versions"

    version_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    module_id = Column(String, ForeignKey("dspy_modules.module_id", ondelete="CASCADE"), nullable=False)
    version = Column(String, nullable=False)
    configuration = Column(JSON, nullable=True)  # Module configuration for this version
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    created_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Optimization metadata
    optimization_metric = Column(String, nullable=True)
    optimization_score = Column(Float, nullable=True)
    optimization_details = Column(JSON, nullable=True)
    
    # Relationships
    module = relationship("DSPyModule")
    creator = relationship("User")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return {
            "version_id": self.version_id,
            "module_id": self.module_id,
            "version": self.version,
            "configuration": self.configuration,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
            "optimization_metric": self.optimization_metric,
            "optimization_score": self.optimization_score,
            "optimization_details": self.optimization_details
        }


class DSPyModuleFeedback(Base):
    """DSPy module feedback model"""
    __tablename__ = "dspy_module_feedback"

    feedback_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    execution_id = Column(String, ForeignKey("dspy_module_executions.execution_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    rating = Column(Integer, nullable=True)  # 1-5 rating
    feedback_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    execution = relationship("DSPyModuleExecution")
    user = relationship("User")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return {
            "feedback_id": self.feedback_id,
            "execution_id": self.execution_id,
            "user_id": self.user_id,
            "rating": self.rating,
            "feedback_text": self.feedback_text,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class DSPyModuleExample(Base):
    """DSPy module example model for optimization and testing"""
    __tablename__ = "dspy_module_examples"

    example_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    module_id = Column(String, ForeignKey("dspy_modules.module_id", ondelete="CASCADE"), nullable=False)
    inputs = Column(JSON, nullable=True)
    expected_outputs = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=True)  # For categorizing examples
    created_at = Column(DateTime, default=datetime.now)
    created_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Relationships
    module = relationship("DSPyModule")
    creator = relationship("User")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return {
            "example_id": self.example_id,
            "module_id": self.module_id,
            "inputs": self.inputs,
            "expected_outputs": self.expected_outputs,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by
        }