"""
Task models for the Medical Research Synthesizer.
This module defines SQLAlchemy models for tasks and task status.
"""
import enum
from datetime import datetime
from typing import Dict, Any
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Text, Boolean, JSON, Enum
from sqlalchemy.orm import relationship
from ..database import MedicalBase  # Updated from Base to MedicalBase

class TaskStatus(str, enum.Enum):
    """Task status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"

class TaskPriority(int, enum.Enum):
    """Task priority enum."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10

class Task(MedicalBase):  # Updated from Base to MedicalBase
    """Task model for tracking asynchronous tasks."""
    __tablename__ = "tasks"
    id = Column(String(36), primary_key=True, index=True)
    type = Column(String(100), nullable=False, index=True)
    status = Column(Enum(TaskStatus), nullable=False, default=TaskStatus.PENDING, index=True)
    priority = Column(Integer, nullable=False, default=TaskPriority.NORMAL)
    user_id = Column(Integer, ForeignKey("medical_users.id"), nullable=True, index=True)  # Updated from "users.id"
    user = relationship("MedicalUser", back_populates="tasks")  # Updated from "User"
    params = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    progress = Column(Float, nullable=False, default=0.0)
    message = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    next_retry_at = Column(DateTime, nullable=True)
    # For long-running tasks, we may want to store the worker ID
    worker_id = Column(String(100), nullable=True)
    # For tasks that can be cancelled
    cancellable = Column(Boolean, nullable=False, default=True)
    cancelled = Column(Boolean, nullable=False, default=False)
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task to a dictionary.
        
        Args:
        
        
        Returns:
            Description of return value
        """
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status.value if self.status else None,
            "priority": self.priority,
            "user_id": self.user_id,
            "params": self.params,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None,
            "worker_id": self.worker_id,
            "cancellable": self.cancellable,
            "cancelled": self.cancelled
        }

class TaskEvent(MedicalBase):  # Updated from Base to MedicalBase
    """Task event model for tracking task lifecycle events."""
    __tablename__ = "task_events"
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=False, index=True)
    task = relationship("Task")
    event_type = Column(String(50), nullable=False, index=True)
    event_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task event to a dictionary.
        
        Args:
        
        
        Returns:
            Description of return value
        """
        return {
            "id": self.id,
            "task_id": self.task_id,
            "event_type": self.event_type,
            "event_data": self.event_data,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class DeadLetterMessage(MedicalBase):  # Updated from Base to MedicalBase
    """Dead letter message model for tracking failed messages."""
    __tablename__ = "dead_letter_messages"
    id = Column(Integer, primary_key=True, index=True)
    original_id = Column(String(100), nullable=True, index=True)
    exchange = Column(String(100), nullable=False)
    routing_key = Column(String(100), nullable=False)
    message = Column(JSON, nullable=False)
    headers = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    # For message reprocessing
    reprocessed = Column(Boolean, nullable=False, default=False)
    reprocessed_at = Column(DateTime, nullable=True)
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the dead letter message to a dictionary.
        
        Args:
        
        
        Returns:
            Description of return value
        """
        return {
            "id": self.id,
            "original_id": self.original_id,
            "exchange": self.exchange,
            "routing_key": self.routing_key,
            "message": self.message,
            "headers": self.headers,
            "error": self.error,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "reprocessed": self.reprocessed,
            "reprocessed_at": self.reprocessed_at.isoformat() if self.reprocessed_at else None
        }