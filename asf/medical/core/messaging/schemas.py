"""
Messaging module for the Medical Research Synthesizer.

This module provides schemas and utilities for managing messaging protocols
and message validation.
"""
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, validator

class MessageType(str, Enum):
    """
    Message types.
    
    Defines the basic types of messages that can be exchanged in the system.
    """
    EVENT = "event"
    TASK = "task"
    COMMAND = "command"
    REPLY = "reply"

class EventType(str, Enum):
    """
    Event types.
    
    Defines the various event types organized by category.
    """
    # User events
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    # Search events
    SEARCH_PERFORMED = "search.performed"
    SEARCH_COMPLETED = "search.completed"
    # Analysis events
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"
    # Task events
    TASK_STARTED = "task.started"
    TASK_PROGRESS = "task.progress"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"

class TaskType(str, Enum):
    """
    Task types.
    
    Defines the various task types organized by category.
    """
    # Search tasks
    SEARCH_PUBMED = "search.pubmed"
    SEARCH_CLINICAL_TRIALS = "search.clinical_trials"
    SEARCH_KNOWLEDGE_BASE = "search.knowledge_base"
    # Analysis tasks
    ANALYZE_CONTRADICTIONS = "analysis.contradictions"
    ANALYZE_BIAS = "analysis.bias"
    ANALYZE_TRENDS = "analysis.trends"
    # Export tasks
    EXPORT_RESULTS = "export.results"
    EXPORT_ANALYSIS = "export.analysis"
    # Import tasks
    IMPORT_STUDIES = "import.studies"
    IMPORT_KNOWLEDGE = "import.knowledge"
    # ML tasks
    TRAIN_MODEL = "ml.train"
    EVALUATE_MODEL = "ml.evaluate"
    PREDICT = "ml.predict"

class CommandType(str, Enum):
    """
    Command types.
    
    Defines the various command types organized by category.
    """
    # User commands
    CREATE_USER = "user.create"
    UPDATE_USER = "user.update"
    DELETE_USER = "user.delete"
    # Search commands
    PERFORM_SEARCH = "search.perform"
    CANCEL_SEARCH = "search.cancel"
    # Analysis commands
    START_ANALYSIS = "analysis.start"
    CANCEL_ANALYSIS = "analysis.cancel"
    # Task commands
    START_TASK = "task.start"
    CANCEL_TASK = "task.cancel"
    # System commands
    SHUTDOWN = "system.shutdown"
    RESTART = "system.restart"

class BaseMessage(BaseModel):
    """
    Base class for all messages.
    
    Provides common fields and functionality for all message types.
    """
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @validator("timestamp", pre=True)
    def parse_timestamp(cls, value):
        """
        Parse timestamp from string if needed.
        
        Args:
            cls: Class reference
            value: Timestamp value to parse
            
        Returns:
            Parsed datetime object or original value if already a datetime
        """
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return value

class Event(BaseMessage):
    """
    Event message.
    
    Represents an event notification in the system.
    """
    type: str
    data: Dict[str, Any] = Field(default_factory=dict)

class Task(BaseMessage):
    """
    Task message.
    
    Represents a background task with progress tracking and result handling.
    """
    id: str
    type: str
    data: Dict[str, Any] = Field(default_factory=dict)
    status: Optional[str] = None
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class Command(BaseMessage):
    """
    Command message.
    
    Represents a command to be executed by a specific target service.
    """
    id: str
    type: str
    target: str
    data: Dict[str, Any] = Field(default_factory=dict)

class Reply(BaseMessage):
    """
    Reply message.
    
    Represents a response to a command or request.
    """
    id: str
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    success: bool = True

# User events
class UserCreatedEvent(Event):
    """
    Event fired when a user is created.
    """
    type: str = EventType.USER_CREATED
    data: Dict[str, Any]

class UserUpdatedEvent(Event):
    """
    Event fired when a user is updated.
    """
    type: str = EventType.USER_UPDATED
    data: Dict[str, Any]

class UserDeletedEvent(Event):
    """
    Event fired when a user is deleted.
    """
    type: str = EventType.USER_DELETED
    data: Dict[str, Any]

# Search events
class SearchPerformedEvent(Event):
    """
    Event fired when a search is performed.
    """
    type: str = EventType.SEARCH_PERFORMED
    data: Dict[str, Any]

class SearchCompletedEvent(Event):
    """
    Event fired when a search is completed.
    """
    type: str = EventType.SEARCH_COMPLETED
    data: Dict[str, Any]

# Analysis events
class AnalysisStartedEvent(Event):
    """
    Event fired when an analysis is started.
    """
    type: str = EventType.ANALYSIS_STARTED
    data: Dict[str, Any]

class AnalysisCompletedEvent(Event):
    """
    Event fired when an analysis is completed.
    """
    type: str = EventType.ANALYSIS_COMPLETED
    data: Dict[str, Any]

class AnalysisFailedEvent(Event):
    """
    Event fired when an analysis fails.
    """
    type: str = EventType.ANALYSIS_FAILED
    data: Dict[str, Any]

# Task events
class TaskStartedEvent(Event):
    """
    Event fired when a task is started.
    """
    type: str = EventType.TASK_STARTED
    data: Dict[str, Any]

class TaskProgressEvent(Event):
    """
    Event fired when a task makes progress.
    """
    type: str = EventType.TASK_PROGRESS
    data: Dict[str, Any]

class TaskCompletedEvent(Event):
    """
    Event fired when a task is completed.
    """
    type: str = EventType.TASK_COMPLETED
    data: Dict[str, Any]

class TaskFailedEvent(Event):
    """
    Event fired when a task fails.
    """
    type: str = EventType.TASK_FAILED
    data: Dict[str, Any]

# Search tasks
class SearchPubMedTask(Task):
    """
    Task for searching PubMed.
    """
    type: str = TaskType.SEARCH_PUBMED
    data: Dict[str, Any]

class SearchClinicalTrialsTask(Task):
    """
    Task for searching ClinicalTrials.gov.
    """
    type: str = TaskType.SEARCH_CLINICAL_TRIALS
    data: Dict[str, Any]

class SearchKnowledgeBaseTask(Task):
    """
    Task for searching the knowledge base.
    """
    type: str = TaskType.SEARCH_KNOWLEDGE_BASE
    data: Dict[str, Any]

# Analysis tasks
class AnalyzeContradictionsTask(Task):
    """
    Task for analyzing contradictions.
    """
    type: str = TaskType.ANALYZE_CONTRADICTIONS
    data: Dict[str, Any]

class AnalyzeBiasTask(Task):
    """
    Task for analyzing bias.
    """
    type: str = TaskType.ANALYZE_BIAS
    data: Dict[str, Any]

class AnalyzeTrendsTask(Task):
    """
    Task for analyzing trends.
    """
    type: str = TaskType.ANALYZE_TRENDS
    data: Dict[str, Any]

# Export tasks
class ExportResultsTask(Task):
    """
    Task for exporting results.
    """
    type: str = TaskType.EXPORT_RESULTS
    data: Dict[str, Any]

class ExportAnalysisTask(Task):
    """
    Task for exporting analysis.
    """
    type: str = TaskType.EXPORT_ANALYSIS
    data: Dict[str, Any]

# User commands
class CreateUserCommand(Command):
    """
    Command for creating a user.
    """
    type: str = CommandType.CREATE_USER
    target: str = "user_service"
    data: Dict[str, Any]

class UpdateUserCommand(Command):
    """
    Command for updating a user.
    """
    type: str = CommandType.UPDATE_USER
    target: str = "user_service"
    data: Dict[str, Any]

class DeleteUserCommand(Command):
    """
    Command for deleting a user.
    """
    type: str = CommandType.DELETE_USER
    target: str = "user_service"
    data: Dict[str, Any]

# Search commands
class PerformSearchCommand(Command):
    """
    Command for performing a search.
    """
    type: str = CommandType.PERFORM_SEARCH
    target: str = "search_service"
    data: Dict[str, Any]

class CancelSearchCommand(Command):
    """
    Command for canceling a search.
    """
    type: str = CommandType.CANCEL_SEARCH
    target: str = "search_service"
    data: Dict[str, Any]

# Analysis commands
class StartAnalysisCommand(Command):
    """
    Command for starting an analysis.
    """
    type: str = CommandType.START_ANALYSIS
    target: str = "analysis_service"
    data: Dict[str, Any]

class CancelAnalysisCommand(Command):
    """
    Command for canceling an analysis.
    """
    type: str = CommandType.CANCEL_ANALYSIS
    target: str = "analysis_service"
    data: Dict[str, Any]