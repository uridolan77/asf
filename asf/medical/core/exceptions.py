"""
Exceptions for ASF Medical Research Synthesizer.

This module defines a comprehensive exception hierarchy for the ASF Medical Research
Synthesizer. It provides specialized exceptions for different types of errors that
can occur in the application.
"""

from typing import Dict, Any, Optional


class ASFError(Exception):
    """Base exception for all ASF-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)


# For backward compatibility
class ASFException(ASFError):
    """Legacy base exception for all ASF exceptions."""

    def __init__(self, message: str = "An error occurred"):
        super().__init__(message)


# Data access errors
class DataError(ASFError):
    """Base exception for data-related errors."""
    pass


class DatabaseError(DataError):
    """Exception raised for database operation errors."""

    def __init__(self, message: str = "Database error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


class RepositoryError(DataError):
    """Exception raised for repository operation errors."""

    def __init__(self, message: str = "Repository error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


class NotFoundError(DataError):
    """Exception raised when a requested resource is not found."""

    def __init__(self, resource_type: str, resource_id: str, details: Optional[Dict[str, Any]] = None):
        message = f"{resource_type} with ID '{resource_id}' not found"
        super().__init__(message, details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class DuplicateError(DataError):
    """Exception raised when attempting to create a duplicate resource."""

    def __init__(self, resource_type: str, resource_id: str, details: Optional[Dict[str, Any]] = None):
        message = f"{resource_type} with ID '{resource_id}' already exists"
        super().__init__(message, details)
        self.resource_type = resource_type
        self.resource_id = resource_id


# Service errors
class ServiceError(ASFError):
    """Base exception for service-related errors."""
    pass


class ValidationError(ServiceError):
    """Exception raised for validation errors."""

    def __init__(self, message: str = "Validation error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


# For backward compatibility
class ResourceNotFoundError(NotFoundError):
    """Legacy exception raised when a resource is not found."""

    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(resource_type, resource_id)


# For backward compatibility
class ResourceAlreadyExistsError(DuplicateError):
    """Legacy exception raised when a resource already exists."""

    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(resource_type, resource_id)


class BusinessLogicError(ServiceError):
    """Exception raised for business logic violations."""

    def __init__(self, message: str = "Business logic error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


class ExternalServiceError(ServiceError):
    """Exception raised for errors in external service calls."""

    def __init__(self, service_name: str, message: str = "External service error", details: Optional[Dict[str, Any]] = None):
        full_message = f"{service_name}: {message}"
        super().__init__(full_message, details)
        self.service_name = service_name


# Authentication and authorization errors
class AuthError(ASFError):
    """Base exception for authentication and authorization errors."""
    pass


class AuthenticationError(AuthError):
    """Exception raised for authentication failures."""

    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


class AuthorizationError(AuthError):
    """Exception raised for authorization failures."""

    def __init__(self, message: str = "Not authorized to perform this action", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


class TokenError(AuthError):
    """Exception raised for token-related errors."""

    def __init__(self, message: str = "Token error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


# ML-related errors
class MLError(ASFError):
    """Base exception for machine learning related errors."""
    pass


class ModelError(MLError):
    """Exception raised for ML model errors."""

    def __init__(self, model_name: str, message: str = "Model error", details: Optional[Dict[str, Any]] = None):
        full_message = f"{model_name}: {message}"
        super().__init__(full_message, details)
        self.model_name = model_name


class ModelNotLoadedError(MLError):
    """Exception raised when a required model is not loaded."""

    def __init__(self, model_name: str, message: str = "Model not loaded", details: Optional[Dict[str, Any]] = None):
        full_message = f"{model_name}: {message}"
        super().__init__(full_message, details)
        self.model_name = model_name


class InferenceError(MLError):
    """Exception raised during model inference."""

    def __init__(self, model_name: str, message: str = "Inference error", details: Optional[Dict[str, Any]] = None):
        full_message = f"{model_name}: {message}"
        super().__init__(full_message, details)
        self.model_name = model_name


class ModelVersionError(MLError):
    """Exception raised for model version incompatibilities."""

    def __init__(self, model_name: str, version: str, message: str = "Model version error", details: Optional[Dict[str, Any]] = None):
        full_message = f"{model_name} (version {version}): {message}"
        super().__init__(full_message, details)
        self.model_name = model_name
        self.version = version


# Cache errors
class CacheError(ASFError):
    """Base exception for cache-related errors."""

    def __init__(self, message: str = "Cache error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


class CacheConnectionError(CacheError):
    """Exception raised for cache connection errors."""

    def __init__(self, message: str = "Cache connection error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


class CacheOperationError(CacheError):
    """Exception raised for cache operation errors."""

    def __init__(self, operation: str, message: str = "Cache operation error", details: Optional[Dict[str, Any]] = None):
        full_message = f"{operation}: {message}"
        super().__init__(full_message, details)
        self.operation = operation


# Messaging errors
class MessagingError(ASFError):
    """Base exception for messaging-related errors."""

    def __init__(self, message: str = "Messaging error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


class ConnectionError(MessagingError):
    """Exception raised for connection errors."""

    def __init__(self, service: str, message: str = "Connection error", details: Optional[Dict[str, Any]] = None):
        full_message = f"{service}: {message}"
        super().__init__(full_message, details)
        self.service = service


class MessageBrokerError(MessagingError):
    """Exception raised for message broker errors."""

    def __init__(self, message: str = "Message broker error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


class MessageError(MessagingError):
    """Exception raised for message-related errors."""

    def __init__(self, operation: str, message: str = "Message error", details: Optional[Dict[str, Any]] = None):
        full_message = f"{operation}: {message}"
        super().__init__(full_message, details)
        self.operation = operation


# Configuration errors
class ConfigurationError(ASFError):
    """Exception raised for configuration errors."""

    def __init__(self, message: str = "Configuration error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


# File system errors
class FileSystemError(ASFError):
    """Base exception for file system errors."""
    pass


class FileError(FileSystemError):
    """Exception raised for file-related errors."""

    def __init__(self, file_path: str, message: str = "File error", details: Optional[Dict[str, Any]] = None):
        full_message = f"{file_path}: {message}"
        super().__init__(full_message, details)
        self.file_path = file_path


class FileNotFoundError(FileSystemError):
    """Exception raised when a file is not found."""

    def __init__(self, file_path: str, message: str = "File not found", details: Optional[Dict[str, Any]] = None):
        full_message = f"{file_path}: {message}"
        super().__init__(full_message, details)
        self.file_path = file_path


class FileAccessError(FileSystemError):
    """Exception raised for file access errors."""

    def __init__(self, file_path: str, message: str = "File access error", details: Optional[Dict[str, Any]] = None):
        full_message = f"{file_path}: {message}"
        super().__init__(full_message, details)
        self.file_path = file_path


# API errors
class APIError(ASFError):
    """Base exception for API-related errors."""
    pass


class RateLimitError(APIError):
    """Exception raised when rate limits are exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


class InvalidRequestError(APIError):
    """Exception raised for invalid API requests."""

    def __init__(self, message: str = "Invalid request", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


# Task errors
class TaskError(ASFError):
    """Base exception for task-related errors."""
    pass


class TaskExecutionError(TaskError):
    """Exception raised for task execution errors."""

    def __init__(self, task_id: str, message: str = "Task execution error", details: Optional[Dict[str, Any]] = None):
        full_message = f"Task {task_id}: {message}"
        super().__init__(full_message, details)
        self.task_id = task_id


class TaskTimeoutError(TaskError):
    """Exception raised when a task times out."""

    def __init__(self, task_id: str, timeout: int, message: str = "Task timed out", details: Optional[Dict[str, Any]] = None):
        full_message = f"Task {task_id} timed out after {timeout} seconds: {message}"
        super().__init__(full_message, details)
        self.task_id = task_id
        self.timeout = timeout


# Domain-specific errors
class ExportError(ServiceError):
    """Exception raised for export errors."""

    def __init__(self, format: str, message: str = "Export error", details: Optional[Dict[str, Any]] = None):
        full_message = f"{format}: {message}"
        super().__init__(full_message, details)
        self.format = format


class SearchError(ServiceError):
    """Exception raised for search errors."""

    def __init__(self, query: str, message: str = "Search error", details: Optional[Dict[str, Any]] = None):
        full_message = f"Query '{query}': {message}"
        super().__init__(full_message, details)
        self.query = query


class AnalysisError(ServiceError):
    """Exception raised for analysis errors."""

    def __init__(self, analysis_type: str, message: str = "Analysis error", details: Optional[Dict[str, Any]] = None):
        full_message = f"{analysis_type}: {message}"
        super().__init__(full_message, details)
        self.analysis_type = analysis_type


class KnowledgeBaseError(ServiceError):
    """Exception raised for knowledge base errors."""

    def __init__(self, kb_name: str, message: str = "Knowledge base error", details: Optional[Dict[str, Any]] = None):
        full_message = f"{kb_name}: {message}"
        super().__init__(full_message, details)
        self.kb_name = kb_name


class ContradictionError(ServiceError):
    """Exception raised for contradiction analysis errors."""

    def __init__(self, message: str = "Contradiction analysis error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
