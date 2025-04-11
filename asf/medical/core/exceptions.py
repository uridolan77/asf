Exceptions for ASF Medical Research Synthesizer.

This module defines a comprehensive exception hierarchy for the ASF Medical Research
Synthesizer. It provides specialized exceptions for different types of errors that
can occur in the application.

from typing import Dict, Any, Optional


class ASFError(Exception):
    Base exception for all ASF-related errors.
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
    Legacy base exception for all ASF exceptions.
        super().__init__(message)


# Data access errors
class DataError(ASFError):
    Base exception for data-related errors.
    pass


class DatabaseError(DataError):
    Exception raised for database operation errors.
        super().__init__(message, details)


class RepositoryError(DataError):
    Exception raised for repository operation errors.
        super().__init__(message, details)


class NotFoundError(DataError):
    Exception raised when a requested resource is not found.
        message = f"{resource_type} with ID '{resource_id}' not found"
        super().__init__(message, details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class DuplicateError(DataError):
    Exception raised when attempting to create a duplicate resource.
        message = f"{resource_type} with ID '{resource_id}' already exists"
        super().__init__(message, details)
        self.resource_type = resource_type
        self.resource_id = resource_id


# Service errors
class ServiceError(ASFError):
    Base exception for service-related errors.
    pass


class ValidationError(ServiceError):
    Exception raised for validation errors.
        super().__init__(message, details)


# For backward compatibility
class ResourceNotFoundError(NotFoundError):
    Legacy exception raised when a resource is not found.
        super().__init__(resource_type, resource_id)


# For backward compatibility
class ResourceAlreadyExistsError(DuplicateError):
    Legacy exception raised when a resource already exists.
        super().__init__(resource_type, resource_id)


class BusinessLogicError(ServiceError):
    Exception raised for business logic violations.
        super().__init__(message, details)


class ExternalServiceError(ServiceError):
    Exception raised for errors in external service calls.
        full_message = f"{service_name}: {message}"
        super().__init__(full_message, details)
        self.service_name = service_name


# Authentication and authorization errors
class AuthError(ASFError):
    Base exception for authentication and authorization errors.
    pass


class AuthenticationError(AuthError):
    Exception raised for authentication failures.
        super().__init__(message, details)


class AuthorizationError(AuthError):
    Exception raised for authorization failures.
        super().__init__(message, details)


class TokenError(AuthError):
    Exception raised for token-related errors.
        super().__init__(message, details)


# ML-related errors
class MLError(ASFError):
    Base exception for machine learning related errors.
    pass


class ModelError(MLError):
    Exception raised for ML model errors.
        full_message = f"{model_name}: {message}"
        super().__init__(full_message, details)
        self.model_name = model_name


class ModelNotLoadedError(MLError):
    Exception raised when a required model is not loaded.
        full_message = f"{model_name}: {message}"
        super().__init__(full_message, details)
        self.model_name = model_name


class InferenceError(MLError):
    Exception raised during model inference.
        full_message = f"{model_name}: {message}"
        super().__init__(full_message, details)
        self.model_name = model_name


class ModelVersionError(MLError):
    Exception raised for model version incompatibilities.
        full_message = f"{model_name} (version {version}): {message}"
        super().__init__(full_message, details)
        self.model_name = model_name
        self.version = version


# Cache errors
class CacheError(ASFError):
    Base exception for cache-related errors.
        super().__init__(message, details)


class CacheConnectionError(CacheError):
    Exception raised for cache connection errors.
        super().__init__(message, details)


class CacheOperationError(CacheError):
    Exception raised for cache operation errors.
        full_message = f"{operation}: {message}"
        super().__init__(full_message, details)
        self.operation = operation


# Messaging errors
class MessagingError(ASFError):
    Base exception for messaging-related errors.
        super().__init__(message, details)


class ConnectionError(MessagingError):
    Exception raised for connection errors.
        full_message = f"{service}: {message}"
        super().__init__(full_message, details)
        self.service = service


class MessageBrokerError(MessagingError):
    Exception raised for message broker errors.
        super().__init__(message, details)


class MessageError(MessagingError):
    Exception raised for message-related errors.
        full_message = f"{operation}: {message}"
        super().__init__(full_message, details)
        self.operation = operation


# Configuration errors
class ConfigurationError(ASFError):
    Exception raised for configuration errors.
        super().__init__(message, details)


# File system errors
class FileSystemError(ASFError):
    Base exception for file system errors.
    pass


class FileError(FileSystemError):
    Exception raised for file-related errors.
        full_message = f"{file_path}: {message}"
        super().__init__(full_message, details)
        self.file_path = file_path


class FileNotFoundError(FileSystemError):
    Exception raised when a file is not found.
        full_message = f"{file_path}: {message}"
        super().__init__(full_message, details)
        self.file_path = file_path


class FileAccessError(FileSystemError):
    Exception raised for file access errors.
        full_message = f"{file_path}: {message}"
        super().__init__(full_message, details)
        self.file_path = file_path


# API errors
class APIError(ASFError):
    Base exception for API-related errors.
    pass


class RateLimitError(APIError):
    Exception raised when rate limits are exceeded.
        super().__init__(message, details)


class InvalidRequestError(APIError):
    Exception raised for invalid API requests.
        super().__init__(message, details)


# Task errors
class TaskError(ASFError):
    Base exception for task-related errors.
    pass


class TaskExecutionError(TaskError):
    Exception raised for task execution errors.
        full_message = f"Task {task_id}: {message}"
        super().__init__(full_message, details)
        self.task_id = task_id


class TaskTimeoutError(TaskError):
    Exception raised when a task times out.
        full_message = f"Task {task_id} timed out after {timeout} seconds: {message}"
        super().__init__(full_message, details)
        self.task_id = task_id
        self.timeout = timeout


# Domain-specific errors
class ExportError(ServiceError):
    Exception raised for export errors.
        full_message = f"{format}: {message}"
        super().__init__(full_message, details)
        self.format = format


class SearchError(ServiceError):
    Exception raised for search errors.
        full_message = f"Query '{query}': {message}"
        super().__init__(full_message, details)
        self.query = query


class AnalysisError(ServiceError):
    Exception raised for analysis errors.
        full_message = f"{analysis_type}: {message}"
        super().__init__(full_message, details)
        self.analysis_type = analysis_type


class KnowledgeBaseError(ServiceError):
    Exception raised for knowledge base errors.
        full_message = f"{kb_name}: {message}"
        super().__init__(full_message, details)
        self.kb_name = kb_name


class ContradictionError(ServiceError):
    Exception raised for contradiction analysis errors.
        super().__init__(message, details)


class OperationError(ServiceError):
    Exception raised for general operation errors.
        super().__init__(message, details)
