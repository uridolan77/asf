"""
Exceptions module for the Medical Research Synthesizer.

This module defines custom exceptions used throughout the application to
provide consistent error handling and reporting.

Classes:
    MedicalResearchSynthesizerError: Base class for all custom exceptions.
    ValidationError: Exception for validation errors.
    DatabaseError: Exception for database-related errors.
    AuthenticationError: Exception for authentication-related errors.
    AuthorizationError: Exception for authorization-related errors.
    APIError: Exception for API-related errors.
    CacheError: Exception for cache-related errors.
    TaskError: Exception for task-related errors.
    ServiceError: Exception for service-related errors.
    ResourceError: Exception for resource-related errors.
    ConfigurationError: Exception for configuration-related errors.
    MLError: Exception for machine learning-related errors.
    OperationError: Exception for operation-related errors.
    ExportError: Exception for export-related errors.
    NotFoundError: Exception for resource not found errors.
    ResourceNotFoundError: Alias for NotFoundError.
    DuplicateError: Exception for duplicate entity errors.
    RepositoryError: Exception for repository-related errors.
    SearchError: Exception for search-related errors.
    ExternalServiceError: Exception for external service-related errors.
    ModelError: Exception for model-related errors.
"""

from typing import Dict, Any, Optional


class MedicalResearchSynthesizerError(Exception):
    """
    Base exception class for all Medical Research Synthesizer errors.

    This provides a consistent interface for all exceptions in the application.

    Attributes:
        message (str): The error message.
        details (Dict[str, Any]): Additional details about the error.
    """

    def __init__(self, message: str, details: Dict[str, Any] = None):
        """
        Initialize the MedicalResearchSynthesizerError.

        Args:
            message (str): The error message.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ValidationError(MedicalResearchSynthesizerError):
    """
    Exception raised for validation errors.

    This is used when input data fails validation checks.

    Attributes:
        field (str): The field that failed validation.
        value (Any): The value that failed validation.
    """

    def __init__(self, message: str, field: str = None, value: Any = None, details: Dict[str, Any] = None):
        """
        Initialize the ValidationError.

        Args:
            message (str): The error message.
            field (str, optional): The field that failed validation. Defaults to None.
            value (Any, optional): The value that failed validation. Defaults to None.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.field = field
        self.value = value
        super().__init__(message, details)


class DatabaseError(MedicalResearchSynthesizerError):
    """
    Exception raised for database-related errors.

    This is used when operations on the database fail.

    Attributes:
        operation (str): The database operation that failed.
        model (str): The model involved in the operation.
    """

    def __init__(self, message: str, operation: str = None, model: str = None, details: Dict[str, Any] = None):
        """
        Initialize the DatabaseError.

        Args:
            message (str): The error message.
            operation (str, optional): The database operation that failed. Defaults to None.
            model (str, optional): The model involved in the operation. Defaults to None.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.operation = operation
        self.model = model
        super().__init__(message, details)


class AuthenticationError(MedicalResearchSynthesizerError):
    """
    Exception raised for authentication-related errors.

    This is used when a user fails to authenticate.

    Attributes:
        user_id (str): The ID of the user who failed to authenticate.
    """

    def __init__(self, message: str, user_id: str = None, details: Dict[str, Any] = None):
        """
        Initialize the AuthenticationError.

        Args:
            message (str): The error message.
            user_id (str, optional): The ID of the user who failed to authenticate. Defaults to None.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.user_id = user_id
        super().__init__(message, details)


class AuthorizationError(MedicalResearchSynthesizerError):
    """
    Exception raised for authorization-related errors.

    This is used when a user is not authorized to perform an action.

    Attributes:
        user_id (str): The ID of the user who is not authorized.
        resource (str): The resource the user attempted to access.
        action (str): The action the user attempted to perform.
    """

    def __init__(self, message: str, user_id: str = None, resource: str = None, action: str = None, details: Dict[str, Any] = None):
        """
        Initialize the AuthorizationError.

        Args:
            message (str): The error message.
            user_id (str, optional): The ID of the user who is not authorized. Defaults to None.
            resource (str, optional): The resource the user attempted to access. Defaults to None.
            action (str, optional): The action the user attempted to perform. Defaults to None.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.user_id = user_id
        self.resource = resource
        self.action = action
        super().__init__(message, details)


class APIError(MedicalResearchSynthesizerError):
    """
    Exception raised for API-related errors.

    This is used when external API calls fail.

    Attributes:
        api (str): The API that failed.
        status_code (int): The HTTP status code returned by the API.
        endpoint (str): The API endpoint that was called.
    """

    def __init__(self, message: str, api: str = None, status_code: int = None, endpoint: str = None, details: Dict[str, Any] = None):
        """
        Initialize the APIError.

        Args:
            message (str): The error message.
            api (str, optional): The API that failed. Defaults to None.
            status_code (int, optional): The HTTP status code returned by the API. Defaults to None.
            endpoint (str, optional): The API endpoint that was called. Defaults to None.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.api = api
        self.status_code = status_code
        self.endpoint = endpoint
        super().__init__(message, details)


class CacheError(MedicalResearchSynthesizerError):
    """
    Exception raised for cache-related errors.

    This is used when cache operations fail.

    Attributes:
        operation (str): The cache operation that failed.
        key (str): The cache key involved in the operation.
    """

    def __init__(self, message: str, operation: str = None, key: str = None, details: Dict[str, Any] = None):
        """
        Initialize the CacheError.

        Args:
            message (str): The error message.
            operation (str, optional): The cache operation that failed. Defaults to None.
            key (str, optional): The cache key involved in the operation. Defaults to None.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.operation = operation
        self.key = key
        super().__init__(message, details)


class TaskError(MedicalResearchSynthesizerError):
    """
    Exception raised for task-related errors.

    This is used when background tasks fail.

    Attributes:
        task_id (str): The ID of the task that failed.
        task_type (str): The type of the task that failed.
    """

    def __init__(self, message: str, task_id: str = None, task_type: str = None, details: Dict[str, Any] = None):
        """
        Initialize the TaskError.

        Args:
            message (str): The error message.
            task_id (str, optional): The ID of the task that failed. Defaults to None.
            task_type (str, optional): The type of the task that failed. Defaults to None.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.task_id = task_id
        self.task_type = task_type
        super().__init__(message, details)


class ServiceError(MedicalResearchSynthesizerError):
    """
    Exception raised for service-related errors.

    This is used when services fail to function properly.

    Attributes:
        service (str): The service that failed.
        operation (str): The operation that failed.
    """

    def __init__(self, message: str, service: str = None, operation: str = None, details: Dict[str, Any] = None):
        """
        Initialize the ServiceError.

        Args:
            message (str): The error message.
            service (str, optional): The service that failed. Defaults to None.
            operation (str, optional): The operation that failed. Defaults to None.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.service = service
        self.operation = operation
        super().__init__(message, details)


class ResourceError(MedicalResearchSynthesizerError):
    """
    Exception raised for resource-related errors.

    This is used when resource allocation or access fails.

    Attributes:
        resource (str): The resource that caused the error.
        resource_type (str): The type of the resource.
    """

    def __init__(self, message: str, resource: str = None, resource_type: str = None, details: Dict[str, Any] = None):
        """
        Initialize the ResourceError.

        Args:
            message (str): The error message.
            resource (str, optional): The resource that caused the error. Defaults to None.
            resource_type (str, optional): The type of the resource. Defaults to None.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.resource = resource
        self.resource_type = resource_type
        super().__init__(message, details)


class ConfigurationError(MedicalResearchSynthesizerError):
    """
    Exception raised for configuration-related errors.

    This is used when the application is misconfigured.

    Attributes:
        setting (str): The configuration setting that caused the error.
        expected (str): The expected value or format of the setting.
        actual (str): The actual value of the setting.
    """

    def __init__(self, message: str, setting: str = None, expected: str = None, actual: str = None, details: Dict[str, Any] = None):
        """
        Initialize the ConfigurationError.

        Args:
            message (str): The error message.
            setting (str, optional): The configuration setting that caused the error. Defaults to None.
            expected (str, optional): The expected value or format of the setting. Defaults to None.
            actual (str, optional): The actual value of the setting. Defaults to None.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.setting = setting
        self.expected = expected
        self.actual = actual
        super().__init__(message, details)


class MLError(MedicalResearchSynthesizerError):
    """
    Exception raised for machine learning-related errors.

    This is used when machine learning operations fail.

    Attributes:
        model (str): The model that failed.
        operation (str): The operation that failed.
        inputs (Any): The inputs that caused the error.
    """

    def __init__(self, message: str, model: str = None, operation: str = None, inputs: Any = None, details: Dict[str, Any] = None):
        """
        Initialize the MLError.

        Args:
            message (str): The error message.
            model (str, optional): The model that failed. Defaults to None.
            operation (str, optional): The operation that failed. Defaults to None.
            inputs (Any, optional): The inputs that caused the error. Defaults to None.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.model = model
        self.operation = operation
        self.inputs = inputs
        super().__init__(message, details)


class OperationError(MedicalResearchSynthesizerError):
    """
    Exception raised for operation-related errors.

    This is used when general operations fail.

    Attributes:
        operation (str): The operation that failed.
        inputs (Any): The inputs that caused the error.
    """

    def __init__(self, message: str, operation: str = None, inputs: Any = None, details: Dict[str, Any] = None):
        """
        Initialize the OperationError.

        Args:
            message (str): The error message.
            operation (str, optional): The operation that failed. Defaults to None.
            inputs (Any, optional): The inputs that caused the error. Defaults to None.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.operation = operation
        self.inputs = inputs
        super().__init__(message, details)


class ExportError(MedicalResearchSynthesizerError):
    """
    Exception raised for export-related errors.

    This is used when exporting data to various formats fails.

    Attributes:
        format (str): The export format that failed.
        data_type (str): The type of data being exported.
    """

    def __init__(self, message: str, format: str = None, data_type: str = None, details: Dict[str, Any] = None):
        """
        Initialize the ExportError.

        Args:
            message (str): The error message.
            format (str, optional): The export format that failed. Defaults to None.
            data_type (str, optional): The type of data being exported. Defaults to None.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.format = format
        self.data_type = data_type
        super().__init__(message, details)


class NotFoundError(MedicalResearchSynthesizerError):
    """
    Exception raised when a resource is not found.

    This is used when a database entity or other resource can't be found.

    Attributes:
        resource_type (str): The type of resource that was not found.
        resource_id (str): The ID of the resource that was not found.
    """

    def __init__(self, resource_type: str, resource_id: str, details: Dict[str, Any] = None):
        """
        Initialize the NotFoundError.

        Args:
            resource_type (str): The type of resource that was not found.
            resource_id (str): The ID of the resource that was not found.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        message = f"{resource_type} with ID {resource_id} not found"
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(message, details)


class ResourceNotFoundError(NotFoundError):
    """
    Exception raised when a resource is not found.
    
    This is an alias for NotFoundError for backward compatibility.
    """
    pass


class DuplicateError(MedicalResearchSynthesizerError):
    """
    Exception raised when a duplicate entity is detected.

    This is used when attempting to create an entity that already exists.

    Attributes:
        resource_type (str): The type of resource that was duplicated.
        resource_id (str): The ID of the resource that was duplicated.
    """

    def __init__(self, resource_type: str, resource_id: str, details: Dict[str, Any] = None):
        """
        Initialize the DuplicateError.

        Args:
            resource_type (str): The type of resource that was duplicated.
            resource_id (str): The ID of the resource that was duplicated.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        message = f"{resource_type} with ID {resource_id} already exists"
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(message, details)


class RepositoryError(MedicalResearchSynthesizerError):
    """
    Exception raised for repository-related errors.

    This is used when repository operations fail.

    Attributes:
        repository (str): The repository that failed.
        operation (str): The operation that failed.
    """

    def __init__(self, message: str, repository: str = None, operation: str = None, details: Dict[str, Any] = None):
        """
        Initialize the RepositoryError.

        Args:
            message (str): The error message.
            repository (str, optional): The repository that failed. Defaults to None.
            operation (str, optional): The operation that failed. Defaults to None.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.repository = repository
        self.operation = operation
        super().__init__(message, details)


class SearchError(MedicalResearchSynthesizerError):
    """
    Exception raised for search-related errors.

    This is used when search operations fail.

    Attributes:
        query (str): The search query that failed.
        source (str): The search source that failed.
        method (str): The search method that failed.
    """

    def __init__(self, message: str, query: str = None, source: str = None, method: str = None, details: Dict[str, Any] = None):
        """
        Initialize the SearchError.

        Args:
            message (str): The error message.
            query (str, optional): The search query that failed. Defaults to None.
            source (str, optional): The search source that failed. Defaults to None.
            method (str, optional): The search method that failed. Defaults to None.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.query = query
        self.source = source
        self.method = method
        super().__init__(message, details)


class ExternalServiceError(MedicalResearchSynthesizerError):
    """
    Exception raised for external service-related errors.

    This is used when external services fail to function properly.

    Attributes:
        service (str): The external service that failed.
        endpoint (str): The endpoint that failed.
        status_code (int): The HTTP status code returned by the service.
    """

    def __init__(self, message: str, service: str = None, endpoint: str = None, status_code: int = None, details: Dict[str, Any] = None):
        """
        Initialize the ExternalServiceError.

        Args:
            message (str): The error message.
            service (str, optional): The external service that failed. Defaults to None.
            endpoint (str, optional): The endpoint that failed. Defaults to None.
            status_code (int, optional): The HTTP status code returned by the service. Defaults to None.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.service = service
        self.endpoint = endpoint
        self.status_code = status_code
        super().__init__(message, details)


class ModelError(MedicalResearchSynthesizerError):
    """
    Exception raised for model-related errors.

    This is used when machine learning models fail to load or make predictions.

    Attributes:
        model (str): The model that failed.
        message (str): The error message.
    """

    def __init__(self, model: str, message: str, details: Dict[str, Any] = None):
        """
        Initialize the ModelError.

        Args:
            model (str): The model that failed.
            message (str): The error message.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        super_message = f"Error in model '{model}': {message}"
        self.model = model
        self.specific_message = message
        super().__init__(super_message, details)


class ResourceNotFoundError(MedicalResearchSynthesizerError):
    """
    Exception raised for model-related errors.

    This is used when machine learning models fail to load or make predictions.

    Attributes:
        model (str): The model that failed.
        message (str): The error message.
    """

    def __init__(self, model: str, message: str, details: Dict[str, Any] = None):
        """
        Initialize the ModelError.

        Args:
            model (str): The model that failed.
            message (str): The error message.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        super_message = f"Error in model '{model}': {message}"
        self.model = model
        self.specific_message = message
        super().__init__(super_message, details)