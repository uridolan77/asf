"""
Exceptions for the Medical Research Synthesizer.

This module provides custom exceptions for the application.
"""

class ASFException(Exception):
    """Base exception for all ASF exceptions."""
    
    def __init__(self, message: str = "An error occurred"):
        self.message = message
        super().__init__(self.message)


class ValidationError(ASFException):
    """Exception raised for validation errors."""
    
    def __init__(self, message: str = "Validation error"):
        super().__init__(message)


class ResourceNotFoundError(ASFException):
    """Exception raised when a resource is not found."""
    
    def __init__(self, resource_type: str, resource_id: str):
        self.resource_type = resource_type
        self.resource_id = resource_id
        message = f"{resource_type} with ID '{resource_id}' not found"
        super().__init__(message)


class ResourceAlreadyExistsError(ASFException):
    """Exception raised when a resource already exists."""
    
    def __init__(self, resource_type: str, resource_id: str):
        self.resource_type = resource_type
        self.resource_id = resource_id
        message = f"{resource_type} with ID '{resource_id}' already exists"
        super().__init__(message)


class AuthenticationError(ASFException):
    """Exception raised for authentication errors."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message)


class AuthorizationError(ASFException):
    """Exception raised for authorization errors."""
    
    def __init__(self, message: str = "Not authorized to perform this action"):
        super().__init__(message)


class ExternalServiceError(ASFException):
    """Exception raised when an external service fails."""
    
    def __init__(self, service_name: str, message: str = "External service error"):
        self.service_name = service_name
        message = f"{service_name}: {message}"
        super().__init__(message)


class DatabaseError(ASFException):
    """Exception raised for database errors."""
    
    def __init__(self, message: str = "Database error"):
        super().__init__(message)


class CacheError(ASFException):
    """Exception raised for cache errors."""
    
    def __init__(self, message: str = "Cache error"):
        super().__init__(message)


class ModelError(ASFException):
    """Exception raised for ML model errors."""
    
    def __init__(self, model_name: str, message: str = "Model error"):
        self.model_name = model_name
        message = f"{model_name}: {message}"
        super().__init__(message)


class ConfigurationError(ASFException):
    """Exception raised for configuration errors."""
    
    def __init__(self, message: str = "Configuration error"):
        super().__init__(message)


class FileError(ASFException):
    """Exception raised for file-related errors."""
    
    def __init__(self, file_path: str, message: str = "File error"):
        self.file_path = file_path
        message = f"{file_path}: {message}"
        super().__init__(message)


class ExportError(ASFException):
    """Exception raised for export errors."""
    
    def __init__(self, format: str, message: str = "Export error"):
        self.format = format
        message = f"{format}: {message}"
        super().__init__(message)


class SearchError(ASFException):
    """Exception raised for search errors."""
    
    def __init__(self, query: str, message: str = "Search error"):
        self.query = query
        message = f"Query '{query}': {message}"
        super().__init__(message)


class AnalysisError(ASFException):
    """Exception raised for analysis errors."""
    
    def __init__(self, analysis_type: str, message: str = "Analysis error"):
        self.analysis_type = analysis_type
        message = f"{analysis_type}: {message}"
        super().__init__(message)


class KnowledgeBaseError(ASFException):
    """Exception raised for knowledge base errors."""
    
    def __init__(self, kb_name: str, message: str = "Knowledge base error"):
        self.kb_name = kb_name
        message = f"{kb_name}: {message}"
        super().__init__(message)
