"""
Unified error/exception system for the LLM Gateway.

This module defines a hierarchical error/exception system for reliable error handling,
reporting, and resilience logic across all providers and transports. It enables
higher-level components to catch and handle errors based on category without
needing to know the specifics of every possible underlying error.
"""
import logging
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Categories of errors to aid in classification and handling."""
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    RESOURCE = "resource"
    NETWORK = "network"
    PROVIDER = "provider"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONTENT_FILTER = "content_filter"
    SYSTEM = "system"
    INPUT = "input"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ProcessingStage(str, Enum):
    """Pipeline stage where an error occurred."""
    INITIALIZATION = "initialization"
    PRE_PROCESSING = "pre_processing"
    PROVIDER_SELECTION = "provider_selection"
    PROVIDER_REQUEST = "provider_request"
    PROVIDER_RESPONSE = "provider_response"
    POST_PROCESSING = "post_processing"
    STREAMING = "streaming"
    RESOURCE_MANAGEMENT = "resource_management"
    COMPLIANCE = "compliance"
    CACHE = "cache"


class RetryStrategy(str, Enum):
    """Strategy for retrying failed operations."""
    NO_RETRY = "no_retry"
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"


class ErrorMetadata:
    """Metadata for error tracking and handling."""
    
    def __init__(
        self,
        *,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        stage: Optional[ProcessingStage] = None,
        retryable: bool = False,
        retry_strategy: RetryStrategy = RetryStrategy.NO_RETRY,
        retry_count: int = 0,
        max_retries: int = 0,
        provider_id: Optional[str] = None,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.category = category
        self.severity = severity
        self.stage = stage
        self.retryable = retryable
        self.retry_strategy = retry_strategy
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.provider_id = provider_id
        self.model_id = model_id
        self.request_id = request_id
        self.timestamp = timestamp or datetime.utcnow()
        self.details = details or {}
    
    def should_retry(self) -> bool:
        """Determine if the operation should be retried."""
        return self.retryable and self.retry_count < self.max_retries
    
    def increment_retry(self) -> 'ErrorMetadata':
        """Increment the retry count and return a new ErrorMetadata instance."""
        new_metadata = ErrorMetadata(
            category=self.category,
            severity=self.severity,
            stage=self.stage,
            retryable=self.retryable,
            retry_strategy=self.retry_strategy,
            retry_count=self.retry_count + 1,
            max_retries=self.max_retries,
            provider_id=self.provider_id,
            model_id=self.model_id,
            request_id=self.request_id,
            timestamp=datetime.utcnow(),
            details=self.details
        )
        return new_metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "stage": self.stage.value if self.stage else None,
            "retryable": self.retryable,
            "retry_strategy": self.retry_strategy.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorMetadata':
        """Create ErrorMetadata from dictionary."""
        stage = data.get("stage")
        if stage is not None:
            try:
                stage = ProcessingStage(stage)
            except ValueError:
                stage = None
        
        return cls(
            category=ErrorCategory(data.get("category", ErrorCategory.UNKNOWN.value)),
            severity=ErrorSeverity(data.get("severity", ErrorSeverity.ERROR.value)),
            stage=stage,
            retryable=data.get("retryable", False),
            retry_strategy=RetryStrategy(data.get("retry_strategy", RetryStrategy.NO_RETRY.value)),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 0),
            provider_id=data.get("provider_id"),
            model_id=data.get("model_id"),
            request_id=data.get("request_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else None,
            details=data.get("details", {})
        )


# Base Exception Classes

class GatewayError(Exception):
    """Base exception for all LLM Gateway errors."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        metadata: Optional[ErrorMetadata] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.metadata = metadata or ErrorMetadata()
        self.cause = cause
        
        # Log the error when it's created
        self._log_error()
    
    def _log_error(self) -> None:
        """Log the error with appropriate level."""
        log_level = logging.ERROR
        if self.metadata.severity == ErrorSeverity.CRITICAL:
            log_level = logging.CRITICAL
        elif self.metadata.severity == ErrorSeverity.WARNING:
            log_level = logging.WARNING
        elif self.metadata.severity == ErrorSeverity.INFO:
            log_level = logging.INFO
        
        logger.log(
            log_level,
            f"{self.code}: {self.message}",
            extra={
                "error_code": self.code,
                "error_category": self.metadata.category.value,
                "error_severity": self.metadata.severity.value,
                "provider_id": self.metadata.provider_id,
                "model_id": self.metadata.model_id,
                "request_id": self.metadata.request_id,
                "stage": self.metadata.stage.value if self.metadata.stage else None
            }
        )
        
        if self.cause:
            logger.log(log_level, f"Caused by: {str(self.cause)}", exc_info=self.cause)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        result = {
            "code": self.code,
            "message": self.message,
            "metadata": self.metadata.to_dict() if self.metadata else {},
        }
        if self.cause:
            result["cause"] = {
                "type": self.cause.__class__.__name__,
                "message": str(self.cause)
            }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GatewayError':
        """Create an error from dictionary representation."""
        metadata = None
        if "metadata" in data:
            metadata = ErrorMetadata.from_dict(data["metadata"])
        
        return cls(
            message=data["message"],
            code=data.get("code"),
            metadata=metadata
        )


class ConfigurationError(GatewayError):
    """Error related to configuration issues."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        config_key: Optional[str] = None,
        config_value: Any = None,
        expected_type: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        metadata = ErrorMetadata(
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.ERROR,
            stage=ProcessingStage.INITIALIZATION,
            details={
                "config_key": config_key,
                "config_value": str(config_value) if config_value is not None else None,
                "expected_type": expected_type
            }
        )
        super().__init__(message, code=code or "ConfigurationError", metadata=metadata, cause=cause)


class ValidationError(GatewayError):
    """Error related to input validation."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        field_name: Optional[str] = None,
        field_value: Any = None,
        expected_type: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        metadata = ErrorMetadata(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            stage=ProcessingStage.PRE_PROCESSING,
            details={
                "field_name": field_name,
                "field_value": str(field_value) if field_value is not None else None,
                "expected_type": expected_type,
                "constraints": constraints
            }
        )
        super().__init__(message, code=code or "ValidationError", metadata=metadata, cause=cause)


class AuthenticationError(GatewayError):
    """Error related to authentication failures."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        provider_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        metadata = ErrorMetadata(
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.ERROR,
            stage=ProcessingStage.PROVIDER_REQUEST,
            provider_id=provider_id,
            details={}
        )
        super().__init__(message, code=code or "AuthenticationError", metadata=metadata, cause=cause)


class AuthorizationError(GatewayError):
    """Error related to authorization failures."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        provider_id: Optional[str] = None,
        model_id: Optional[str] = None,
        permission: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        metadata = ErrorMetadata(
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.ERROR,
            stage=ProcessingStage.PROVIDER_REQUEST,
            provider_id=provider_id,
            model_id=model_id,
            details={
                "permission": permission
            }
        )
        super().__init__(message, code=code or "AuthorizationError", metadata=metadata, cause=cause)


class ProviderError(GatewayError):
    """Base class for provider-related errors."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        provider_id: Optional[str] = None,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
        provider_error_code: Optional[str] = None,
        provider_error_message: Optional[str] = None,
        retryable: bool = False,
        stage: ProcessingStage = ProcessingStage.PROVIDER_REQUEST,
        cause: Optional[Exception] = None
    ):
        metadata = ErrorMetadata(
            category=ErrorCategory.PROVIDER,
            severity=ErrorSeverity.ERROR,
            stage=stage,
            provider_id=provider_id,
            model_id=model_id,
            request_id=request_id,
            retryable=retryable,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF if retryable else RetryStrategy.NO_RETRY,
            details={
                "provider_error_code": provider_error_code,
                "provider_error_message": provider_error_message
            }
        )
        super().__init__(message, code=code or "ProviderError", metadata=metadata, cause=cause)


class ContentFilterError(ProviderError):
    """Error related to content filtering/moderation."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        provider_id: Optional[str] = None,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
        filter_categories: Optional[List[str]] = None,
        flagged_content: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        metadata = ErrorMetadata(
            category=ErrorCategory.CONTENT_FILTER,
            severity=ErrorSeverity.WARNING,
            stage=ProcessingStage.PROVIDER_RESPONSE,
            provider_id=provider_id,
            model_id=model_id,
            request_id=request_id,
            retryable=False,
            details={
                "filter_categories": filter_categories,
                "flagged_content": flagged_content
            }
        )
        super().__init__(
            message, 
            code=code or "ContentFilterError", 
            provider_id=provider_id,
            model_id=model_id,
            request_id=request_id,
            provider_error_code="content_filtered",
            provider_error_message=message,
            retryable=False,
            stage=ProcessingStage.PROVIDER_RESPONSE,
            cause=cause
        )


class RateLimitError(ProviderError):
    """Error related to rate limiting."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        provider_id: Optional[str] = None,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
        limit_type: Optional[str] = None,
        retry_after_seconds: Optional[int] = None,
        cause: Optional[Exception] = None
    ):
        metadata = ErrorMetadata(
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.WARNING,
            stage=ProcessingStage.PROVIDER_REQUEST,
            provider_id=provider_id,
            model_id=model_id,
            request_id=request_id,
            retryable=retry_after_seconds is not None,
            retry_strategy=RetryStrategy.LINEAR_BACKOFF,
            details={
                "limit_type": limit_type,
                "retry_after_seconds": retry_after_seconds
            }
        )
        super().__init__(
            message, 
            code=code or "RateLimitError", 
            provider_id=provider_id,
            model_id=model_id,
            request_id=request_id,
            provider_error_code="rate_limit_exceeded",
            provider_error_message=message,
            retryable=retry_after_seconds is not None,
            stage=ProcessingStage.PROVIDER_REQUEST,
            cause=cause
        )


class ProviderTimeoutError(ProviderError):
    """Error related to provider request timeouts."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        provider_id: Optional[str] = None,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        cause: Optional[Exception] = None
    ):
        metadata = ErrorMetadata(
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.ERROR,
            stage=ProcessingStage.PROVIDER_REQUEST,
            provider_id=provider_id,
            model_id=model_id,
            request_id=request_id,
            retryable=True,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            details={
                "timeout_seconds": timeout_seconds
            }
        )
        super().__init__(
            message, 
            code=code or "ProviderTimeoutError", 
            provider_id=provider_id,
            model_id=model_id,
            request_id=request_id,
            provider_error_code="timeout",
            provider_error_message=message,
            retryable=True,
            stage=ProcessingStage.PROVIDER_REQUEST,
            cause=cause
        )


class NetworkError(GatewayError):
    """Error related to network connectivity issues."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        provider_id: Optional[str] = None,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        retryable: bool = True,
        cause: Optional[Exception] = None
    ):
        metadata = ErrorMetadata(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            stage=ProcessingStage.PROVIDER_REQUEST,
            provider_id=provider_id,
            model_id=model_id,
            request_id=request_id,
            retryable=retryable,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF if retryable else RetryStrategy.NO_RETRY,
            details={
                "endpoint": endpoint,
                "status_code": status_code
            }
        )
        super().__init__(message, code=code or "NetworkError", metadata=metadata, cause=cause)


class ResourceError(GatewayError):
    """Error related to resource management issues."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        pool_id: Optional[str] = None,
        operation: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        metadata = ErrorMetadata(
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.ERROR,
            stage=ProcessingStage.RESOURCE_MANAGEMENT,
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
                "pool_id": pool_id,
                "operation": operation
            }
        )
        super().__init__(message, code=code or "ResourceError", metadata=metadata, cause=cause)


class ResourcePoolError(ResourceError):
    """Error specific to resource pool operations."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        pool_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        operation: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message,
            code=code or "ResourcePoolError",
            resource_type=resource_type,
            pool_id=pool_id,
            operation=operation,
            cause=cause
        )


class ResourceAcquisitionError(ResourceError):
    """Error when acquiring a resource from a pool."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        pool_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        resource_limit_reached: bool = False,
        retryable: bool = True,
        cause: Optional[Exception] = None
    ):
        metadata = ErrorMetadata(
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.ERROR,
            stage=ProcessingStage.RESOURCE_MANAGEMENT,
            retryable=retryable,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF if retryable else RetryStrategy.NO_RETRY,
            details={
                "resource_type": resource_type,
                "pool_id": pool_id,
                "timeout_seconds": timeout_seconds,
                "resource_limit_reached": resource_limit_reached,
                "operation": "acquire"
            }
        )
        super().__init__(message, code=code or "ResourceAcquisitionError", metadata=metadata, cause=cause)


class ResourceReleaseError(ResourceError):
    """Error when releasing a resource back to a pool."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        pool_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message,
            code=code or "ResourceReleaseError",
            resource_type=resource_type,
            resource_id=resource_id,
            pool_id=pool_id,
            operation="release",
            cause=cause
        )


class TransportError(GatewayError):
    """Error related to transport mechanisms (HTTP, WebSocket, gRPC, etc.)."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        transport_type: Optional[str] = None,
        provider_id: Optional[str] = None,
        request_id: Optional[str] = None,
        retryable: bool = True,
        cause: Optional[Exception] = None
    ):
        metadata = ErrorMetadata(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            stage=ProcessingStage.PROVIDER_REQUEST,
            provider_id=provider_id,
            request_id=request_id,
            retryable=retryable,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF if retryable else RetryStrategy.NO_RETRY,
            details={
                "transport_type": transport_type
            }
        )
        super().__init__(message, code=code or "TransportError", metadata=metadata, cause=cause)


class GrpcTransportError(TransportError):
    """Error specific to gRPC transport."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        provider_id: Optional[str] = None,
        request_id: Optional[str] = None,
        grpc_code: Optional[int] = None,
        retryable: Optional[bool] = None,
        cause: Optional[Exception] = None
    ):
        # Determine retryability based on gRPC status code
        if retryable is None:
            # Standard gRPC retryable status codes
            # UNAVAILABLE (14), RESOURCE_EXHAUSTED (8), ABORTED (10)
            retryable_codes = {8, 10, 14}
            retryable = grpc_code in retryable_codes
        
        metadata = ErrorMetadata(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            stage=ProcessingStage.PROVIDER_REQUEST,
            provider_id=provider_id,
            request_id=request_id,
            retryable=retryable,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF if retryable else RetryStrategy.NO_RETRY,
            details={
                "transport_type": "grpc",
                "grpc_code": grpc_code
            }
        )
        super().__init__(
            message, 
            code=code or "GrpcTransportError", 
            transport_type="grpc",
            provider_id=provider_id,
            request_id=request_id,
            retryable=retryable,
            cause=cause
        )


class HttpTransportError(TransportError):
    """Error specific to HTTP transport."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        provider_id: Optional[str] = None,
        request_id: Optional[str] = None,
        status_code: Optional[int] = None,
        url: Optional[str] = None,
        retryable: Optional[bool] = None,
        cause: Optional[Exception] = None
    ):
        # Determine retryability based on HTTP status code
        if retryable is None:
            # 5xx, 408 (Request Timeout), 429 (Too Many Requests) are typically retryable
            retryable_codes = {408, 429, 500, 502, 503, 504}
            retryable = status_code in retryable_codes if status_code else False
        
        metadata = ErrorMetadata(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            stage=ProcessingStage.PROVIDER_REQUEST,
            provider_id=provider_id,
            request_id=request_id,
            retryable=retryable,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF if retryable else RetryStrategy.NO_RETRY,
            details={
                "transport_type": "http",
                "status_code": status_code,
                "url": url
            }
        )
        super().__init__(
            message, 
            code=code or "HttpTransportError", 
            transport_type="http",
            provider_id=provider_id,
            request_id=request_id,
            retryable=retryable,
            cause=cause
        )


class WebSocketTransportError(TransportError):
    """Error specific to WebSocket transport."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        provider_id: Optional[str] = None,
        request_id: Optional[str] = None,
        close_code: Optional[int] = None,
        close_reason: Optional[str] = None,
        retryable: bool = True,
        cause: Optional[Exception] = None
    ):
        metadata = ErrorMetadata(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            stage=ProcessingStage.PROVIDER_REQUEST,
            provider_id=provider_id,
            request_id=request_id,
            retryable=retryable,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF if retryable else RetryStrategy.NO_RETRY,
            details={
                "transport_type": "websocket",
                "close_code": close_code,
                "close_reason": close_reason
            }
        )
        super().__init__(
            message, 
            code=code or "WebSocketTransportError", 
            transport_type="websocket",
            provider_id=provider_id,
            request_id=request_id,
            retryable=retryable,
            cause=cause
        )


class MCPSessionError(GatewayError):
    """Error related to MCP session management."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        provider_id: Optional[str] = None,
        session_id: Optional[str] = None,
        operation: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        metadata = ErrorMetadata(
            category=ErrorCategory.PROVIDER,
            severity=ErrorSeverity.ERROR,
            stage=ProcessingStage.PROVIDER_REQUEST,
            provider_id=provider_id,
            details={
                "session_id": session_id,
                "operation": operation
            }
        )
        super().__init__(message, code=code or "MCPSessionError", metadata=metadata, cause=cause)


class CircuitBreakerError(GatewayError):
    """Error when a circuit breaker prevents an operation."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        provider_id: Optional[str] = None,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
        failure_count: Optional[int] = None,
        failure_threshold: Optional[int] = None,
        reset_timeout_seconds: Optional[int] = None,
        cause: Optional[Exception] = None
    ):
        metadata = ErrorMetadata(
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.WARNING,
            stage=ProcessingStage.PROVIDER_SELECTION,
            provider_id=provider_id,
            model_id=model_id,
            request_id=request_id,
            retryable=False,
            details={
                "failure_count": failure_count,
                "failure_threshold": failure_threshold,
                "reset_timeout_seconds": reset_timeout_seconds
            }
        )
        super().__init__(message, code=code or "CircuitBreakerError", metadata=metadata, cause=cause)


class RoutingError(GatewayError):
    """Error when no suitable provider can be found."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
        tried_providers: Optional[List[str]] = None,
        cause: Optional[Exception] = None
    ):
        metadata = ErrorMetadata(
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.ERROR,
            stage=ProcessingStage.PROVIDER_SELECTION,
            model_id=model_id,
            request_id=request_id,
            retryable=False,
            details={
                "tried_providers": tried_providers or []
            }
        )
        super().__init__(message, code=code or "RoutingError", metadata=metadata, cause=cause)


class ComplianceError(GatewayError):
    """Error related to compliance checks."""
    
    def __init__(
        self, 
        message: str, 
        *,
        code: Optional[str] = None,
        request_id: Optional[str] = None,
        framework_id: Optional[str] = None,
        rule_id: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        cause: Optional[Exception] = None
    ):
        metadata = ErrorMetadata(
            category=ErrorCategory.CONTENT_FILTER,
            severity=severity,
            stage=ProcessingStage.COMPLIANCE,
            request_id=request_id,
            retryable=False,
            details={
                "framework_id": framework_id,
                "rule_id": rule_id
            }
        )
        super().__init__(message, code=code or "ComplianceError", metadata=metadata, cause=cause)


# Error handling utilities

def format_provider_error(provider_error: Exception, provider_id: str) -> ProviderError:
    """Convert a provider-specific exception to a standardized ProviderError."""
    # Default conversion
    error_message = str(provider_error)
    error_class = ProviderError
    
    # Check error class to determine the appropriate conversion
    error_type = type(provider_error).__name__.lower()
    
    if "timeout" in error_type or "timeout" in error_message.lower():
        error_class = ProviderTimeoutError
    elif "rate" in error_type and "limit" in error_type:
        error_class = RateLimitError
    elif "content" in error_type and ("filter" in error_type or "moderation" in error_type):
        error_class = ContentFilterError
    elif "auth" in error_type:
        if "key" in error_type or "credential" in error_type:
            error_class = AuthenticationError
        else:
            error_class = AuthorizationError
    
    # Create and return the appropriate error
    return error_class(
        f"Provider error from {provider_id}: {error_message}",
        provider_id=provider_id,
        cause=provider_error
    )


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error is retryable."""
    if isinstance(error, GatewayError):
        return error.metadata.retryable
    
    # For non-GatewayError exceptions, check common patterns
    error_type = type(error).__name__.lower()
    error_message = str(error).lower()
    
    # Common network/temporary errors that should be retried
    retryable_patterns = [
        "timeout", "timed out",
        "connection refused", "connection reset", "connection aborted",
        "temporary failure", "temporary error",
        "service unavailable", "server unavailable",
        "rate limit", "too many requests",
        "internal server error", "bad gateway",
        "retry", "try again"
    ]
    
    for pattern in retryable_patterns:
        if pattern in error_type or pattern in error_message:
            return True
    
    return False


def get_retry_delay(error: Exception, attempt: int = 1, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """Calculate retry delay for an error based on retry strategy."""
    if not isinstance(error, GatewayError) or not error.metadata.retryable:
        return 0.0
    
    strategy = error.metadata.retry_strategy
    
    if strategy == RetryStrategy.NO_RETRY:
        return 0.0
    elif strategy == RetryStrategy.IMMEDIATE:
        return 0.1  # Small delay to prevent hammering
    elif strategy == RetryStrategy.LINEAR_BACKOFF:
        # Linear backoff: base_delay * attempt
        return min(base_delay * attempt, max_delay)
    elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
        # Exponential backoff: base_delay * 2^attempt + small random jitter
        import random
        jitter = random.uniform(0, 0.1 * base_delay)
        return min(base_delay * (2 ** (attempt - 1)) + jitter, max_delay)
    elif strategy == RetryStrategy.CIRCUIT_BREAKER:
        # Circuit breaker doesn't retry directly
        return 0.0
    
    return base_delay  # Default