"""
Logging Utilities

This module provides logging utilities for DSPy.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, Union, List

# Set up logging
logger = logging.getLogger(__name__)


class DSPyLogger:
    """Logger for DSPy operations."""
    
    def __init__(
        self,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        enable_console: bool = True
    ):
        """
        Initialize the logger.
        
        Args:
            log_level: Log level
            log_file: Log file path
            log_format: Log format
            enable_console: Whether to enable console logging
        """
        # Get logger
        self.logger = logging.getLogger("dspy")
        
        # Set log level
        log_level_value = getattr(logging, log_level)
        self.logger.setLevel(log_level_value)
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # Clear existing handlers
        if self.logger.handlers:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
        
        # Add console handler if enabled
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level_value)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler if log file is provided
        if log_file:
            # Ensure directory exists
            log_dir = os.path.dirname(os.path.abspath(log_file))
            os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level_value)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log(self, level: str, message: str, **kwargs) -> None:
        """
        Log a message.
        
        Args:
            level: Log level
            message: Log message
            **kwargs: Additional log data
        """
        # Format kwargs as JSON if present
        if kwargs:
            log_message = f"{message} - {json.dumps(kwargs, default=str)}"
        else:
            log_message = message
        
        # Log at the appropriate level
        log_func = getattr(self.logger, level.lower())
        log_func(log_message)
    
    def debug(self, message: str, **kwargs) -> None:
        """
        Log a debug message.
        
        Args:
            message: Log message
            **kwargs: Additional log data
        """
        self.log("debug", message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """
        Log an info message.
        
        Args:
            message: Log message
            **kwargs: Additional log data
        """
        self.log("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """
        Log a warning message.
        
        Args:
            message: Log message
            **kwargs: Additional log data
        """
        self.log("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """
        Log an error message.
        
        Args:
            message: Log message
            **kwargs: Additional log data
        """
        self.log("error", message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """
        Log a critical message.
        
        Args:
            message: Log message
            **kwargs: Additional log data
        """
        self.log("critical", message, **kwargs)
    
    def log_request(
        self,
        module_name: str,
        request_type: str,
        request_data: Dict[str, Any],
        sensitive_keys: List[str] = None
    ) -> None:
        """
        Log an API request.
        
        Args:
            module_name: Name of the module making the request
            request_type: Type of request
            request_data: Request data
            sensitive_keys: List of sensitive keys to redact
        """
        # Make a copy of the request data
        request_data_copy = request_data.copy() if request_data else {}
        
        # Redact sensitive data
        if sensitive_keys:
            for key in sensitive_keys:
                if key in request_data_copy:
                    request_data_copy[key] = "[REDACTED]"
        
        self.info(
            f"API Request: {module_name} - {request_type}",
            module=module_name,
            request_type=request_type,
            request_data=request_data_copy
        )
    
    def log_response(
        self,
        module_name: str,
        request_type: str,
        response_data: Dict[str, Any],
        duration_ms: float,
        sensitive_keys: List[str] = None
    ) -> None:
        """
        Log an API response.
        
        Args:
            module_name: Name of the module receiving the response
            request_type: Type of request
            response_data: Response data
            duration_ms: Request duration in milliseconds
            sensitive_keys: List of sensitive keys to redact
        """
        # Make a copy of the response data
        response_data_copy = response_data.copy() if response_data else {}
        
        # Redact sensitive data
        if sensitive_keys:
            for key in sensitive_keys:
                if key in response_data_copy:
                    response_data_copy[key] = "[REDACTED]"
        
        self.info(
            f"API Response: {module_name} - {request_type} - {duration_ms:.2f}ms",
            module=module_name,
            request_type=request_type,
            duration_ms=duration_ms,
            response_data=response_data_copy
        )
    
    def log_error(
        self,
        module_name: str,
        request_type: str,
        error: Exception,
        request_data: Optional[Dict[str, Any]] = None,
        sensitive_keys: List[str] = None
    ) -> None:
        """
        Log an API error.
        
        Args:
            module_name: Name of the module receiving the error
            request_type: Type of request
            error: Error exception
            request_data: Request data
            sensitive_keys: List of sensitive keys to redact
        """
        # Make a copy of the request data
        request_data_copy = request_data.copy() if request_data else {}
        
        # Redact sensitive data
        if sensitive_keys:
            for key in sensitive_keys:
                if key in request_data_copy:
                    request_data_copy[key] = "[REDACTED]"
        
        self.error(
            f"API Error: {module_name} - {request_type} - {str(error)}",
            module=module_name,
            request_type=request_type,
            error=str(error),
            error_type=type(error).__name__,
            request_data=request_data_copy
        )


# Singleton instance
_dspy_logger: Optional[DSPyLogger] = None


def get_dspy_logger() -> DSPyLogger:
    """
    Get the DSPy logger singleton.
    
    Returns:
        DSPyLogger: The DSPy logger
    """
    global _dspy_logger
    if _dspy_logger is None:
        _dspy_logger = DSPyLogger()
    
    return _dspy_logger


def configure_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    enable_console: bool = True
) -> DSPyLogger:
    """
    Configure the DSPy logger.
    
    Args:
        log_level: Log level
        log_file: Log file path
        log_format: Log format
        enable_console: Whether to enable console logging
        
    Returns:
        DSPyLogger: The configured DSPy logger
    """
    global _dspy_logger
    _dspy_logger = DSPyLogger(
        log_level=log_level,
        log_file=log_file,
        log_format=log_format,
        enable_console=enable_console
    )
    
    return _dspy_logger


# Export
__all__ = [
    "DSPyLogger",
    "get_dspy_logger",
    "configure_logger",
]