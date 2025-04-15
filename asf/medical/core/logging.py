"""
Logging utilities module for the Medical Research Synthesizer.

This module provides logging utilities beyond basic configuration,
including structured logging, context-aware logging, and specialized loggers.

Classes:
    ContextLogger: Context-aware logger that includes context in log messages.
    RequestIdFilter: Filter to add request ID to log records.

Functions:
    log_service_call: Log a service call with timing information.
    log_api_call: Log an API call with detailed information.
    log_ml_event: Log a machine learning event.
    log_error: Log an error with details.
"""

import logging
import json
import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Any
import contextvars

from .config import settings

class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add a request ID to each request.
    
    This middleware adds a unique ID to each request, which can be used for tracing.
    """
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

class JSONLogFormatter(logging.Formatter):
    """
    JSON formatter for logs.
    
    This formatter outputs logs in JSON format, which is easier to parse and analyze.
    """
    def format(self, record):
        """
        Format a log record as JSON.
        
        Args:
            record: The log record
            
        Returns:
            JSON-formatted log string
        """
        log_record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in log_record:
                continue
            log_record[key] = value
        
        return json.dumps(log_record)

def setup_logging():
    """
    Set up logging for the application.
    
    This function configures logging with appropriate handlers and formatters.
    
    Returns:
        The root logger
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONLogFormatter())
    
    file_handler = logging.FileHandler("medical_research_synthesizer.log")
    file_handler.setFormatter(JSONLogFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    app_logger = logging.getLogger("asf")
    app_logger.setLevel(log_level)
    
    for logger_name in ["uvicorn", "uvicorn.access"]:
        logging.getLogger(logger_name).propagate = False
    
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class ContextLogger:
    """
    Context-aware logger.

    This class wraps a standard logger and adds context information
    to all log messages, such as request ID, user ID, and other context.

    Attributes:
        logger (logging.Logger): The underlying logger.
        context (Dict[str, Any]): Context information to include in logs.
    """

    def __init__(self, logger: logging.Logger, context: Dict[str, Any] = None):
        """
        Initialize the ContextLogger.

        Args:
            logger (logging.Logger): The underlying logger.
            context (Dict[str, Any], optional): Context information to include in logs. Defaults to None.
        """
        self.logger = logger
        self.context = context or {}

    def debug(self, message: str, **kwargs) -> None:
        """
        Log a debug message with context.

        Args:
            message (str): The log message.
            **kwargs: Additional context information for this specific log.
        """
        self.logger.debug(message, extra={**self.context, **kwargs})

    def info(self, message: str, **kwargs) -> None:
        """
        Log an info message with context.

        Args:
            message (str): The log message.
            **kwargs: Additional context information for this specific log.
        """
        self.logger.info(message, extra={**self.context, **kwargs})

    def warning(self, message: str, **kwargs) -> None:
        """
        Log a warning message with context.

        Args:
            message (str): The log message.
            **kwargs: Additional context information for this specific log.
        """
        self.logger.warning(message, extra={**self.context, **kwargs})

    def error(self, message: str, **kwargs) -> None:
        """
        Log an error message with context.

        Args:
            message (str): The log message.
            **kwargs: Additional context information for this specific log.
        """
        self.logger.error(message, extra={**self.context, **kwargs})

    def exception(self, message: str, exc_info=True, **kwargs) -> None:
        """
        Log an exception message with context.

        Args:
            message (str): The log message.
            exc_info (bool, optional): Whether to include exception info. Defaults to True.
            **kwargs: Additional context information for this specific log.
        """
        self.logger.exception(message, exc_info=exc_info, extra={**self.context, **kwargs})

    def critical(self, message: str, **kwargs) -> None:
        """
        Log a critical message with context.

        Args:
            message (str): The log message.
            **kwargs: Additional context information for this specific log.
        """
        self.logger.critical(message, extra={**self.context, **kwargs})

    def update_context(self, context: Dict[str, Any]) -> None:
        """
        Update the logger's context.

        Args:
            context (Dict[str, Any]): New context information to include in logs.
        """
        self.context.update(context)

class RequestIdFilter(logging.Filter):
    """
    Filter to add request ID to log records.

    This filter adds a request_id field to log records based on the
    current request's ID stored in a context variable.

    Attributes:
        request_id_var (contextvars.ContextVar): Context variable for request ID.
    """

    def __init__(self, request_id_var):
        """
        Initialize the RequestIdFilter.

        Args:
            request_id_var (contextvars.ContextVar): Context variable for request ID.
        """
        self.request_id_var = request_id_var

    def filter(self, record):
        """
        Filter log records, adding request_id attribute.

        Args:
            record (logging.LogRecord): The log record to filter.

        Returns:
            bool: True to include the record in the log, False to exclude it.
        """
        record.request_id = self.request_id_var.get(None)
        return True

def log_service_call(service: str, method: str, duration: float, success: bool, **kwargs) -> None:
    """
    Log a service call with timing information.

    Args:
        service (str): Name of the service.
        method (str): Method of the service that was called.
        duration (float): Duration of the call in seconds.
        success (bool): Whether the call succeeded.
        **kwargs: Additional context information.
    """
    logger = get_logger("service_call")
    logger.info("Service call", extra={"service": service, "method": method, "duration": duration, "success": success, **kwargs})

def log_api_call(api: str, endpoint: str, status_code: int, duration: float, **kwargs) -> None:
    """
    Log an API call with detailed information.

    Args:
        api (str): Name of the API.
        endpoint (str): API endpoint that was called.
        status_code (int): HTTP status code returned by the API.
        duration (float): Duration of the call in seconds.
        **kwargs: Additional context information.
    """
    logger = get_logger("api_call")
    logger.info("API call", extra={"api": api, "endpoint": endpoint, "status_code": status_code, "duration": duration, **kwargs})

def log_ml_event(model: str, operation: str, status: str, duration: float = None, **kwargs) -> None:
    """
    Log a machine learning event.

    Args:
        model (str): Name of the machine learning model.
        operation (str): Operation performed with the model.
        status (str): Status of the operation.
        duration (float, optional): Duration of the operation in seconds. Defaults to None.
        **kwargs: Additional context information.
    """
    logger = get_logger("ml_event")
    logger.info("ML event", extra={"model": model, "operation": operation, "status": status, "duration": duration, **kwargs})

def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """
    Log an error with details.

    Args:
        error (Exception): The error to log.
        context (Dict[str, Any], optional): Additional context information. Defaults to None.
    """
    logger = get_logger("error")
    logger.error("Error occurred", exc_info=error, extra=context or {})
