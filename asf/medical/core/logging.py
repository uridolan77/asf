"""
Logging module for the Medical Research Synthesizer.

This module provides logging configuration and utilities.
"""

import logging
import json
import time
import uuid
from typing import Dict, Any, Optional
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from asf.medical.core.config import settings

class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add a request ID to each request.
    
    This middleware adds a unique ID to each request, which can be used for tracing.
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Add a request ID to the request and response.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response
        """
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
        
        # Add extra fields from record
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
    # Get log level from settings
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONLogFormatter())
    
    file_handler = logging.FileHandler("medical_research_synthesizer.log")
    file_handler.setFormatter(JSONLogFormatter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Configure app logger
    app_logger = logging.getLogger("asf")
    app_logger.setLevel(log_level)
    
    # Disable propagation for some loggers
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
