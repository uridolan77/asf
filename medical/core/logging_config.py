"""
Logging configuration module for the Medical Research Synthesizer.

This module provides functions for configuring and retrieving loggers
with consistent formatting and behavior throughout the application.

Functions:
    configure_logging: Configure the logging system.
    get_logger: Get a logger with the specified name.
    get_structlog_logger: Get a structured logger with the specified name.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog
from structlog.stdlib import LoggerFactory

from .config import settings

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

APP_LOG_FILE = LOGS_DIR / "app.log"
ERROR_LOG_FILE = LOGS_DIR / "error.log"
ACCESS_LOG_FILE = LOGS_DIR / "access.log"

LOG_LEVEL = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

MAX_LOG_SIZE = 10 * 1024 * 1024

BACKUP_COUNT = 5

def configure_logging() -> None:
    """
    Configure logging for the application.
    
    This sets up handlers for console and file logging, configures formatters,
    and sets the appropriate log levels based on the environment.
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.setLevel(LOG_LEVEL)
    
    standard_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(standard_formatter)
    root_logger.addHandler(console_handler)
    
    file_handler = RotatingFileHandler(
        APP_LOG_FILE,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(standard_formatter)
    root_logger.addHandler(file_handler)
    
    error_file_handler = RotatingFileHandler(
        ERROR_LOG_FILE,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_file_handler)
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    if settings.ENVIRONMENT == "development":
        logging.getLogger("asf.medical.api").setLevel(logging.DEBUG)
        logging.getLogger("asf.medical.services").setLevel(logging.DEBUG)
    else:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    
    logging.info(f"Logging configured with level: {settings.LOG_LEVEL.upper()}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)

def get_structlog_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger with the specified name.
    
    Args:
        name (str): The name of the logger, typically __name__
        
    Returns:
        structlog.stdlib.BoundLogger: A configured structlog logger instance.
    """
    return structlog.get_logger(name)

configure_logging()
