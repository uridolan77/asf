"""
Debug configuration for the BO backend.
This file contains settings to enable more verbose debugging.
"""

import logging
import os
import sys

def setup_debug_logging():
    """
    Set up debug logging for the backend.
    This function configures logging with DEBUG level and detailed formatting.
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler with detailed formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Set specific loggers to DEBUG level
    logging.getLogger("asf.medical.llm_gateway").setLevel(logging.DEBUG)
    logging.getLogger("asf.bo.backend.api").setLevel(logging.DEBUG)
    logging.getLogger("asf.medical.llm_gateway.providers").setLevel(logging.DEBUG)
    logging.getLogger("asf.medical.llm_gateway.core").setLevel(logging.DEBUG)
    
    # Log debug configuration
    logging.info("Debug logging configured")

def inject_debug_statements():
    """
    Inject debug statements into key modules.
    This is a more invasive approach that adds print statements at critical points.
    """
    # This is a placeholder for potential monkey patching if needed
    pass

# Setup debug logging when this module is imported
setup_debug_logging()
