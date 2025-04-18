"""
Error handling utilities for the API.
"""

from typing import Dict, Any, Optional
from fastapi import HTTPException, status

def handle_api_error(error: Exception, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR, detail: Optional[str] = None) -> Dict[str, Any]:
    """
    Handle API errors and return a standardized error response.
    
    Args:
        error: The exception that occurred
        status_code: HTTP status code to return
        detail: Optional detailed error message
        
    Returns:
        Standardized error response
    """
    error_message = detail or str(error)
    
    # Log the error
    print(f"API Error: {error_message}")
    
    # Return standardized error response
    return {
        "success": False,
        "error": error_message,
        "status": status_code
    }
