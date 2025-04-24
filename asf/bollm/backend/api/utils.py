"""
Utility functions for the BO backend API.
"""

from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)

def handle_api_error(response):
    """
    Handle API errors from external services.
    
    Args:
        response: The response from the external service
        
    Returns:
        HTTPException: An exception with the appropriate status code and detail
    """
    try:
        error_data = response.json()
        error_message = error_data.get("detail", error_data.get("message", "Unknown error"))
        
        logger.error(f"API error: {error_message} (Status code: {response.status_code})")
        
        raise HTTPException(
            status_code=response.status_code,
            detail=error_message
        )
    except ValueError:
        # If the response is not valid JSON
        logger.error(f"API error: {response.text} (Status code: {response.status_code})")
        
        raise HTTPException(
            status_code=response.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=response.text or "Unknown error"
        )
