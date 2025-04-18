"""
CrossRef API exceptions.
This module defines exceptions specific to the CrossRef API client.
"""

class CrossRefError(Exception):
    """Base class for CrossRef API exceptions."""
    pass

class CrossRefRateLimitError(CrossRefError):
    """Raised when the CrossRef API rate limit is exceeded."""
    pass

class CrossRefAuthenticationError(CrossRefError):
    """Raised when authentication to the CrossRef API fails."""
    pass

class CrossRefResourceNotFoundError(CrossRefError):
    """Raised when a requested resource is not found."""
    pass

class CrossRefValidationError(CrossRefError):
    """Raised when the input data fails validation."""
    pass

class CrossRefAPIError(CrossRefError):
    """Raised when the CrossRef API returns an error."""
    def __init__(self, status_code=None, message=None):
        self.status_code = status_code
        self.message = message
        super().__init__(f"CrossRef API Error (Status {status_code}): {message}")