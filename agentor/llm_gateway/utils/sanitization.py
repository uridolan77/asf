from lxml.html.clean import Cleaner
import re
import logging

logger = logging.getLogger(__name__)


def sanitize_html(text: str) -> str:
    """Sanitize HTML content.
    
    Args:
        text: The HTML content to sanitize
        
    Returns:
        The sanitized HTML content
    """
    cleaner = Cleaner(
        scripts=True,
        javascript=True,
        style=True,
        inline_style=True,
        meta=True,
        links=True,
        page_structure=False,
        safe_attrs_only=True
    )
    return cleaner.clean_html(text)


def sanitize_input(text: str) -> str:
    """Sanitize user input.
    
    Args:
        text: The user input to sanitize
        
    Returns:
        The sanitized user input
    """
    # Remove any HTML tags
    text = sanitize_html(text)
    
    # Remove any control characters
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    
    # Limit the length
    if len(text) > 10000:
        logger.warning(f"Input text truncated from {len(text)} to 10000 characters")
        text = text[:10000]
    
    return text


def sanitize_output(text: str) -> str:
    """Sanitize output before sending it to the user.
    
    Args:
        text: The output to sanitize
        
    Returns:
        The sanitized output
    """
    # Remove any HTML tags
    text = sanitize_html(text)
    
    # Remove any control characters
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    
    return text
