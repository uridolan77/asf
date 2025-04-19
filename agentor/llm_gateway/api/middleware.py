import json
import re
import logging
from typing import Dict, Any, List, Callable, Optional
import asyncio
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from agentor.llm_gateway.utils.sanitization import sanitize_input, sanitize_output

logger = logging.getLogger(__name__)


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for validating and sanitizing input."""
    
    def __init__(
        self,
        app: ASGIApp,
        prompt_injection_patterns: Optional[List[str]] = None,
        sensitive_data_patterns: Optional[List[str]] = None
    ):
        """Initialize the middleware.
        
        Args:
            app: The ASGI application
            prompt_injection_patterns: Patterns to detect prompt injection attacks
            sensitive_data_patterns: Patterns to detect sensitive data
        """
        super().__init__(app)
        self.prompt_injection_patterns = prompt_injection_patterns or [
            r"ignore previous instructions",
            r"disregard all prior commands",
            r"system: ignore",
            r"ignore the above instructions",
            r"you are now",
            r"your new instructions are",
            r"your purpose is now",
            r"you must ignore",
            r"do not follow",
            r"override previous",
            r"new directive",
            r"forget your previous instructions"
        ]
        
        self.sensitive_data_patterns = sensitive_data_patterns or [
            # Credit card numbers
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            # Social Security Numbers
            r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
            # API keys (generic pattern)
            r"\b(?:api|key|token|secret)[-_]?[a-zA-Z0-9]{16,}\b",
            # Email addresses
            r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            # Phone numbers
            r"\b(?:\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b",
            # IP addresses
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            # AWS keys
            r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
            # OpenAI API keys
            r"\bsk-[a-zA-Z0-9]{32,}\b"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process the request.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response
        """
        # Only process POST requests with JSON bodies
        if request.method == "POST" and request.headers.get("content-type", "").startswith("application/json"):
            # Get the request body
            body = await request.body()
            
            if body:
                # Parse the JSON body
                try:
                    data = json.loads(body)
                    
                    # Check for prompt injection
                    if self._contains_prompt_injection(data):
                        logger.warning("Potential prompt injection detected")
                        return Response(
                            content=json.dumps({"detail": "Potential prompt injection detected"}),
                            status_code=400,
                            media_type="application/json"
                        )
                    
                    # Check for and redact sensitive data
                    if self._contains_sensitive_data(data):
                        logger.warning("Sensitive data detected and redacted")
                        data = self._redact_sensitive_data(data)
                        
                        # Create a new request with the sanitized body
                        body = json.dumps(data).encode()
                        
                        # We need to create a new request with the sanitized body
                        # This is a bit hacky, but it works
                        async def receive():
                            return {
                                "type": "http.request",
                                "body": body,
                                "more_body": False
                            }
                        
                        request._receive = receive
                
                except json.JSONDecodeError:
                    # Not valid JSON, let the application handle it
                    pass
        
        # Process the request
        response = await call_next(request)
        
        # Sanitize the response if it's JSON
        if response.headers.get("content-type", "").startswith("application/json"):
            # Get the response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            if body:
                # Parse the JSON body
                try:
                    data = json.loads(body)
                    
                    # Sanitize the response
                    data = self._sanitize_response(data)
                    
                    # Create a new response with the sanitized body
                    body = json.dumps(data).encode()
                    
                    # Create a new response
                    return Response(
                        content=body,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type="application/json"
                    )
                
                except json.JSONDecodeError:
                    # Not valid JSON, return the original response
                    pass
        
        return response
    
    def _contains_prompt_injection(self, data: Dict[str, Any]) -> bool:
        """Check if the data contains prompt injection patterns.
        
        Args:
            data: The data to check
            
        Returns:
            True if the data contains prompt injection patterns, False otherwise
        """
        # Convert the data to a string
        text = json.dumps(data).lower()
        
        # Check for prompt injection patterns
        for pattern in self.prompt_injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _contains_sensitive_data(self, data: Dict[str, Any]) -> bool:
        """Check if the data contains sensitive data patterns.
        
        Args:
            data: The data to check
            
        Returns:
            True if the data contains sensitive data patterns, False otherwise
        """
        # Convert the data to a string
        text = json.dumps(data)
        
        # Check for sensitive data patterns
        for pattern in self.sensitive_data_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _redact_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive data from the data.
        
        Args:
            data: The data to redact
            
        Returns:
            The redacted data
        """
        # Convert the data to a string
        text = json.dumps(data)
        
        # Redact sensitive data
        for pattern in self.sensitive_data_patterns:
            text = re.sub(pattern, "[REDACTED]", text)
        
        # Convert back to a dictionary
        return json.loads(text)
    
    def _sanitize_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize the response data.
        
        Args:
            data: The data to sanitize
            
        Returns:
            The sanitized data
        """
        # Convert the data to a string
        text = json.dumps(data)
        
        # Sanitize the text
        text = sanitize_output(text)
        
        # Convert back to a dictionary
        return json.loads(text)
