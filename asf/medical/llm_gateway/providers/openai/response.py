"""
Enhanced response format support for OpenAI API client.

This module extends the existing OpenAI client implementation with support for
advanced response formatting options, including JSON mode and response format control.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union

from asf.medical.llm_gateway.core.models import LLMRequest

class ResponseFormat(Enum):
    """Response format options for OpenAI API."""
    TEXT = "text"
    JSON_OBJECT = "json_object"


def add_response_format_params(
    openai_params: Dict[str, Any], 
    request: LLMRequest
) -> Dict[str, Any]:
    """
    Add response format parameters to the OpenAI API request.
    
    Args:
        openai_params: Existing OpenAI parameters dict
        request: The LLM request with configuration options
        
    Returns:
        The updated OpenAI parameters dict
    """
    # Check if response_format is specified in extra_params
    response_format = request.config.extra_params.get("response_format")
    if response_format:
        if response_format == ResponseFormat.JSON_OBJECT.value:
            # Set "response_format": {"type": "json_object"}
            openai_params["response_format"] = {"type": "json_object"}
        elif isinstance(response_format, dict) and response_format.get("type") in ["json_object", "text"]:
            # Pass through if already in correct format
            openai_params["response_format"] = response_format
            
    # Check for json_mode flag (convenience parameter)
    if request.config.extra_params.get("json_mode", False):
        openai_params["response_format"] = {"type": "json_object"}
    
    # Check for seed parameter for deterministic outputs
    if request.config.extra_params.get("seed") is not None:
        seed = request.config.extra_params.get("seed")
        if isinstance(seed, int):
            openai_params["seed"] = seed

    # Support for logprobs parameter
    if request.config.extra_params.get("logprobs") is not None:
        logprobs = request.config.extra_params.get("logprobs")
        if isinstance(logprobs, bool):
            openai_params["logprobs"] = logprobs
        
    return openai_params


def extract_structured_data(content: str, expected_format: str = "json") -> Optional[Dict[str, Any]]:
    """
    Extract structured data from text content based on expected format.
    
    Args:
        content: The text content to parse
        expected_format: The expected format ('json' or other formats in future)
        
    Returns:
        Parsed structured data or None if parsing fails
    """
    import json
    import re
    
    if not content or not isinstance(content, str):
        return None
        
    if expected_format.lower() == "json":
        # First try direct JSON parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
            
        # Look for JSON-like content within markdown code blocks
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(json_pattern, content)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
                
        # Try to find JSON object anywhere in the text
        obj_pattern = r"(\{[\s\S]*\})"
        matches = re.findall(obj_pattern, content)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # If all parsing attempts fail
    return None


def update_mappings_for_response_format(mappings_module):
    """
    Update the OpenAI mappings module to include response format parameters.
    
    Args:
        mappings_module: The mappings module to update
    """
    original_map_request = mappings_module.map_request
    
    def enhanced_map_request(request: LLMRequest, model_id: str) -> Dict[str, Any]:
        """Enhanced request mapping with response format support."""
        openai_params = original_map_request(request, model_id)
        return add_response_format_params(openai_params, request)
    
    # Replace the original function with our enhanced version
    mappings_module.map_request = enhanced_map_request