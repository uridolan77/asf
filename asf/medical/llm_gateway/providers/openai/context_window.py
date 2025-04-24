"""
Context window management utilities for OpenAI API integration.

This module provides utilities for managing token counts and context windows
when working with OpenAI models, helping prevent token limit errors.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union, Any

from asf.medical.llm_gateway.core.models import (
    ContentItem, 
    LLMRequest, 
    MCPContentType, 
    MCPRole, 
    MCPTurn,
)

logger = logging.getLogger(__name__)

# Token limit constants for different models
MODEL_TOKEN_LIMITS = {
    # GPT-3.5 models
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-3.5-turbo-instruct": 4096,
    # GPT-4 models
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,  # GPT-4 Turbo with 128k context
    "gpt-4-turbo-preview": 128000,
    "gpt-4-vision-preview": 128000,
    "gpt-4o": 128000,  # GPT-4o with 128k context
    # Default for unknown models
    "default": 4096,
}

# Azure OpenAI deployments might use custom names - add them here as needed
# or consider fetching model info from deployment

def get_token_limit(model_id: str) -> int:
    """
    Get the token limit for a specific model.
    
    Args:
        model_id: The model identifier
        
    Returns:
        The token limit for the model
    """
    # For Azure deployments, try to extract the base model
    if "gpt-4" in model_id.lower():
        if any(x in model_id.lower() for x in ["32k", "32000"]):
            return MODEL_TOKEN_LIMITS["gpt-4-32k"]
        elif any(x in model_id.lower() for x in ["turbo", "o"]):
            return MODEL_TOKEN_LIMITS["gpt-4-turbo"]
        else:
            return MODEL_TOKEN_LIMITS["gpt-4"]
    elif "gpt-3.5" in model_id.lower():
        if "16k" in model_id.lower():
            return MODEL_TOKEN_LIMITS["gpt-3.5-turbo-16k"]
        else:
            return MODEL_TOKEN_LIMITS["gpt-3.5-turbo"]
            
    # Return the exact match if found, otherwise default
    return MODEL_TOKEN_LIMITS.get(model_id, MODEL_TOKEN_LIMITS["default"])

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    This is a rough approximation; for precise counts, use tiktoken.
    
    Args:
        text: The text to estimate token count for
        
    Returns:
        Estimated token count
    """
    try:
        import tiktoken
        # Use tiktoken for more accurate estimation when available
        encoder = tiktoken.encoding_for_model("gpt-4")
        return len(encoder.encode(text))
    except (ImportError, Exception):
        # Fallback to rough approximation: ~4 chars per token
        # Slightly overestimates to be safe
        return len(text) // 4 + 1

def estimate_image_tokens(image_item: ContentItem) -> int:
    """
    Estimate tokens for an image based on resolution.
    
    Args:
        image_item: ContentItem containing image data
        
    Returns:
        Estimated token count
    """
    # OpenAI charges different token amounts based on image size and detail level
    # Default to high resolution estimate if we can't determine size
    image_data = image_item.data.get("image", {})
    width = image_data.get("width", 0)
    height = image_data.get("height", 0)
    
    # Detail level affects token usage
    detail = image_data.get("detail", "auto")
    
    if width == 0 or height == 0:
        # Unknown dimensions, assume high resolution (85 tokens)
        return 85
    
    # Calculate based on image dimensions
    # For "low" detail: ~85 tokens regardless of size
    # For "high" detail: ~170 tokens regardless of size
    # For "auto" detail: ~85 tokens if smaller than 512x512, otherwise ~170
    if detail == "low":
        return 85
    elif detail == "high":
        return 170
    else:  # "auto"
        if width <= 512 and height <= 512:
            return 85
        else:
            return 170

def estimate_content_tokens(content: Union[str, List[ContentItem], Any]) -> int:
    """
    Estimate tokens for content which may be text or multimodal.
    
    Args:
        content: Content to estimate token count for
        
    Returns:
        Estimated token count
    """
    if isinstance(content, str):
        return estimate_tokens(content)
    elif isinstance(content, list) and all(isinstance(item, ContentItem) for item in content):
        total = 0
        for item in content:
            if item.type == MCPContentType.TEXT and item.text_content:
                total += estimate_tokens(item.text_content)
            elif item.type == MCPContentType.IMAGE:
                total += estimate_image_tokens(item)
        return total
    else:
        # For unknown content types, return a conservative estimate
        if content is None:
            return 0
        return 20  # Arbitrary small value

def estimate_request_tokens(request: LLMRequest) -> Tuple[int, Dict[str, int]]:
    """
    Estimate the total tokens in a request, including history and current prompt.
    
    Args:
        request: LLMRequest to estimate token count for
        
    Returns:
        Tuple of (total_tokens, breakdown_dict)
    """
    token_counts = {
        "system_prompt": 0,
        "conversation_history": 0,
        "current_prompt": 0,
        "total": 0,
    }
    
    # System prompt tokens
    if request.config.system_prompt:
        system_tokens = estimate_tokens(request.config.system_prompt)
        token_counts["system_prompt"] = system_tokens
        token_counts["total"] += system_tokens
    
    # Conversation history tokens
    history_tokens = 0
    for turn in request.initial_context.conversation_history:
        # Add 4 tokens for role and formatting overhead per turn
        turn_tokens = 4 + estimate_content_tokens(turn.content)
        history_tokens += turn_tokens
    
    token_counts["conversation_history"] = history_tokens
    token_counts["total"] += history_tokens
    
    # Current prompt tokens
    if request.prompt_content:
        prompt_tokens = estimate_content_tokens(request.prompt_content)
        token_counts["current_prompt"] = prompt_tokens
        token_counts["total"] += prompt_tokens
    
    # Add estimated tokens for completion (conservative)
    # This depends on max_tokens setting or model default
    max_tokens = request.config.max_tokens or 1024  # Default if not specified
    
    # Return total and breakdown
    return token_counts["total"] + max_tokens, token_counts

def trim_conversation_history(
    request: LLMRequest,
    model_id: str,
    buffer_tokens: int = 100,
    preserve_system_prompt: bool = True,
    preserve_recent_turns: int = 2,
) -> LLMRequest:
    """
    Trim conversation history to fit within token limits.
    
    Args:
        request: LLMRequest to trim
        model_id: Model identifier to determine token limit
        buffer_tokens: Buffer tokens to leave for safety
        preserve_system_prompt: Whether to preserve the system prompt
        preserve_recent_turns: Minimum number of recent turns to preserve
        
    Returns:
        Trimmed LLMRequest
    """
    # Get model token limit
    token_limit = get_token_limit(model_id)
    available_tokens = token_limit - buffer_tokens
    
    # Initial token estimate
    estimated_tokens, token_breakdown = estimate_request_tokens(request)
    
    # If we're already within limits, return unchanged
    if estimated_tokens <= available_tokens:
        return request
    
    # Need to trim - create a modified request
    trimmed_request = request.model_copy(deep=True)
    history = trimmed_request.initial_context.conversation_history
    
    # Calculate tokens to remove
    tokens_to_remove = estimated_tokens - available_tokens
    logger.info(f"Trimming {tokens_to_remove} tokens from conversation with {len(history)} turns")
    
    # If system prompt is too large and we're not preserving it, remove it
    if not preserve_system_prompt and token_breakdown["system_prompt"] > 0:
        tokens_saved = token_breakdown["system_prompt"]
        trimmed_request.config.system_prompt = ""
        tokens_to_remove -= tokens_saved
        
        if tokens_to_remove <= 0:
            return trimmed_request
    
    # Identify turns to preserve (based on preserve_recent_turns)
    preserved_indices = set(range(max(0, len(history) - preserve_recent_turns), len(history)))
    
    # Start removing oldest turns first (except system turns if preserving system prompt)
    for i in range(len(history)):
        if i in preserved_indices:
            continue  # Skip preserved turns
            
        if preserve_system_prompt and history[i].role == MCPRole.SYSTEM.value:
            continue  # Skip system turns if preserving
        
        # Estimate tokens for this turn
        turn_tokens = 4 + estimate_content_tokens(history[i].content)
        
        # Remove this turn
        history[i] = None  # Mark for removal
        tokens_to_remove -= turn_tokens
        
        if tokens_to_remove <= 0:
            break
    
    # Remove None entries
    trimmed_request.initial_context.conversation_history = [
        turn for turn in history if turn is not None
    ]
    
    # If we still need to trim more, we'll need to truncate content
    if tokens_to_remove > 0:
        logger.warning(f"Need to truncate content to remove additional {tokens_to_remove} tokens")
        # Implementation for content truncation could be added here
        # This would involve more complex logic to partially trim content
    
    return trimmed_request

def update_openai_client_for_context_management(client_class):
    """
    Update the OpenAI client class to add context window management.
    
    Args:
        client_class: The OpenAI client class to update
    """
    original_generate = client_class.generate
    
    async def enhanced_generate(self, request: LLMRequest) -> Any:
        """Enhanced generate method with context window management."""
        model_id = request.config.model_identifier
        
        # Perform token estimation
        estimated_tokens, token_breakdown = estimate_request_tokens(request)
        token_limit = get_token_limit(model_id)
        
        # Log token usage information
        logger.info(f"Estimated tokens for request: {estimated_tokens}/{token_limit} "
                   f"(system:{token_breakdown['system_prompt']}, "
                   f"history:{token_breakdown['conversation_history']}, "
                   f"prompt:{token_breakdown['current_prompt']})")
        
        # Check if we need to trim
        if estimated_tokens > token_limit:
            logger.warning(f"Request exceeds token limit ({estimated_tokens}/{token_limit}), trimming conversation")
            trimmed_request = trim_conversation_history(
                request, 
                model_id,
                buffer_tokens=100,
                preserve_system_prompt=True,
                preserve_recent_turns=2
            )
            
            # Re-estimate after trimming
            new_tokens, new_breakdown = estimate_request_tokens(trimmed_request)
            logger.info(f"After trimming: {new_tokens}/{token_limit} tokens "
                       f"(removed {estimated_tokens - new_tokens} tokens, "
                       f"history size {len(request.initial_context.conversation_history)} → "
                       f"{len(trimmed_request.initial_context.conversation_history)})")
            
            # Replace the request with the trimmed version
            request = trimmed_request
        
        # Call the original generate method
        return await original_generate(self, request)
    
    # Replace the original method with our enhanced version
    client_class.generate = enhanced_generate