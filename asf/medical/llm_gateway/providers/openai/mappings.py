"""
Mapping functions between Gateway and OpenAI formats.

This module provides functions for mapping Gateway LLM models to OpenAI API formats and vice versa.
"""

import logging
import base64
from enum import Enum
from typing import Any, Dict, List, Optional, Union, cast

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionContentPartParam,
    ChatCompletionMessageToolCallParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall
from openai.types.completion_usage import CompletionUsage

from asf.medical.llm_gateway.core.models import (
    ContentItem,
    FinishReason,
    GatewayConfig,
    LLMRequest,
    LLMResponse,
    MCPContentType as GatewayContentType,
    MCPRole as GatewayRole,
    PerformanceMetrics,
    ProviderConfig,
    StreamChunk,
    ToolDefinition,
    ToolFunction,
    ToolResult,
    ToolUseRequest,
    UsageStats,
)

logger = logging.getLogger(__name__)

class OpenAIRole(str, Enum):
    """Specific roles recognized by the OpenAI Chat Completions API."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

# Mapping OpenAI finish reasons to Gateway FinishReason
# Reference: https://platform.openai.com/docs/api-reference/chat/object (finish_reason)
OPENAI_FINISH_REASON_MAP = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTERED,
    # "function_call" (legacy) is superseded by tool_calls
    # Others like "error" are usually indicated by API errors
}

# Roles mapping
OPENAI_ROLE_MAP = {
    GatewayRole.SYSTEM: "system",
    GatewayRole.USER: "user",
    GatewayRole.ASSISTANT: "assistant",
    GatewayRole.TOOL: "tool",  # Used for providing tool results back to the model
}

def map_role(gateway_role_str: str) -> Optional[OpenAIRole]:
    """
    Maps gateway role string to the OpenAIRole enum.

    Args:
        gateway_role_str: The role string (e.g., "user", "assistant") from gateway models.

    Returns:
        The corresponding OpenAIRole enum member, or None if mapping fails.
    """
    try:
        # Convert the input string to the GatewayRole Enum member
        gateway_role_enum = GatewayRole(gateway_role_str)
    except ValueError:
        logger.warning(f"Unknown gateway role '{gateway_role_str}' cannot be mapped to OpenAI role.")
        return None

    # Look up the corresponding OpenAIRole Enum member in the map
    openai_role_enum = OPENAI_ROLE_MAP.get(gateway_role_enum)

    if openai_role_enum is None:
        # This case should ideally not happen if OPENAI_ROLE_MAP is complete
        # relative to GatewayRole, but handles potential inconsistencies.
        logger.warning(f"Gateway role '{gateway_role_enum.value}' has no corresponding mapping in OPENAI_ROLE_MAP.")

    return openai_role_enum


def map_content_to_openai(content: Union[str, List[ContentItem], Any]) -> Union[str, List[ChatCompletionContentPartParam], None]:
    """Maps gateway content to OpenAI message content format (string or list of parts)."""
    if isinstance(content, str):
        return content  # Simple text
    elif isinstance(content, list) and all(isinstance(item, ContentItem) for item in content):
        # Multimodal content or multiple text parts
        parts: List[ChatCompletionContentPartParam] = []
        for item in content:
            if item.type == GatewayContentType.TEXT and item.text_content:
                parts.append({"type": "text", "text": item.text_content})
            elif item.type == GatewayContentType.IMAGE:
                try:
                    image_url_data = _create_openai_image_url(item)
                    parts.append({"type": "image_url", "image_url": image_url_data})
                except ValueError as e:
                    logger.warning(f"Skipping image content item due to error: {e}")
            else:
                logger.warning(f"Skipping unsupported content item type for OpenAI: {item.type}")

        if not parts:
            return None
        # OpenAI API limitation: If image_url is present, text content cannot be simple string, must be part list.
        # If only text parts exist, we *could* combine them into a single string, but list is safer.
        return parts
    else:
        logger.warning(f"Unsupported content type for OpenAI mapping: {type(content)}")
        return None


def _create_openai_image_url(item: ContentItem) -> Dict[str, Any]:
    """Creates the image_url dictionary for OpenAI from a ContentItem."""
    # OpenAI accepts base64 encoded images OR public URLs. URLs preferred if available.
    # Format: {"url": "data:image/jpeg;base64,{base64_image_data}"} OR {"url": "http://...", "detail": "auto|low|high"}
    source = item.data.get("image", {}).get("source", {})
    source_type = source.get("type")

    if source_type == "url":
        url = source.get("url")
        if not url:
            raise ValueError("Image source type is 'url' but URL is missing.")
        # Optionally add detail level if needed
        return {"url": url}  # Add "detail": "auto" if needed

    elif source_type == "base64":
        b64_data = source.get("data")
        mime_type = item.mime_type
        if not mime_type:
            raise ValueError("MIME type missing for base64 image")
        if not b64_data:
            raise ValueError("Base64 data missing")
        return {"url": f"data:{mime_type};base64,{b64_data}"}

    else:
        # Try to extract base64 from other fields if needed (e.g. direct item.data)
        raise ValueError(f"Unsupported image source type for OpenAI: {source_type}. Requires 'url' or 'base64'.")


def map_tools_to_openai(tools: List[ToolDefinition]) -> List[ChatCompletionToolParam]:
    """Maps gateway ToolDefinition list to OpenAI ChatCompletionToolParam list."""
    openai_tools: List[ChatCompletionToolParam] = []
    for tool in tools:
        try:
            func = tool.function
            # OpenAI expects 'function' type tools with parameters schema
            openai_tools.append(ChatCompletionToolParam(
                type="function",
                function={
                    "name": func.name,
                    "description": func.description or "",  # Description recommended
                    "parameters": func.parameters or {"type": "object", "properties": {}},  # Schema required
                }
            ))
        except Exception as e:
            logger.error(f"Failed to map tool '{getattr(tool.function, 'name', 'N/A')}' to OpenAI format: {e}", exc_info=True)
    return openai_tools


def map_request(request: LLMRequest, model_id: str) -> Dict[str, Any]:
    """Maps the gateway LLMRequest to OpenAI's chat.completions.create parameters."""
    openai_params: Dict[str, Any] = {}

    # 1. Model Identifier (passed in for Azure compatibility)
    openai_params["model"] = model_id

    # 2. Messages (History + Current Prompt)
    messages: List[ChatCompletionMessageParam] = []

    # Handle System Prompt first
    system_prompt = request.config.system_prompt
    if system_prompt:
        messages.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))

    # Map conversation history
    for turn in request.initial_context.conversation_history:
        # Skip system message if already handled above
        if turn.role == GatewayRole.SYSTEM.value and system_prompt:
            continue

        role = map_role(turn.role)
        if not role:  # Skip unmappable roles
            logger.warning(f"Skipping history turn with unmappable role '{turn.role}'")
            continue

        # Handle different content types and roles
        if role == "tool":
            # Expect content to be ToolResult
            if isinstance(turn.content, ToolResult):
                messages.append(ChatCompletionToolMessageParam(
                    role="tool",
                    content=str(turn.content.output),  # OpenAI expects string content for tool result
                    tool_call_id=turn.content.tool_call_id
                ))
            else:
                logger.warning(f"History turn with role 'tool' has unexpected content type: {type(turn.content)}. Skipping.")
        else:  # System (if not handled above), User, Assistant
            content = map_content_to_openai(turn.content)
            if content:
                # Cast needed because mapped roles aren't specific enough for Param types
                if role == "system":
                    msg = ChatCompletionSystemMessageParam(role=role, content=content)
                elif role == "user":
                    msg = ChatCompletionUserMessageParam(role=role, content=content)
                elif role == "assistant":
                    msg = ChatCompletionAssistantMessageParam(role=role, content=content)  # Handle potential tool_calls here if replaying history?
                else:
                    msg = None  # Should not happen due to role mapping

                if msg:
                    messages.append(msg)
            else:
                logger.warning(f"Could not map content for history turn with role '{role}'. Type: {type(turn.content)}. Skipping.")

    # Map current prompt content (always as 'user' message)
    current_content = map_content_to_openai(request.prompt_content)
    if current_content:
        messages.append(ChatCompletionUserMessageParam(role="user", content=current_content))
    elif not messages:
        # Handle case where history is empty and prompt is empty/unmappable
        raise ValueError("Cannot send request to OpenAI with no valid message content.")

    openai_params["messages"] = messages

    # 3. Configuration Parameters
    if request.config.max_tokens is not None:
        openai_params["max_tokens"] = request.config.max_tokens
    if request.config.temperature is not None:
        openai_params["temperature"] = request.config.temperature
    if request.config.top_p is not None:
        openai_params["top_p"] = request.config.top_p
    if request.config.stop_sequences:
        openai_params["stop"] = request.config.stop_sequences  # Renamed param
    if request.config.presence_penalty is not None:
        openai_params["presence_penalty"] = request.config.presence_penalty
    if request.config.frequency_penalty is not None:
        openai_params["frequency_penalty"] = request.config.frequency_penalty
    # Map other supported params like logit_bias, user, seed if needed

    # 4. Tools & Tool Choice
    if request.tools:
        mapped_tools = map_tools_to_openai(request.tools)
        if mapped_tools:
            openai_params["tools"] = mapped_tools
            # Handle tool_choice if needed
            # Example: openai_params["tool_choice"] = "auto" # or {"type": "function", "function": {"name": "my_func"}}
            # tool_choice_config = request.config.extra_params.get("tool_choice")
            # if tool_choice_config: openai_params["tool_choice"] = tool_choice_config

    return openai_params


def map_finish_reason(openai_reason: Optional[str]) -> FinishReason:
    """Maps OpenAI finish reason string to gateway FinishReason."""
    if not openai_reason:
        return FinishReason.UNKNOWN
    return OPENAI_FINISH_REASON_MAP.get(openai_reason, FinishReason.UNKNOWN)


def map_usage(openai_usage: Optional[CompletionUsage]) -> Optional[UsageStats]:
    """Maps OpenAI CompletionUsage object to gateway UsageStats."""
    if not openai_usage:
        return None
    # Ensure tokens > 0 before creating stats object
    prompt_tokens = openai_usage.prompt_tokens
    completion_tokens = openai_usage.completion_tokens
    if prompt_tokens > 0 or completion_tokens > 0:
        return UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=openai_usage.total_tokens
        )
    return None


def map_tool_calls(tool_calls: List[ChatCompletionMessageToolCallParam], original_tools: Optional[List[ToolDefinition]]) -> List[ToolUseRequest]:
    """Maps OpenAI tool_calls list to gateway ToolUseRequest list."""
    tool_requests = []
    for call in tool_calls:
        # Only handle function calls for now
        if call.type == "function":
            try:
                # Find original tool definition for potential schema/description enrichment
                original_tool_def_func = next((t.function for t in original_tools or [] if t.function.name == call.function.name), None)

                tool_requests.append(ToolUseRequest(
                    id=call.id,  # OpenAI provides the ID for the tool result message
                    type="function",
                    function=ToolFunction(
                        name=call.function.name,
                        description=original_tool_def_func.description if original_tool_def_func else None,
                        # OpenAI provides arguments as a string needing parsing
                        parameters={"arguments": call.function.arguments},  # Store raw string args
                    )
                ))
            except Exception as e:
                logger.error(f"Failed to map OpenAI function tool call (id: {call.id}, name: {getattr(call.function, 'name', 'N/A')}): {e}", exc_info=True)
        else:
            logger.warning(f"Unsupported OpenAI tool call type: {call.type}")
    return tool_requests


def map_tool_call_deltas(deltas: Dict[int, ChoiceDeltaToolCall], original_tools: Optional[List[ToolDefinition]]) -> List[ToolUseRequest]:
    """Maps accumulated OpenAI stream tool deltas to gateway ToolUseRequest list."""
    tool_requests = []
    for index, call_delta in deltas.items():
        if call_delta.type == "function" and call_delta.function:
            try:
                # Find original tool definition
                original_tool_def_func = next((t.function for t in original_tools or [] if t.function.name == call_delta.function.name), None)

                tool_requests.append(ToolUseRequest(
                    id=call_delta.id or f"tool_{index}",  # Use index if ID missing in delta (shouldn't happen?)
                    type="function",
                    function=ToolFunction(
                        name=call_delta.function.name or "",
                        description=original_tool_def_func.description if original_tool_def_func else None,
                        parameters={"arguments": call_delta.function.arguments or ""},
                    )
                ))
            except Exception as e:
                logger.error(f"Failed to map accumulated OpenAI function tool call delta (index: {index}): {e}", exc_info=True)
        else:
            logger.warning(f"Unsupported accumulated OpenAI tool call delta type: {call_delta.type}")
    return tool_requests


def map_response(
    openai_response: Optional[ChatCompletion],
    original_request: LLMRequest,
    error_details: Optional[ErrorDetails],
    llm_latency_ms: Optional[float],
    total_duration_ms: Optional[float],
) -> LLMResponse:
    """Maps the OpenAI ChatCompletion object back to the gateway's LLMResponse."""
    from asf.medical.llm_gateway.core.models import ErrorDetails  # Import here to avoid circular imports

    generated_content: Optional[Union[str, List[ContentItem]]] = None
    tool_use_requests: Optional[List[ToolUseRequest]] = None
    finish_reason: FinishReason = FinishReason.UNKNOWN  # Default
    usage: Optional[UsageStats] = None
    first_choice: Optional[Choice] = None

    if error_details:
        finish_reason = FinishReason.ERROR
    elif openai_response and openai_response.choices:
        first_choice = openai_response.choices[0]
        message = first_choice.message

        # Map Finish Reason
        finish_reason = map_finish_reason(first_choice.finish_reason)

        # Map Content / Tool Calls from assistant message
        if message.content:
            generated_content = message.content  # Usually string content

        if message.tool_calls:
            tool_use_requests = map_tool_calls(message.tool_calls, original_request.tools)
            # Per OpenAI spec, content is null if tool_calls are present
            generated_content = None

        # Map Usage
        usage = map_usage(openai_response.usage)

    # Create performance metrics
    perf_metrics = PerformanceMetrics(
        total_duration_ms=total_duration_ms,
        llm_latency_ms=llm_latency_ms,
    )

    # Final context state might be enriched by interventions later
    final_context_state = original_request.initial_context

    return LLMResponse(
        version=original_request.version,
        request_id=original_request.initial_context.request_id,
        generated_content=generated_content,
        finish_reason=finish_reason,
        tool_use_requests=tool_use_requests,
        usage=usage,
        compliance_result=None,  # Filled by interventions layer
        final_context=final_context_state,
        error_details=error_details,
        performance_metrics=perf_metrics,
        # Optionally store raw response for debugging
        raw_provider_response=openai_response.model_dump() if openai_response else None,
    )