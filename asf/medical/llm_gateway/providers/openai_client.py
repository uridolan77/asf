# llm_gateway/providers/openai_client.py

"""
Provider implementation for OpenAI models (e.g., GPT-4, GPT-3.5) using the
official OpenAI Python SDK (v1.0+). Supports both standard OpenAI and Azure OpenAI.
"""
from enum import Enum
import asyncio
import base64
import logging
import os
import json
from asf.medical.core.secrets import SecretManager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union, cast

# --- OpenAI SDK Imports ---
try:
    import openai
    from openai import AsyncOpenAI, AsyncAzureOpenAI
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
    from openai._exceptions import (
        APIError,
        APIStatusError,
        APITimeoutError,
        RateLimitError,
        AuthenticationError,
        PermissionDeniedError,
        BadRequestError, # Replaces InvalidRequestError in v1.x
        OpenAIError,
    )
except ImportError:
    raise ImportError(
        "OpenAI SDK not found. Please install it using 'pip install openai'"
    )

# Gateway imports
from asf.medical.llm_gateway.core.models import (
    ContentItem,
    ErrorDetails,
    ErrorLevel,
    FinishReason,
    GatewayConfig,
    InterventionContext,
    LLMConfig,
    LLMRequest,
    LLMResponse,
    MCPContentType as GatewayContentType,
    MCPRole as GatewayRole,
    PerformanceMetrics,
    ProviderConfig,
    StreamChunk,
    ToolDefinition,
    ToolFunction,
    ToolResult, # Need this if mapping tool results back
    ToolUseRequest,
    UsageStats,
)
from asf.medical.llm_gateway.providers.base import BaseProvider
from asf.medical.llm_gateway.core.connection_pool import LLMConnectionPool

class OpenAIRole(str, Enum):
    """Specific roles recognized by the OpenAI Chat Completions API."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

logger = logging.getLogger(__name__)

# --- Constants ---
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
    GatewayRole.TOOL: "tool", # Used for providing tool results back to the model
}

class OpenAIClient(BaseProvider):
    """
    LLM Gateway provider for OpenAI models using the Chat Completions API.
    Supports standard OpenAI and Azure OpenAI endpoints.
    """

    def __init__(self, provider_config: ProviderConfig, gateway_config: GatewayConfig):
        """
        Initialize the OpenAI Client.

        Args:
            provider_config: Configuration specific to this OpenAI provider instance.
                             Handles standard API key or Azure parameters.
            gateway_config: Global gateway configuration.
        """
        super().__init__(provider_config)
        self.gateway_config = gateway_config
        self._client_pool: Optional[LLMConnectionPool[Union[AsyncOpenAI, AsyncAzureOpenAI]]] = None
        self.is_azure = self.provider_config.connection_params.get("is_azure", False)
        self._secret_manager = SecretManager()

        # Get retry/timeout settings from provider_config or fallback to gateway_config
        self._max_retries = self.provider_config.connection_params.get("max_retries", gateway_config.max_retries)
        self._timeout = self.provider_config.connection_params.get("timeout_seconds", gateway_config.default_timeout_seconds)
        self._retry_delay = self.provider_config.connection_params.get("retry_delay_seconds", gateway_config.retry_delay_seconds)
        self._pool_size = self.provider_config.connection_params.get("max_connections", 10)
        self._min_pool_size = self.provider_config.connection_params.get("min_connections", 2)

        try:
            # Initialize the connection pool with the client factory
            self._client_pool = LLMConnectionPool(
                factory=self._create_client_instance,
                max_size=self._pool_size,
                min_size=self._min_pool_size,
                max_idle_time=self.provider_config.connection_params.get("max_idle_time_seconds", 300.0),
                connection_timeout=self._timeout,
                name=f"openai_{self.provider_id}_pool",
            )
            logger.info(f"Initialized connection pool for OpenAI provider '{self.provider_id}' "
                       f"(max={self._pool_size}, min={self._min_pool_size})")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI connection pool for provider {self.provider_id}: {e}", exc_info=True)
            raise ConnectionError(f"OpenAI connection pool initialization failed: {e}") from e

    def _create_client_instance(self) -> Union[AsyncOpenAI, AsyncAzureOpenAI]:
        """
        Factory function to create a new OpenAI client instance.
        Used by the connection pool to create new connections as needed.
        
        Returns:
            A configured AsyncOpenAI or AsyncAzureOpenAI client instance
        """
        if self.is_azure:
            # --- Azure OpenAI Configuration ---
            api_key_env_var = self.provider_config.connection_params.get("api_key_env_var", "AZURE_OPENAI_API_KEY")
            endpoint_env_var = self.provider_config.connection_params.get("endpoint_env_var", "AZURE_OPENAI_ENDPOINT")
            api_version_env_var = self.provider_config.connection_params.get("api_version_env_var", "AZURE_OPENAI_API_VERSION")

            api_key = os.environ.get(api_key_env_var)
            azure_endpoint = os.environ.get(endpoint_env_var)
            api_version = os.environ.get(api_version_env_var)

            if not all([api_key, azure_endpoint, api_version]):
                raise ValueError(
                    "Azure OpenAI requires API key, endpoint, and API version environment variables. "
                    f"Checked: '{api_key_env_var}', '{endpoint_env_var}', '{api_version_env_var}'"
                )

            client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                max_retries=self._max_retries,
                timeout=self._timeout,
            )
            logger.debug(f"Created Azure OpenAI client instance for endpoint {azure_endpoint}")
            return client

        else:
            # --- Standard OpenAI Configuration ---
            # Get API key from connection parameters
            api_key = self.provider_config.connection_params.get("api_key")

            # Try environment variable if direct key not provided
            if not api_key:
                env_var = self.provider_config.connection_params.get("api_key_env_var")
                if env_var:
                    api_key = os.environ.get(env_var)

            # Try to get API key from database first
            if not api_key and hasattr(self, 'db') and self.db:
                try:
                    # Import here to avoid circular imports
                    from asf.bo.backend.repositories.provider_repository import ProviderRepository
                    from asf.bo.backend.utils.crypto import generate_key

                    # Get encryption key (in production, this should be loaded from a secure source)
                    encryption_key = generate_key()

                    # Initialize repository
                    provider_repo = ProviderRepository(self.db, encryption_key)

                    # Get API keys for this provider
                    api_keys = provider_repo.get_api_keys_by_provider_id(self.provider_id)
                    if api_keys:
                        # Use the first active API key
                        api_key = provider_repo.get_decrypted_api_key(api_keys[0].key_id)
                        if api_key:
                            logger.info(f"Using API key from database for provider '{self.provider_id}'")
                except Exception as e:
                    logger.warning(f"Failed to get API key from database for provider '{self.provider_id}': {e}")

            # Try secret reference if still no key
            if not api_key:
                secret_ref = self.provider_config.connection_params.get("api_key_secret")
                if secret_ref and self._secret_manager:
                    try:
                        # Parse the secret reference in the format "category:name"
                        if ":" in secret_ref:
                            category, name = secret_ref.split(":", 1)
                            api_key = self._secret_manager.get_secret(category, name)
                        else:
                            logger.warning(f"Invalid secret reference format: '{secret_ref}'. Expected format: 'category:name'")
                    except Exception as e:
                        logger.warning(f"Failed to get API key from secret reference '{secret_ref}': {e}")

            # Log the API key (masked) for debugging
            if api_key:
                masked_key = f"{api_key[:5]}...{api_key[-4:]}" if len(api_key) > 10 else "***MASKED***"
                logger.debug(f"Created OpenAI client instance with API key: {masked_key}")

            client = AsyncOpenAI(
                api_key=api_key,
                organization=self.provider_config.connection_params.get("organization"),
                max_retries=self._max_retries,
                timeout=self._timeout,
            )
            return client

    async def initialize(self):
        """Initialize the connection pool."""
        if self._client_pool:
            await self._client_pool.initialize()
            logger.info(f"Initialized connection pool for OpenAI provider '{self.provider_id}'")

    async def cleanup(self):
        """Closes the OpenAI connection pool."""
        if self._client_pool:
            try:
                await self._client_pool.close()
                logger.info(f"Closed connection pool for OpenAI provider '{self.provider_id}'")
            except Exception as e:
                logger.warning(f"Error during OpenAI connection pool cleanup for '{self.provider_id}': {e}", exc_info=True)
        self._client_pool = None

    # --- Core Generation Methods ---

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response from OpenAI using the non-streaming Chat Completions API.
        """
        start_time = datetime.now(timezone.utc)
        llm_latency_ms: Optional[float] = None
        response_obj: Optional[ChatCompletion] = None
        error_details: Optional[ErrorDetails] = None
        request_id = request.initial_context.request_id
        model_id = request.config.model_identifier

        # Enhanced logging for request tracing
        logger.info(f"[OPENAI_REQUEST_START] ID: {request_id} | Model: {model_id} | Provider: {self.provider_id}")
        logger.debug(f"[OPENAI_REQUEST_PARAMS] ID: {request_id} | Temperature: {request.config.temperature} | MaxTokens: {request.config.max_tokens} | Top_p: {request.config.top_p}")

        if not self._client_pool:
             logger.error(f"[OPENAI_ERROR] ID: {request_id} | OpenAI connection pool not initialized")
             raise ConnectionError("OpenAI connection pool is not initialized.")

        try:
            # 1. Map Gateway Request to OpenAI API format
            # Azure requires deployment_id which is often the model name in the request
            logger.debug(f"[OPENAI_REQUEST_MAPPING] ID: {request_id} | Mapping request to OpenAI format")
            openai_params = self._map_request(request, model_id)

            # Log prompt content length for debugging (not the actual content for privacy)
            msg_count = len(openai_params.get("messages", []))
            prompt_len = sum(len(str(msg.get("content", ""))) for msg in openai_params.get("messages", []))
            tool_count = len(openai_params.get("tools", []))
            logger.info(f"[OPENAI_REQUEST_STATS] ID: {request_id} | MessageCount: {msg_count} | PromptLength: {prompt_len} | ToolCount: {tool_count}")

            # 2. Acquire client from pool and call OpenAI API
            logger.info(f"[OPENAI_API_CALL] ID: {request_id} | Sending request to {self.is_azure and 'Azure ' or ''}OpenAI model '{model_id}'")
            llm_call_start = datetime.now(timezone.utc)

            # Use the connection pool to acquire a client
            async with self._client_pool.acquire() as client:
                response_obj = await client.chat.completions.create(**openai_params)

            llm_latency_ms = (datetime.now(timezone.utc) - llm_call_start).total_seconds() * 1000
            logger.info(f"[OPENAI_API_RESPONSE] ID: {request_id} | Received response in {llm_latency_ms:.2f}ms | ResponseID: {response_obj.id}")

            # Log response stats
            choice_count = len(response_obj.choices) if response_obj and response_obj.choices else 0
            first_choice = response_obj.choices[0] if choice_count > 0 else None
            content_len = len(first_choice.message.content or "") if first_choice and first_choice.message else 0
            tool_call_count = len(first_choice.message.tool_calls or []) if first_choice and first_choice.message else 0
            finish_reason = first_choice.finish_reason if first_choice else "unknown"

            # Log token usage
            token_info = ""
            if response_obj and response_obj.usage:
                token_info = f"PromptTokens: {response_obj.usage.prompt_tokens} | CompletionTokens: {response_obj.usage.completion_tokens} | TotalTokens: {response_obj.usage.total_tokens}"

            logger.info(f"[OPENAI_RESPONSE_STATS] ID: {request_id} | ContentLength: {content_len} | ToolCalls: {tool_call_count} | FinishReason: {finish_reason} | {token_info}")

        except (APITimeoutError, asyncio.TimeoutError) as e:
            logger.error(f"[OPENAI_TIMEOUT] ID: {request_id} | Request timed out after {self._timeout}s: {e}", exc_info=True)
            error_details = self._map_error(e, retryable=True)
        except RateLimitError as e:
            logger.error(f"[OPENAI_RATE_LIMIT] ID: {request_id} | Rate limit exceeded: {e}", exc_info=True)
            retry_after = self._get_retry_after(e)
            error_details = self._map_error(e, retryable=True, retry_after=retry_after)
            if retry_after:
                logger.info(f"[OPENAI_RETRY_AFTER] ID: {request_id} | Retry suggested after {retry_after} seconds")
        except AuthenticationError as e:
            logger.error(f"[OPENAI_AUTH_ERROR] ID: {request_id} | Authentication failed: {e}", exc_info=True)
            error_details = self._map_error(e, retryable=False) # Not retryable
        except PermissionDeniedError as e:
            logger.error(f"OpenAI permission denied for request {request_id}: {e}", exc_info=True)
            error_details = self._map_error(e, retryable=False)
        except Exception as e:
            logger.error(f"[OPENAI_UNEXPECTED_ERROR] ID: {request_id} | Error: {e}", exc_info=True)
            error_details = self._map_error(e, retryable=False)

        # Calculate total duration and create response
        total_duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return self._map_response(response_obj, request, error_details, llm_latency_ms, total_duration_ms)

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate a streaming response from OpenAI using the Chat Completions API.
        Yields gateway StreamChunk objects.
        """
        request_id = request.initial_context.request_id
        chunk_index = 0
        
        # Just yield a simple error chunk for now
        yield StreamChunk(
            chunk_id=chunk_index,
            request_id=request_id,
            delta_text="Streaming not implemented yet",
            finish_reason=FinishReason.ERROR
        )

    # --- Helper Methods for Request/Response Mapping ---

    def _map_request(self, request: LLMRequest, model_id: str) -> Dict[str, Any]:
        """
        Map a Gateway LLMRequest to OpenAI API parameters.
        
        Args:
            request: The Gateway LLMRequest to map
            model_id: The model identifier to use
            
        Returns:
            A dictionary of parameters for the OpenAI API
        """
        # Start with basic parameters
        params = {
            "model": model_id,
            "messages": [],
            "temperature": request.config.temperature or 0.7,
            "max_tokens": request.config.max_tokens,
            "top_p": request.config.top_p,
            "presence_penalty": request.config.presence_penalty,
            "frequency_penalty": request.config.frequency_penalty,
            "response_format": {"type": "text"},  # Default to text
        }
        
        # For Azure, use deployment_id instead of model
        if self.is_azure:
            params["deployment_id"] = model_id
            del params["model"]
        
        # Handle conversation history from context
        if request.initial_context and request.initial_context.conversation_history:
            for turn in request.initial_context.conversation_history:
                role = OPENAI_ROLE_MAP.get(turn.role, "user")  # Default to user if unknown
                
                # Handle different content types
                if turn.content_type == GatewayContentType.TEXT:
                    params["messages"].append({
                        "role": role,
                        "content": turn.content,
                    })
                elif turn.content_type == GatewayContentType.TOOL_RESULT:
                    # Tool results need special handling
                    params["messages"].append({
                        "role": "tool",
                        "tool_call_id": turn.metadata.get("tool_call_id", "unknown"),
                        "content": turn.content,
                    })
                # Add other content types as needed
        
        # Add the current prompt
        if isinstance(request.prompt_content, str):
            params["messages"].append({
                "role": "user",
                "content": request.prompt_content,
            })
        elif isinstance(request.prompt_content, list):
            # Handle content items
            content_parts = []
            for item in request.prompt_content:
                if item.content_type == GatewayContentType.TEXT:
                    content_parts.append({"type": "text", "text": item.content})
                elif item.content_type == GatewayContentType.IMAGE:
                    # Handle image content
                    image_url = item.content
                    if item.content.startswith("data:"):
                        # It's a data URL
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        })
                    else:
                        # Assume it's a file path or URL
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        })
            
            params["messages"].append({
                "role": "user",
                "content": content_parts
            })
        
        # Add tools if provided
        if request.tools:
            openai_tools = []
            for tool in request.tools:
                if tool.type == "function":
                    # Convert to OpenAI tool format
                    function_tool = {
                        "type": "function",
                        "function": {
                            "name": tool.function.name,
                            "description": tool.function.description,
                            "parameters": tool.function.parameters,
                        }
                    }
                    openai_tools.append(function_tool)
            
            if openai_tools:
                params["tools"] = openai_tools
                # If tools are provided, set response format to include tool calls
                params["response_format"] = {"type": "json_object"}
        
        # Add system prompt if provided
        if request.config.system_prompt:
            # Insert at the beginning
            params["messages"].insert(0, {
                "role": "system",
                "content": request.config.system_prompt,
            })
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        return params

    def _map_response(
        self, 
        response_obj: Optional[ChatCompletion], 
        request: LLMRequest, 
        error_details: Optional[ErrorDetails] = None,
        llm_latency_ms: Optional[float] = None,
        total_duration_ms: Optional[float] = None
    ) -> LLMResponse:
        """
        Map an OpenAI API response to a Gateway LLMResponse.
        
        Args:
            response_obj: The OpenAI API response
            request: The original Gateway LLMRequest
            error_details: Optional error details if an error occurred
            llm_latency_ms: Optional latency of the LLM call
            total_duration_ms: Optional total duration of the request
            
        Returns:
            A Gateway LLMResponse
        """
        # If there's an error, return an error response
        if error_details:
            return LLMResponse(
                request_id=request.initial_context.request_id,
                error_details=error_details,
                final_context=request.initial_context,
                finish_reason=FinishReason.ERROR,
                performance_metrics=PerformanceMetrics(
                    total_duration_ms=total_duration_ms,
                    llm_latency_ms=llm_latency_ms,
                )
            )
        
        # If there's no response object, return an error response
        if not response_obj or not response_obj.choices:
            return LLMResponse(
                request_id=request.initial_context.request_id,
                error_details=ErrorDetails(
                    code="PROVIDER_EMPTY_RESPONSE",
                    message="Provider returned empty response",
                    level=ErrorLevel.ERROR,
                ),
                final_context=request.initial_context,
                finish_reason=FinishReason.ERROR,
                performance_metrics=PerformanceMetrics(
                    total_duration_ms=total_duration_ms,
                    llm_latency_ms=llm_latency_ms,
                )
            )
        
        # Extract the first choice from the response
        choice = response_obj.choices[0]
        
        # Map the finish reason
        finish_reason = self._map_finish_reason(choice.finish_reason)
        
        # Map the usage
        usage = self._map_usage(response_obj.usage) if response_obj.usage else None
        
        # Extract the content
        content = choice.message.content
        
        # Create the response
        return LLMResponse(
            request_id=request.initial_context.request_id,
            generated_content=content,
            final_context=request.initial_context,
            finish_reason=finish_reason,
            usage=usage,
            performance_metrics=PerformanceMetrics(
                total_duration_ms=total_duration_ms,
                llm_latency_ms=llm_latency_ms,
            )
        )

    def _map_finish_reason(self, finish_reason_str: Optional[str]) -> FinishReason:
        """Map OpenAI finish reason to Gateway FinishReason."""
        if not finish_reason_str:
            return FinishReason.UNKNOWN
        
        return OPENAI_FINISH_REASON_MAP.get(finish_reason_str, FinishReason.UNKNOWN)

    def _map_usage(self, usage: CompletionUsage) -> UsageStats:
        """Map OpenAI usage to Gateway UsageStats."""
        return UsageStats(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    def _map_error(self, error: Exception, retryable: bool = False, retry_after: Optional[int] = None) -> ErrorDetails:
        """Map OpenAI error to Gateway ErrorDetails."""
        code = "PROVIDER_ERROR"
        message = str(error)
        level = ErrorLevel.ERROR
        
        # Map specific error types
        if isinstance(error, APITimeoutError) or isinstance(error, asyncio.TimeoutError):
            code = "PROVIDER_TIMEOUT"
            message = f"Request timed out after {self._timeout}s: {message}"
        elif isinstance(error, RateLimitError):
            code = "PROVIDER_RATE_LIMIT"
            message = f"Rate limit exceeded: {message}"
        elif isinstance(error, AuthenticationError):
            code = "PROVIDER_AUTHENTICATION"
            message = f"Authentication failed: {message}"
        elif isinstance(error, PermissionDeniedError):
            code = "PROVIDER_PERMISSION_DENIED"
            message = f"Permission denied: {message}"
        elif isinstance(error, BadRequestError):
            code = "PROVIDER_BAD_REQUEST"
            message = f"Bad request: {message}"
        elif isinstance(error, APIStatusError):
            code = f"PROVIDER_STATUS_{getattr(error, 'status_code', 'UNKNOWN')}"
            message = f"API error: {message}"
        elif isinstance(error, APIError):
            code = "PROVIDER_API_ERROR"
            message = f"API error: {message}"
        elif isinstance(error, ConnectionError):
            code = "PROVIDER_CONNECTION_ERROR"
            message = f"Connection error: {message}"
        
        return ErrorDetails(
            code=code,
            message=message,
            level=level,
            retryable=retryable,
            retry_after_seconds=retry_after,
        )

    def _get_retry_after(self, error: Exception) -> Optional[int]:
        """Extract retry-after information from rate limit errors."""
        # Try to extract retry-after from response headers
        if hasattr(error, "response") and hasattr(error.response, "headers"):
            retry_after = error.response.headers.get("retry-after")
            if retry_after:
                try:
                    return int(retry_after)
                except (ValueError, TypeError):
                    pass
        
        # Default retry delay
        return self._retry_delay

    def _create_error_chunk(self, request_id: str, chunk_id: int, error_details: ErrorDetails) -> StreamChunk:
        """Create an error chunk for streaming responses."""
        return StreamChunk(
            chunk_id=chunk_id,
            request_id=request_id,
            delta_text=f"Error: {error_details.message}",
            finish_reason=FinishReason.ERROR,
            provider_specific_data={"error": error_details.model_dump()},
        )
