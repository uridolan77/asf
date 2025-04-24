"""
OpenAI provider client implementation for the domain-agnostic LLM Gateway.

This module provides an OpenAI client implementation that uses resilience patterns
like circuit breaking, retries, and rate limiting to ensure reliable operation.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union, cast, BinaryIO

# OpenAI SDK Imports
import openai
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from openai import (
    APIError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
    PermissionDeniedError,
    BadRequestError,
)

# Gateway imports
from asf.conexus.llm_gateway.core.models import (
    ErrorDetails,
    FinishReason,
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    StreamChunk,
    UsageStats,
)

# Import resilient provider wrapper
from asf.conexus.llm_gateway.providers.resilient_provider import ResilientProviderWrapper
from asf.conexus.llm_gateway.resilience.factory import get_resilience_factory
from asf.conexus.llm_gateway.resilience.retry import RetryPolicy, CONSERVATIVE_RETRY_POLICY
from asf.conexus.llm_gateway.resilience.rate_limiter import RateLimiter, RateLimitConfig, RateLimitStrategy

# Import the base provider class
from asf.conexus.llm_gateway.providers.base import BaseProvider

logger = logging.getLogger("conexus.llm_gateway.providers.openai")


class OpenAIClient(BaseProvider):
    """
    OpenAI provider implementation for the domain-agnostic LLM Gateway.
    
    This client supports both standard OpenAI and Azure OpenAI endpoints,
    with built-in resilience patterns to ensure reliable operations.
    """

    def __init__(self, provider_config: ProviderConfig):
        """
        Initialize the OpenAI Client with resilience patterns.

        Args:
            provider_config: Configuration specific to this OpenAI provider instance.
                             Handles standard API key or Azure parameters.
        """
        super().__init__(provider_config)
        self._client: Optional[Union[AsyncOpenAI, AsyncAzureOpenAI]] = None
        self.is_azure = provider_config.connection_params.get("is_azure", False)
        
        # Get retry/timeout settings from provider_config
        self._max_retries = provider_config.connection_params.get("max_retries", 3)
        self._timeout = provider_config.connection_params.get("timeout_seconds", 60)
        self._retry_delay = provider_config.connection_params.get("retry_delay_seconds", 1)
        
        # Create resilience components
        self._setup_resilience()

        try:
            if self.is_azure:
                self._initialize_azure_client(provider_config)
            else:
                self._initialize_openai_client(provider_config)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI SDK client for provider {self.provider_id}: {e}", exc_info=True)
            raise ConnectionError(f"OpenAI SDK client initialization failed: {e}") from e

    def _setup_resilience(self):
        """
        Set up resilience components for this provider.
        """
        # Get the resilience factory
        resilience_factory = get_resilience_factory()
        
        # Create circuit breaker
        self.circuit_breaker = resilience_factory.get_or_create_circuit_breaker(
            name=f"openai_{self.provider_id}",
            provider_id=self.provider_id,
            failure_threshold=5,
            recovery_timeout=30,
            half_open_max_calls=1
        )
        
        # Create retry policy
        self.retry_policy = CONSERVATIVE_RETRY_POLICY
        
        # Create rate limiter with appropriate limits based on tier
        rpm = self.provider_config.connection_params.get("requests_per_minute", 60)
        self.rate_limiter = resilience_factory.create_rate_limiter(
            name=f"openai_{self.provider_id}",
            requests_per_minute=rpm,
            strategy="adaptive"  # Use adaptive rate limiting for OpenAI
        )
        
        logger.info(
            f"Set up resilience for OpenAI provider {self.provider_id} "
            f"with circuit_breaker={self.circuit_breaker.name}, "
            f"retry_policy={self.retry_policy.name}, "
            f"rate_limiter={rpm}rpm"
        )

    def _initialize_openai_client(self, provider_config: ProviderConfig):
        """
        Initialize the standard OpenAI client.
        
        Args:
            provider_config: Provider configuration with connection parameters
        """
        # Get API key from configuration or environment variables
        api_key = provider_config.connection_params.get("api_key")
        if not api_key:
            # Try environment variable
            api_key_env_var = provider_config.connection_params.get("api_key_env_var", "OPENAI_API_KEY")
            api_key = os.environ.get(api_key_env_var)
            
        if not api_key:
            raise ValueError(f"No API key found for OpenAI provider '{self.provider_id}'")
            
        # Get organization ID if available
        org_id = provider_config.connection_params.get("organization_id")
        if not org_id:
            # Try environment variable
            org_id_env_var = provider_config.connection_params.get("org_id_env_var", "OPENAI_ORGANIZATION")
            org_id = os.environ.get(org_id_env_var)
        
        # Set base URL if specified
        base_url = provider_config.connection_params.get("base_url")
        
        # Log masked key for debugging
        if api_key:
            masked_key = f"{api_key[:5]}...{api_key[-4:]}" if len(api_key) > 10 else "***MASKED***"
            logger.info(f"Using API key for provider '{self.provider_id}': {masked_key}")

        # Initialize the OpenAI client
        client_kwargs = {
            "api_key": api_key,
            "organization": org_id,
            "max_retries": self._max_retries,
            "timeout": self._timeout,
        }
        
        # Add base_url if specified (allows redirecting to a proxy or alternative endpoint)
        if base_url:
            client_kwargs["base_url"] = base_url
            logger.info(f"Using custom base URL for provider '{self.provider_id}': {base_url}")
            
        self._client = AsyncOpenAI(**client_kwargs)
        logger.info(f"Initialized OpenAIClient provider '{self.provider_id}'")

    def _initialize_azure_client(self, provider_config: ProviderConfig):
        """
        Initialize the Azure OpenAI client.
        
        Args:
            provider_config: Provider configuration with connection parameters
        """
        # Get Azure-specific configuration
        connection_params = provider_config.connection_params
        
        # Get API key from configuration or environment variables
        api_key = connection_params.get("api_key")
        if not api_key:
            # Try environment variable
            api_key_env_var = connection_params.get("api_key_env_var", "AZURE_OPENAI_API_KEY")
            api_key = os.environ.get(api_key_env_var)
            
        # Get required Azure parameters
        azure_endpoint = connection_params.get("azure_endpoint")
        if not azure_endpoint:
            # Try environment variable
            azure_endpoint_env_var = connection_params.get("endpoint_env_var", "AZURE_OPENAI_ENDPOINT")
            azure_endpoint = os.environ.get(azure_endpoint_env_var)
            
        api_version = connection_params.get("api_version")
        if not api_version:
            # Try environment variable
            api_version_env_var = connection_params.get("api_version_env_var", "AZURE_OPENAI_API_VERSION")
            api_version = os.environ.get(api_version_env_var)

        # Verify all required parameters are present
        if not all([api_key, azure_endpoint, api_version]):
            raise ValueError(
                "Azure OpenAI requires API key, endpoint, and API version. "
                f"Missing: {', '.join(p for p, v in [('API key', api_key), ('endpoint', azure_endpoint), ('API version', api_version)] if not v)}"
            )

        # Initialize the Azure OpenAI client
        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            max_retries=self._max_retries,
            timeout=self._timeout,
        )
        logger.info(f"Initialized AzureOpenAIClient provider '{self.provider_id}' for endpoint {azure_endpoint}")

    async def cleanup(self):
        """Closes the OpenAI client."""
        if self._client:
            try:
                # httpx client used internally should manage connections.
                # await self._client.close() # Usually not required for OpenAI SDK v1+
                logger.info(f"OpenAI client for provider '{self.provider_id}' cleanup called (usually managed by httpx).")
            except Exception as e:
                logger.warning(f"Error during OpenAI client cleanup for '{self.provider_id}': {e}")
        self._client = None
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response from OpenAI using the non-streaming Chat Completions API.
        
        This implementation includes resilience patterns such as circuit breaking,
        retries, and rate limiting via decorators applied dynamically.
        
        Args:
            request: The LLM request to process
            
        Returns:
            LLMResponse: The response from OpenAI
        """
        start_time = datetime.now(timezone.utc)
        llm_latency_ms: Optional[float] = None
        response_obj: Optional[ChatCompletion] = None
        error_details: Optional[ErrorDetails] = None
        request_id = request.request_id
        model_id = request.model_identifier

        # Enhanced logging for request tracing
        logger.info(f"[OPENAI_REQUEST_START] ID: {request_id} | Model: {model_id} | Provider: {self.provider_id}")
        logger.debug(f"[OPENAI_REQUEST_PARAMS] ID: {request_id} | Temperature: {request.temperature} | MaxTokens: {request.max_tokens}")

        if not self._client:
            logger.error(f"[OPENAI_ERROR] ID: {request_id} | OpenAI client not initialized")
            raise ConnectionError("OpenAI client is not initialized.")

        try:
            # Apply rate limiting (already done via wrapper decorators)
            await self.rate_limiter.wait()
            
            # 1. Map Gateway Request to OpenAI API format
            # Azure requires deployment_id which is often the model name in the request
            logger.debug(f"[OPENAI_REQUEST_MAPPING] ID: {request_id} | Mapping request to OpenAI format")
            openai_params = self._map_request(request, model_id)

            # Log prompt content length for debugging (not the actual content for privacy)
            msg_count = len(openai_params.get("messages", []))
            prompt_len = sum(len(str(msg.get("content", ""))) for msg in openai_params.get("messages", []))
            tool_count = len(openai_params.get("tools", []))
            logger.info(f"[OPENAI_REQUEST_STATS] ID: {request_id} | MessageCount: {msg_count} | PromptLength: {prompt_len} | ToolCount: {tool_count}")

            # 2. Call OpenAI API
            logger.info(f"[OPENAI_API_CALL] ID: {request_id} | Sending request to {self.is_azure and 'Azure ' or ''}OpenAI model '{model_id}'")
            llm_call_start = datetime.now(timezone.utc)

            response_obj = await self._client.chat.completions.create(**openai_params)
            
            # Record success to the rate limiter and circuit breaker
            await self.rate_limiter.record_success()
            self.circuit_breaker.record_success()

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
            # Rate limiting/circuit breaker feedback
            await self.rate_limiter.record_failure()
            self.circuit_breaker.record_failure()
            
            logger.error(f"[OPENAI_TIMEOUT] ID: {request_id} | Request timed out after {self._timeout}s: {e}")
            error_details = self._map_error(e, timeout=self._timeout, retryable=True)
        except RateLimitError as e:
            # Rate limiting/circuit breaker feedback
            await self.rate_limiter.record_failure()
            self.circuit_breaker.record_failure()
            
            logger.error(f"[OPENAI_RATE_LIMIT] ID: {request_id} | Rate limit exceeded: {e}")
            retry_after = self._get_retry_after(e)
            error_details = self._map_error(e, retryable=True, retry_after=retry_after)
            if retry_after:
                logger.info(f"[OPENAI_RETRY_AFTER] ID: {request_id} | Retry suggested after {retry_after} seconds")
        except AuthenticationError as e:
            # Record failure for circuit breaker
            self.circuit_breaker.record_failure()
            
            logger.error(f"[OPENAI_AUTH_ERROR] ID: {request_id} | Authentication failed: {e}")
            error_details = self._map_error(e, retryable=False)  # Not retryable
        except PermissionDeniedError as e:
            # Record failure for circuit breaker
            self.circuit_breaker.record_failure()
            
            logger.error(f"OpenAI permission denied for request {request_id}: {e}")
            error_details = self._map_error(e, retryable=False)
        except BadRequestError as e:
            # Record failure for circuit breaker (but this may not be a provider issue)
            self.circuit_breaker.record_failure()
            
            logger.error(f"OpenAI bad request for request {request_id}: {e}")
            error_details = self._map_error(e, retryable=False)
        except APIStatusError as e:
            # Rate limiting/circuit breaker feedback
            await self.rate_limiter.record_failure()
            self.circuit_breaker.record_failure()
            
            logger.error(f"OpenAI API status error for request {request_id}: Status={e.status_code}, Response={e.response.text if e.response else 'N/A'}")
            retryable = e.status_code in [429, 500, 502, 503, 504]
            error_details = self._map_error(e, retryable=retryable)
        except APIError as e:
            # Rate limiting/circuit breaker feedback
            await self.rate_limiter.record_failure()
            self.circuit_breaker.record_failure()
            
            logger.error(f"OpenAI API error for request {request_id}: {e}")
            error_details = self._map_error(e, retryable=False)
        except Exception as e:
            # Record failure for circuit breaker
            self.circuit_breaker.record_failure()
            
            logger.error(f"Unexpected error for OpenAI request {request_id}: {e}")
            error_details = self._map_error(e, retryable=False)

        total_duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        # 3. Map the response or error to the Gateway format
        return self._map_response(
            response_obj,
            request,
            error_details,
            llm_latency_ms,
            total_duration_ms,
        )

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate a streaming response from OpenAI using the Chat Completions API.
        Yields gateway StreamChunk objects.
        
        Args:
            request: The LLM request to process
            
        Yields:
            StreamChunk: Chunks of the streaming response
        """
        request_id = request.request_id
        chunk_index = 0
        accumulated_content = ""
        final_usage: Optional[UsageStats] = None
        final_finish_reason: Optional[FinishReason] = None

        if not self._client:
            logger.error(f"OpenAI client not initialized for streaming request {request_id}")
            yield self._create_error_chunk(
                request_id, 0, self._map_error(ConnectionError("OpenAI client not initialized."))
            )
            return

        try:
            # Apply rate limiting
            await self.rate_limiter.wait()
            
            # 1. Map request
            model_id = request.model_identifier
            openai_params = self._map_request(request, model_id)
            openai_params["stream"] = True
            # stream_options needed for usage in stream (Azure supports, OpenAI might beta)
            openai_params["stream_options"] = {"include_usage": True}

            logger.debug(f"Starting stream request to OpenAI model '{model_id}'")

            # 2. Call OpenAI streaming API
            stream = await self._client.chat.completions.create(**openai_params)
            
            # Record success to circuit breaker once we start getting responses
            first_chunk_received = False

            async for chunk in stream:
                if not first_chunk_received:
                    first_chunk_received = True
                    self.circuit_breaker.record_success()
                
                mapped_chunk = None
                delta = None
                finish_reason_str = None

                if chunk.choices:
                    delta = chunk.choices[0].delta
                    finish_reason_str = chunk.choices[0].finish_reason

                # Check for usage in the chunk (supported by Azure, maybe OpenAI beta)
                chunk_usage = getattr(chunk, 'usage', None)
                if chunk_usage:
                    final_usage = self._map_usage(chunk_usage)  # Overwrite with latest usage chunk

                if delta:
                    # -- Text Delta --
                    if delta.content:
                        accumulated_content += delta.content
                        mapped_chunk = StreamChunk(
                            index=chunk_index,
                            request_id=request_id,
                            content=delta.content,
                        )

                # -- Finish Reason --
                if finish_reason_str:
                    final_finish_reason = self._map_finish_reason(finish_reason_str)

                    # Create final chunk with finish reason and usage (if available)
                    mapped_chunk = StreamChunk(
                        index=chunk_index,
                        request_id=request_id,
                        finish_reason=final_finish_reason,
                        usage=final_usage,  # Might be None
                    )

                # Yield the mapped chunk if one was created
                if mapped_chunk:
                    yield mapped_chunk
                    chunk_index += 1
            
            # Record success to rate limiter
            await self.rate_limiter.record_success()

        except (APITimeoutError, asyncio.TimeoutError) as e:
            # Rate limiting/circuit breaker feedback
            await self.rate_limiter.record_failure()
            self.circuit_breaker.record_failure()
            
            logger.error(f"OpenAI stream timed out for request {request_id}: {e}")
            yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, timeout=self._timeout, retryable=True))
        except RateLimitError as e:
            # Rate limiting/circuit breaker feedback
            await self.rate_limiter.record_failure()
            self.circuit_breaker.record_failure()
            
            logger.error(f"OpenAI stream rate limit for request {request_id}: {e}")
            retry_after = self._get_retry_after(e)
            yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=True, retry_after=retry_after))
        except AuthenticationError as e:
            # Record failure for circuit breaker
            self.circuit_breaker.record_failure()
            
            logger.error(f"OpenAI stream authentication error for request {request_id}: {e}")
            yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=False))
        except PermissionDeniedError as e:
            # Record failure for circuit breaker
            self.circuit_breaker.record_failure()
            
            logger.error(f"OpenAI stream permission denied for request {request_id}: {e}")
            yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=False))
        except BadRequestError as e:
            # Record failure for circuit breaker
            self.circuit_breaker.record_failure()
            
            logger.error(f"OpenAI stream bad request error for request {request_id}: {e}")
            yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=False))
        except APIStatusError as e:
            # Rate limiting/circuit breaker feedback
            await self.rate_limiter.record_failure()
            self.circuit_breaker.record_failure()
            
            logger.error(f"OpenAI stream API error for request {request_id}: Status={e.status_code}, Response={e.response.text if e.response else 'N/A'}")
            retryable = e.status_code in [429, 500, 502, 503, 504]
            yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=retryable))
        except APIError as e:  # Catch-all for other OpenAI API errors
            # Rate limiting/circuit breaker feedback
            await self.rate_limiter.record_failure()
            self.circuit_breaker.record_failure()
            
            logger.error(f"OpenAI stream API error for request {request_id}: {e}")
            yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=False))
        except Exception as e:
            # Record failure for circuit breaker
            self.circuit_breaker.record_failure()
            
            logger.error(f"Unexpected error during OpenAI stream for request {request_id}: {e}")
            yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=False))

        finally:
            logger.debug(f"OpenAI stream finished for request {request_id}.")
    
    # --- Utility methods for request/response mapping ---
    
    def _map_request(self, request: LLMRequest, model_id: str) -> Dict[str, Any]:
        """
        Map Gateway Request to OpenAI API format.
        
        Args:
            request: The LLM request to process
            model_id: The model identifier
            
        Returns:
            OpenAI API request parameters
        """
        # If using Azure, we need to use deployment_id instead of model
        model_param_name = "deployment_id" if self.is_azure else "model"
        
        # Build the OpenAI request parameters
        openai_params = {
            model_param_name: model_id,
            "messages": self._map_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }
        
        # Add optional parameters if they are set
        if request.stop_sequences:
            openai_params["stop"] = request.stop_sequences
        
        if request.presence_penalty is not None:
            openai_params["presence_penalty"] = request.presence_penalty
            
        if request.frequency_penalty is not None:
            openai_params["frequency_penalty"] = request.frequency_penalty
            
        # Add tools if present
        if request.tools:
            openai_params["tools"] = self._map_tools(request.tools)
        
        # Add other parameters from extra_params
        if request.extra_params:
            for key, value in request.extra_params.items():
                if key not in openai_params and key != "stream":
                    openai_params[key] = value
        
        return openai_params
    
    def _map_messages(self, request: LLMRequest) -> List[Dict[str, Any]]:
        """
        Map Gateway Request messages to OpenAI message format.
        
        Args:
            request: The LLM request to process
            
        Returns:
            List of OpenAI formatted messages
        """
        # For simplicity, assume request.messages contains messages
        # in the correct format for OpenAI (system, user, assistant)
        # This should be expanded to properly map your internal message format
        return request.messages
    
    def _map_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map Gateway tools to OpenAI tool format.
        
        Args:
            tools: List of tools to map
            
        Returns:
            List of OpenAI formatted tools
        """
        # For simplicity, assume tools are already in the correct format
        # This should be expanded to properly map your internal tool format
        return tools
    
    def _map_response(
        self,
        response_obj: Optional[ChatCompletion],
        request: LLMRequest,
        error_details: Optional[ErrorDetails],
        llm_latency_ms: Optional[float],
        total_duration_ms: float
    ) -> LLMResponse:
        """
        Map OpenAI response to Gateway response format.
        
        Args:
            response_obj: The OpenAI response object
            request: The original LLM request
            error_details: Optional error details
            llm_latency_ms: Time taken by the LLM API call
            total_duration_ms: Total time taken for processing
            
        Returns:
            LLMResponse: The formatted response
        """
        if error_details:
            # Return an error response
            return LLMResponse(
                request_id=request.request_id,
                provider_id=self.provider_id,
                model_id=request.model_identifier,
                created_at=datetime.now(timezone.utc),
                latency_ms=total_duration_ms,
                llm_latency_ms=llm_latency_ms,
                error=error_details
            )
        
        if not response_obj or not response_obj.choices or not response_obj.choices[0].message:
            # Return a generic error if we somehow got here without a proper response
            return LLMResponse(
                request_id=request.request_id,
                provider_id=self.provider_id,
                model_id=request.model_identifier,
                created_at=datetime.now(timezone.utc),
                latency_ms=total_duration_ms,
                llm_latency_ms=llm_latency_ms,
                error=ErrorDetails(
                    message="No response from model",
                    code="NO_RESPONSE",
                    provider_id=self.provider_id,
                    retryable=True
                )
            )
        
        # Get the primary choice
        choice = response_obj.choices[0]
        message = choice.message
        
        # Map usage
        usage = self._map_usage(response_obj.usage) if response_obj.usage else None
        
        # Map finish reason
        finish_reason = self._map_finish_reason(choice.finish_reason) if choice.finish_reason else None
        
        # Build the response
        return LLMResponse(
            request_id=request.request_id,
            provider_id=self.provider_id,
            model_id=request.model_identifier,
            content=message.content,
            created_at=datetime.now(timezone.utc),
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=total_duration_ms,
            llm_latency_ms=llm_latency_ms,
            tool_calls=self._map_tool_calls(message.tool_calls) if message.tool_calls else None
        )
    
    def _map_usage(self, usage: CompletionUsage) -> UsageStats:
        """
        Map OpenAI usage to Gateway usage format.
        
        Args:
            usage: OpenAI usage statistics
            
        Returns:
            UsageStats: Gateway usage statistics
        """
        return UsageStats(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens
        )
    
    def _map_finish_reason(self, finish_reason: str) -> FinishReason:
        """
        Map OpenAI finish reason to Gateway finish reason.
        
        Args:
            finish_reason: OpenAI finish reason string
            
        Returns:
            FinishReason: Gateway finish reason enum value
        """
        if finish_reason == "stop":
            return FinishReason.STOP
        elif finish_reason == "length":
            return FinishReason.LENGTH
        elif finish_reason == "content_filter":
            return FinishReason.CONTENT_FILTER
        elif finish_reason == "tool_calls":
            return FinishReason.FUNCTION_CALL
        else:
            return FinishReason.UNKNOWN
    
    def _map_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """
        Map OpenAI tool calls to Gateway format.
        
        Args:
            tool_calls: OpenAI tool calls
            
        Returns:
            List of Gateway formatted tool calls
        """
        # For simplicity, assume a basic mapping for now
        # This should be expanded to properly map tool calls
        result = []
        for tool_call in tool_calls:
            if tool_call.type == "function":
                result.append({
                    "type": "function",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                })
        return result
    
    def _map_error(
        self,
        error: Exception,
        timeout: Optional[int] = None,
        retryable: bool = False,
        retry_after: Optional[int] = None
    ) -> ErrorDetails:
        """
        Map OpenAI error to Gateway error format.
        
        Args:
            error: The exception to map
            timeout: Optional timeout value if it's a timeout error
            retryable: Whether this error is retryable
            retry_after: Optional seconds to wait before retrying
            
        Returns:
            ErrorDetails: Gateway error details
        """
        # Map common OpenAI errors to Gateway error codes
        if isinstance(error, RateLimitError):
            return ErrorDetails(
                message=f"Rate limit exceeded: {str(error)}",
                code="RATE_LIMIT_EXCEEDED",
                provider_id=self.provider_id,
                retryable=True,
                retry_after=retry_after
            )
        elif isinstance(error, (APITimeoutError, asyncio.TimeoutError)):
            return ErrorDetails(
                message=f"Request timed out after {timeout}s",
                code="TIMEOUT",
                provider_id=self.provider_id,
                retryable=True
            )
        elif isinstance(error, AuthenticationError):
            return ErrorDetails(
                message="Authentication failed",
                code="AUTHENTICATION_ERROR",
                provider_id=self.provider_id,
                retryable=False
            )
        elif isinstance(error, PermissionDeniedError):
            return ErrorDetails(
                message="Permission denied",
                code="PERMISSION_DENIED",
                provider_id=self.provider_id,
                retryable=False
            )
        elif isinstance(error, BadRequestError):
            return ErrorDetails(
                message=f"Bad request: {str(error)}",
                code="BAD_REQUEST",
                provider_id=self.provider_id,
                retryable=False
            )
        elif isinstance(error, APIStatusError):
            message = str(error)
            try:
                if error.response:
                    message = f"Status {error.status_code}: {error.response.text}"
            except:
                pass
            
            return ErrorDetails(
                message=message,
                code=f"API_ERROR_{error.status_code}" if getattr(error, 'status_code', None) else "API_ERROR",
                provider_id=self.provider_id,
                retryable=retryable
            )
        elif isinstance(error, APIError):
            return ErrorDetails(
                message=str(error),
                code="API_ERROR",
                provider_id=self.provider_id,
                retryable=retryable
            )
        else:
            return ErrorDetails(
                message=str(error),
                code="UNKNOWN_ERROR",
                provider_id=self.provider_id,
                retryable=retryable
            )
    
    def _create_error_chunk(self, request_id: str, chunk_index: int, error: ErrorDetails) -> StreamChunk:
        """
        Create a streaming error chunk.
        
        Args:
            request_id: The request ID
            chunk_index: The chunk index
            error: Error details
            
        Returns:
            StreamChunk: An error chunk
        """
        return StreamChunk(
            index=chunk_index,
            request_id=request_id,
            error=error
        )
    
    def _get_retry_after(self, error: Exception) -> Optional[int]:
        """
        Extract retry-after header value from OpenAI error.
        
        Args:
            error: The OpenAI exception
            
        Returns:
            Optional retry after seconds
        """
        # Try to extract retry-after value from headers
        retry_after = None
        try:
            if hasattr(error, 'response') and error.response and error.response.headers:
                retry_after_header = error.response.headers.get('retry-after')
                if retry_after_header:
                    retry_after = int(retry_after_header)
        except:
            pass
        
        # Default retry after for rate limits if not specified
        if retry_after is None and isinstance(error, RateLimitError):
            retry_after = 30
            
        return retry_after
    
    # --- Health check implementation ---
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the OpenAI provider.
        
        Returns:
            Dict with health check results
        """
        start_time = time.time()
        is_healthy = False
        error_message = None
        
        try:
            # Try listing models as a basic health check
            if self._client:
                # Just get one model to verify connectivity
                await self._client.models.list(limit=1)
                is_healthy = True
                
                # Also check circuit breaker state
                if self.circuit_breaker.is_open():
                    is_healthy = False
                    error_message = f"Circuit breaker is open: {self.circuit_breaker.name}"
            else:
                error_message = "OpenAI client is not initialized"
                
        except Exception as e:
            error_message = f"Health check failed: {str(e)}"
            
        duration_ms = (time.time() - start_time) * 1000
        
        return {
            "provider_id": self.provider_id,
            "provider_type": "openai",
            "is_healthy": is_healthy,
            "latency_ms": duration_ms,
            "error": error_message if error_message else None,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "rate_limiter_stats": self.rate_limiter.get_stats()
        }


def create_resilient_openai_provider(provider_config: ProviderConfig) -> OpenAIClient:
    """
    Create a resilient OpenAI provider.
    
    This function creates an OpenAI provider with resilience patterns
    like circuit breaking, retries, and rate limiting.
    
    Args:
        provider_config: Provider configuration
        
    Returns:
        OpenAI provider with resilience patterns
    """
    # Create the base OpenAI client
    client = OpenAIClient(provider_config)
    
    # Wrap it with resilient provider wrapper (for additional safety/monitoring)
    # Note: The OpenAIClient already has internal resilience patterns
    wrapper = ResilientProviderWrapper(
        provider=client,
        provider_id=provider_config.provider_id
    )
    
    # Return the original client as it already has resilience built-in
    # We created the wrapper for better monitoring, but will use the client directly
    return client