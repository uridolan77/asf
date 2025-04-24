"""
Enhanced OpenAI client implementation for the LLM Gateway.

This module provides a comprehensive OpenAI client class that handles all
major OpenAI API endpoints, including:
- Chat Completions
- Embeddings
- Audio (speech, transcription, translation)
- Images
- Assistants
- Files
- Vector stores
- Models
- Moderations

Supports both standard OpenAI and Azure OpenAI endpoints.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union, cast, BinaryIO

# OpenAI SDK Imports
import openai
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.embedding import Embedding
from openai.types.audio import (
    Transcription, Translation, Speech
)
from openai.types.image import Image as OpenAIImage
from openai.types.file_object import FileObject
from openai.types.model import Model
from openai.types.moderation import (
    ModerationCreateResponse
)
from openai.types.assistant import Assistant
from openai.types.vector_store import VectorStore
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
from asf.medical.llm_gateway.core.models import (
    ErrorDetails,
    FinishReason,
    GatewayConfig,
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    StreamChunk,
    UsageStats,
)

# Local imports from the openai package
from asf.medical.llm_gateway.providers.openai.config import get_api_key, get_organization_id
from asf.medical.llm_gateway.providers.openai.errors import map_error, create_error_chunk, get_retry_after
from asf.medical.llm_gateway.providers.openai.mappings import (
    map_request,
    map_response,
    map_finish_reason,
    map_usage,
    map_tool_calls,
    map_tool_call_deltas,
)

# Import the base provider class
from asf.medical.llm_gateway.providers.base import BaseProvider

logger = logging.getLogger(__name__)

class OpenAIClient(BaseProvider):
    """
    LLM Gateway provider for OpenAI models supporting the full OpenAI API.
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
        self._client: Optional[Union[AsyncOpenAI, AsyncAzureOpenAI]] = None
        self.is_azure = provider_config.connection_params.get("is_azure", False)

        # Get retry/timeout settings from provider_config or fallback to gateway_config
        self._max_retries = provider_config.connection_params.get("max_retries", gateway_config.max_retries)
        self._timeout = provider_config.connection_params.get("timeout_seconds", gateway_config.default_timeout_seconds)
        self._retry_delay = provider_config.connection_params.get("retry_delay_seconds", gateway_config.retry_delay_seconds)

        try:
            if self.is_azure:
                self._initialize_azure_client(provider_config)
            else:
                self._initialize_openai_client(provider_config)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI SDK client for provider {self.provider_id}: {e}", exc_info=True)
            raise ConnectionError(f"OpenAI SDK client initialization failed: {e}") from e

    def _initialize_openai_client(self, provider_config: ProviderConfig):
        """
        Initialize the standard OpenAI client.
        
        Args:
            provider_config: Provider configuration with connection parameters
        """
        # Get API key from appropriate source
        api_key = get_api_key(provider_config, self.provider_id)
        
        # Get organization ID if available
        org_id = get_organization_id(provider_config)
        
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
        
        # Get API key from the various sources
        api_key = get_api_key(provider_config, self.provider_id)
        
        # Get required Azure parameters
        azure_endpoint = connection_params.get("azure_endpoint")
        if not azure_endpoint:
            # Try environment variable
            import os
            azure_endpoint_env_var = connection_params.get("endpoint_env_var", "AZURE_OPENAI_ENDPOINT")
            azure_endpoint = os.environ.get(azure_endpoint_env_var)
            
        api_version = connection_params.get("api_version")
        if not api_version:
            # Try environment variable
            import os
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
                logger.warning(f"Error during OpenAI client cleanup for '{self.provider_id}': {e}", exc_info=True)
        self._client = None
    
    # --- CHAT COMPLETIONS API ---
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response from OpenAI using the non-streaming Chat Completions API.
        
        Args:
            request: The LLM request to process
            
        Returns:
            LLMResponse: The response from OpenAI
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

        if not self._client:
            logger.error(f"[OPENAI_ERROR] ID: {request_id} | OpenAI client not initialized")
            raise ConnectionError("OpenAI client is not initialized.")

        try:
            # 1. Map Gateway Request to OpenAI API format
            # Azure requires deployment_id which is often the model name in the request
            logger.debug(f"[OPENAI_REQUEST_MAPPING] ID: {request_id} | Mapping request to OpenAI format")
            openai_params = map_request(request, model_id)

            # Log prompt content length for debugging (not the actual content for privacy)
            msg_count = len(openai_params.get("messages", []))
            prompt_len = sum(len(str(msg.get("content", ""))) for msg in openai_params.get("messages", []))
            tool_count = len(openai_params.get("tools", []))
            logger.info(f"[OPENAI_REQUEST_STATS] ID: {request_id} | MessageCount: {msg_count} | PromptLength: {prompt_len} | ToolCount: {tool_count}")

            # 2. Call OpenAI API
            logger.info(f"[OPENAI_API_CALL] ID: {request_id} | Sending request to {self.is_azure and 'Azure ' or ''}OpenAI model '{model_id}'")
            llm_call_start = datetime.now(timezone.utc)

            response_obj = await self._client.chat.completions.create(**openai_params)

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
            error_details = map_error(e, self.provider_id, timeout=self._timeout, retryable=True)
        except RateLimitError as e:
            logger.error(f"[OPENAI_RATE_LIMIT] ID: {request_id} | Rate limit exceeded: {e}", exc_info=True)
            retry_after = get_retry_after(e)
            error_details = map_error(e, self.provider_id, retryable=True, retry_after=retry_after)
            if retry_after:
                logger.info(f"[OPENAI_RETRY_AFTER] ID: {request_id} | Retry suggested after {retry_after} seconds")
        except AuthenticationError as e:
            logger.error(f"[OPENAI_AUTH_ERROR] ID: {request_id} | Authentication failed: {e}", exc_info=True)
            error_details = map_error(e, self.provider_id, retryable=False)  # Not retryable
        except PermissionDeniedError as e:
            logger.error(f"OpenAI permission denied for request {request_id}: {e}", exc_info=True)
            error_details = map_error(e, self.provider_id, retryable=False)
        except BadRequestError as e:
            logger.error(f"OpenAI bad request for request {request_id}: {e.code} - {e.message}", exc_info=True)
            error_details = map_error(e, self.provider_id, retryable=False)
        except APIStatusError as e:
            logger.error(f"OpenAI API status error for request {request_id}: Status={e.status_code}, Response={e.response.text if e.response else 'N/A'}", exc_info=True)
            retryable = e.status_code in [429, 500, 502, 503, 504]
            error_details = map_error(e, self.provider_id, retryable=retryable)
        except APIError as e:
            logger.error(f"OpenAI API error for request {request_id}: {e}", exc_info=True)
            error_details = map_error(e, self.provider_id, retryable=False)
        except Exception as e:
            logger.error(f"Unexpected error for OpenAI request {request_id}: {e}", exc_info=True)
            error_details = map_error(e, self.provider_id, retryable=False)

        total_duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        # 3. Map the response or error to the Gateway format
        return map_response(
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
        request_id = request.initial_context.request_id
        chunk_index = 0
        accumulated_content = ""
        accumulated_tool_calls: Dict[int, Any] = {}  # Accumulate tool call deltas by index
        final_usage: Optional[UsageStats] = None  # OpenAI doesn't stream usage typically
        final_finish_reason: Optional[FinishReason] = None

        if not self._client:
            logger.error(f"OpenAI client not initialized for streaming request {request_id}")
            yield create_error_chunk(
                request_id, 0, map_error(ConnectionError("OpenAI client not initialized."), self.provider_id)
            )
            return

        try:
            # 1. Map request
            model_id = request.config.model_identifier
            openai_params = map_request(request, model_id)
            openai_params["stream"] = True
            # stream_options needed for usage in stream (Azure supports, OpenAI might beta)
            openai_params["stream_options"] = {"include_usage": True}

            logger.debug(f"Starting stream request to OpenAI model '{model_id}'")

            # 2. Call OpenAI streaming API
            stream = await self._client.chat.completions.create(**openai_params)

            async for chunk in stream:
                # Time tracking for chunk processing if needed
                # chunk_start_time = datetime.now(timezone.utc)
                mapped_chunk = None
                delta = None
                finish_reason_str = None

                if chunk.choices:
                    delta = chunk.choices[0].delta
                    finish_reason_str = chunk.choices[0].finish_reason

                # Check for usage in the chunk (supported by Azure, maybe OpenAI beta)
                chunk_usage = getattr(chunk, 'usage', None)
                if chunk_usage:
                    final_usage = map_usage(chunk_usage)  # Overwrite with latest usage chunk

                if delta:
                    # -- Text Delta --
                    if delta.content:
                        accumulated_content += delta.content
                        mapped_chunk = StreamChunk(
                            chunk_id=chunk_index,
                            request_id=request_id,
                            delta_text=delta.content,
                        )

                    # -- Tool Call Delta --
                    if delta.tool_calls:
                        # Accumulate tool call chunks
                        for tool_call_delta in delta.tool_calls:
                            index = tool_call_delta.index
                            if index not in accumulated_tool_calls:
                                # Start of a new tool call
                                accumulated_tool_calls[index] = tool_call_delta
                            else:
                                # Append function arguments chunk
                                if tool_call_delta.function and tool_call_delta.function.arguments:
                                    existing_args = accumulated_tool_calls[index].function.arguments or ""
                                    accumulated_tool_calls[index].function.arguments = existing_args + tool_call_delta.function.arguments
                                # Update other fields if necessary (id, name, type usually come first)
                                if tool_call_delta.id:
                                    accumulated_tool_calls[index].id = tool_call_delta.id
                                if tool_call_delta.type:
                                    accumulated_tool_calls[index].type = tool_call_delta.type
                                if tool_call_delta.function and tool_call_delta.function.name:
                                    accumulated_tool_calls[index].function.name = tool_call_delta.function.name

                        # We only yield the *complete* tool call info when the finish reason arrives.
                        # Alternatively, yield deltas as they come? Less useful for gateway.
                        # Let's yield a placeholder chunk indicating tool activity if needed.
                        # mapped_chunk = StreamChunk(...) # Indicate tool delta without full info

                # -- Finish Reason --
                if finish_reason_str:
                    final_finish_reason = map_finish_reason(finish_reason_str)
                    # If finished due to tool calls, map the accumulated calls now
                    final_tool_requests = None
                    if final_finish_reason == FinishReason.TOOL_CALLS:
                        final_tool_requests = map_tool_call_deltas(accumulated_tool_calls, request.tools)

                    # Create final chunk with finish reason, usage (if available), and completed tool calls
                    mapped_chunk = StreamChunk(
                        chunk_id=chunk_index,
                        request_id=request_id,
                        finish_reason=final_finish_reason,
                        usage_update=final_usage,  # Might be None
                        delta_tool_calls=final_tool_requests,  # Attach final tools here
                        # Optionally add final accumulated text if finish reason != tool_calls?
                        # delta_text=accumulated_content if final_finish_reason != FinishReason.TOOL_CALLS else None
                    )
                    # Clear accumulated state after final chunk
                    accumulated_content = ""
                    accumulated_tool_calls = {}

                # Yield the mapped chunk if one was created
                if mapped_chunk:
                    yield mapped_chunk
                    chunk_index += 1

        except (APITimeoutError, asyncio.TimeoutError) as e:
            logger.error(f"OpenAI stream timed out for request {request_id}: {e}", exc_info=True)
            yield create_error_chunk(request_id, chunk_index, map_error(e, self.provider_id, timeout=self._timeout, retryable=True))
        except RateLimitError as e:
            logger.error(f"OpenAI stream rate limit for request {request_id}: {e}", exc_info=True)
            retry_after = get_retry_after(e)
            yield create_error_chunk(request_id, chunk_index, map_error(e, self.provider_id, retryable=True, retry_after=retry_after))
        except AuthenticationError as e:
            logger.error(f"OpenAI stream authentication error for request {request_id}: {e}", exc_info=True)
            yield create_error_chunk(request_id, chunk_index, map_error(e, self.provider_id, retryable=False))
        except PermissionDeniedError as e:
            logger.error(f"OpenAI stream permission denied for request {request_id}: {e}", exc_info=True)
            yield create_error_chunk(request_id, chunk_index, map_error(e, self.provider_id, retryable=False))
        except BadRequestError as e:
            logger.error(f"OpenAI stream bad request error for request {request_id}: {e.code} - {e.message}", exc_info=True)
            error = map_error(e, self.provider_id, retryable=False)
            if e.code == 'content_filter':
                error.code = "PROVIDER_CONTENT_FILTER"
            yield create_error_chunk(request_id, chunk_index, error)
        except APIStatusError as e:
            logger.error(f"OpenAI stream API error for request {request_id}: Status={e.status_code}, Response={e.response.text if e.response else 'N/A'}", exc_info=True)
            retryable = e.status_code in [429, 500, 502, 503, 504]
            yield create_error_chunk(request_id, chunk_index, map_error(e, self.provider_id, retryable=retryable))
        except APIError as e:  # Catch-all for other OpenAI API errors
            logger.error(f"OpenAI stream API error for request {request_id}: {e}", exc_info=True)
            yield create_error_chunk(request_id, chunk_index, map_error(e, self.provider_id, retryable=False))
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI stream for request {request_id}: {e}", exc_info=True)
            yield create_error_chunk(request_id, chunk_index, map_error(e, self.provider_id, retryable=False))

        finally:
            logger.debug(f"OpenAI stream finished for request {request_id}.")
    
    # --- EMBEDDINGS API ---
    
    async def create_embeddings(self, texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
        """
        Create embeddings for the provided texts using the OpenAI Embeddings API.
        
        Args:
            texts: List of texts to generate embeddings for
            model: Embedding model to use, defaults to text-embedding-ada-002
            
        Returns:
            List of embedding vectors
        """
        if not self._client:
            raise ConnectionError("OpenAI client is not initialized")
        
        logger.info(f"Creating embeddings for {len(texts)} texts using model {model}")
        start_time = datetime.now(timezone.utc)
        
        try:
            # Azure requires deployment_id parameter instead of model for embeddings
            if self.is_azure:
                response = await self._client.embeddings.create(
                    input=texts,
                    deployment_id=model,
                )
            else:
                response = await self._client.embeddings.create(
                    input=texts,
                    model=model,
                )
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Generated {len(response.data)} embeddings in {duration_ms:.2f}ms")
            
            # Extract embeddings data from response
            embeddings = [item.embedding for item in response.data]
            
            # Log usage statistics if available
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                logger.debug(f"Embeddings usage - Prompt tokens: {usage.prompt_tokens}, Total tokens: {usage.total_tokens}")
            
            return embeddings
            
        except Exception as e:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.error(f"Error creating embeddings: {e} after {duration_ms:.2f}ms", exc_info=True)
            raise
    
    # --- AUDIO API ---
    
    async def create_speech(self, text: str, voice: str = "alloy", model: str = "tts-1", 
                     format: str = "mp3", speed: float = 1.0) -> bytes:
        """
        Generate speech from the given text using OpenAI's Text-to-Speech API.
        
        Args:
            text: The text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS model to use (tts-1, tts-1-hd)
            format: Audio format (mp3, opus, aac, flac)
            speed: Speech speed multiplier (0.25 to 4.0)
            
        Returns:
            The generated audio as bytes
        """
        if not self._client:
            raise ConnectionError("OpenAI client is not initialized")
        
        if self.is_azure:
            # Currently Azure doesn't support the speech API, fallback to OpenAI directly
            logger.warning("Speech generation not supported on Azure OpenAI, contact your administrator for a standard OpenAI provider setup")
            raise NotImplementedError("Speech generation not supported via Azure OpenAI")
        
        logger.info(f"Creating speech for '{text[:30]}...' using voice={voice}, model={model}")
        start_time = datetime.now(timezone.utc)
        
        try:
            response = await self._client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=format,
                speed=speed,
            )
            
            # Get the binary content
            audio_data = response.content
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Generated {len(audio_data)} bytes of audio in {duration_ms:.2f}ms")
            
            return audio_data
            
        except Exception as e:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.error(f"Error creating speech: {e} after {duration_ms:.2f}ms", exc_info=True)
            raise
    
    async def create_transcription(self, audio_file: BinaryIO, model: str = "whisper-1", 
                           language: Optional[str] = None, prompt: Optional[str] = None) -> str:
        """
        Transcribe audio to text using OpenAI's Whisper model.
        
        Args:
            audio_file: Audio file object to transcribe
            model: Whisper model to use, defaults to whisper-1
            language: Optional language code (e.g., "en", "fr")
            prompt: Optional prompt to guide the transcription
            
        Returns:
            The transcribed text
        """
        if not self._client:
            raise ConnectionError("OpenAI client is not initialized")
        
        if self.is_azure:
            # Check if Azure supports transcription in your API version
            # Currently Azure may not support audio API (check your API version)
            logger.warning("Transcription may not be supported on Azure OpenAI, fallback to standard OpenAI provider if this fails")
        
        try:
            logger.info(f"Transcribing audio using model={model}")
            start_time = datetime.now(timezone.utc)
            
            # Prepare parameters
            params = {
                "file": audio_file,
                "model": model,
            }
            
            # Add optional parameters if provided
            if language:
                params["language"] = language
            if prompt:
                params["prompt"] = prompt
                
            # For Azure, may need deployment_id instead of model
            if self.is_azure:
                params.pop("model", None)
                params["deployment_id"] = model
            
            # Create transcription
            response = await self._client.audio.transcriptions.create(**params)
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Transcribed audio in {duration_ms:.2f}ms")
            
            # Extract text from response
            # Depending on the OpenAI SDK version, this might be accessed differently
            if hasattr(response, "text"):
                return response.text
            else:
                # Handle older SDK versions or alternative response formats
                return response.get("text", str(response))
            
        except Exception as e:
            logger.error(f"Error creating transcription: {e}", exc_info=True)
            raise
    
    async def create_translation(self, audio_file: BinaryIO, model: str = "whisper-1", 
                          prompt: Optional[str] = None) -> str:
        """
        Translate audio to English text using OpenAI's Whisper model.
        
        Args:
            audio_file: Audio file object to translate
            model: Whisper model to use, defaults to whisper-1
            prompt: Optional prompt to guide the translation
            
        Returns:
            The translated text in English
        """
        if not self._client:
            raise ConnectionError("OpenAI client is not initialized")
        
        if self.is_azure:
            # Check if Azure supports translation in your API version
            logger.warning("Translation may not be supported on Azure OpenAI, fallback to standard OpenAI provider if this fails")
        
        try:
            logger.info(f"Translating audio to English using model={model}")
            start_time = datetime.now(timezone.utc)
            
            # Prepare parameters
            params = {
                "file": audio_file,
                "model": model,
            }
            
            # Add optional prompt if provided
            if prompt:
                params["prompt"] = prompt
                
            # For Azure, may need deployment_id instead of model
            if self.is_azure:
                params.pop("model", None)
                params["deployment_id"] = model
            
            # Create translation
            response = await self._client.audio.translations.create(**params)
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Translated audio in {duration_ms:.2f}ms")
            
            # Extract text from response
            if hasattr(response, "text"):
                return response.text
            else:
                # Handle older SDK versions or alternative response formats
                return response.get("text", str(response))
            
        except Exception as e:
            logger.error(f"Error creating translation: {e}", exc_info=True)
            raise
    
    # --- IMAGES API ---
    
    async def create_image(self, prompt: str, model: str = "dall-e-3", n: int = 1, 
                    size: str = "1024x1024", quality: str = "standard", 
                    style: str = "vivid") -> List[Dict[str, Any]]:
        """
        Generate images from a text prompt using DALL-E models.
        
        Args:
            prompt: Text description of the desired image
            model: Image model to use (dall-e-2, dall-e-3)
            n: Number of images to generate (1-10 for DALL-E 2, 1 for DALL-E 3)
            size: Image size (256x256, 512x512, 1024x1024 for DALL-E 2, 
                              1024x1024, 1792x1024, 1024x1792 for DALL-E 3)
            quality: Image quality (standard, hd) - DALL-E 3 only
            style: Image style (vivid, natural) - DALL-E 3 only
            
        Returns:
            List of generated image data with URLs and metadata
        """
        if not self._client:
            raise ConnectionError("OpenAI client is not initialized")
        
        if self.is_azure:
            # Check if Azure supports DALL-E in your API version
            logger.warning("Image generation may not be supported on Azure OpenAI, fallback to standard OpenAI provider if this fails")
        
        try:
            logger.info(f"Creating {n} images with prompt: '{prompt[:50]}...' using model={model}")
            start_time = datetime.now(timezone.utc)
            
            # Prepare parameters based on the model
            params = {
                "model": model,
                "prompt": prompt,
                "n": n,
                "size": size,
            }
            
            # DALL-E 3 specific parameters
            if model == "dall-e-3":
                params["quality"] = quality
                params["style"] = style
                # DALL-E 3 only supports n=1
                params["n"] = 1
                
            # For Azure, may need deployment_id instead of model
            if self.is_azure:
                params.pop("model", None)
                params["deployment_id"] = model
            
            # Create images
            response = await self._client.images.generate(**params)
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Generated {len(response.data)} images in {duration_ms:.2f}ms")
            
            # Convert response to a list of dicts with image URLs and metadata
            result = []
            for img in response.data:
                image_data = {
                    "url": img.url,
                    "revised_prompt": getattr(img, "revised_prompt", None),
                }
                result.append(image_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating images: {e}", exc_info=True)
            raise
    
    async def create_image_variation(self, image_file: BinaryIO, model: str = "dall-e-2", 
                              n: int = 1, size: str = "1024x1024") -> List[Dict[str, str]]:
        """
        Create variations of an image using DALL-E models.
        
        Args:
            image_file: Image file object to create variations from
            model: Image model to use (typically dall-e-2)
            n: Number of variations to generate (1-10)
            size: Image size (256x256, 512x512, 1024x1024)
            
        Returns:
            List of generated image variations with URLs
        """
        if not self._client:
            raise ConnectionError("OpenAI client is not initialized")
        
        if self.is_azure:
            # Check if Azure supports image variations in your API version
            logger.warning("Image variations may not be supported on Azure OpenAI, fallback to standard OpenAI provider if this fails")
        
        try:
            logger.info(f"Creating {n} image variations using model={model}")
            start_time = datetime.now(timezone.utc)
            
            # Prepare parameters
            params = {
                "image": image_file,
                "n": n,
                "size": size,
            }
            
            # Model parameter is not used in current OpenAI SDK for variations
            # but kept for future compatibility
            
            # Create image variations
            response = await self._client.images.create_variation(**params)
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Generated {len(response.data)} image variations in {duration_ms:.2f}ms")
            
            # Convert response to a list of dicts with image URLs
            result = [{"url": img.url} for img in response.data]
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating image variations: {e}", exc_info=True)
            raise
    
    async def create_image_edit(self, image_file: BinaryIO, prompt: str, mask: Optional[BinaryIO] = None,
                         model: str = "dall-e-2", n: int = 1, size: str = "1024x1024") -> List[Dict[str, str]]:
        """
        Edit an image based on a prompt using DALL-E models.
        
        Args:
            image_file: Image file object to edit
            prompt: Text description of the desired edits
            mask: Optional mask file object to specify edit areas (transparent areas will be edited)
            model: Image model to use (typically dall-e-2)
            n: Number of edits to generate (1-10)
            size: Image size (256x256, 512x512, 1024x1024)
            
        Returns:
            List of generated edited images with URLs
        """
        if not self._client:
            raise ConnectionError("OpenAI client is not initialized")
        
        if self.is_azure:
            # Check if Azure supports image edits in your API version
            logger.warning("Image edits may not be supported on Azure OpenAI, fallback to standard OpenAI provider if this fails")
        
        try:
            logger.info(f"Creating {n} image edits with prompt: '{prompt[:50]}...' using model={model}")
            start_time = datetime.now(timezone.utc)
            
            # Prepare parameters
            params = {
                "image": image_file,
                "prompt": prompt,
                "n": n,
                "size": size,
            }
            
            # Add mask if provided
            if mask:
                params["mask"] = mask
            
            # Model parameter is not used in current OpenAI SDK for edits
            # but kept for future compatibility
            
            # Create image edits
            response = await self._client.images.edit(**params)
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Generated {len(response.data)} image edits in {duration_ms:.2f}ms")
            
            # Convert response to a list of dicts with image URLs
            result = [{"url": img.url} for img in response.data]
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating image edits: {e}", exc_info=True)
            raise
    
    # --- MODELS API ---
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from the OpenAI API.
        
        Returns:
            List of available models with their metadata
        """
        if not self._client:
            raise ConnectionError("OpenAI client is not initialized")
        
        try:
            logger.info("Listing available models")
            start_time = datetime.now(timezone.utc)
            
            response = await self._client.models.list()
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Listed {len(response.data)} models in {duration_ms:.2f}ms")
            
            # Convert Model objects to dicts
            models = []
            for model in response.data:
                model_data = {
                    "id": model.id,
                    "created": model.created,
                    "owned_by": model.owned_by,
                    "object": model.object,
                }
                models.append(model_data)
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}", exc_info=True)
            raise
    
    async def get_model(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_id: The ID of the model to retrieve
            
        Returns:
            Model information
        """
        if not self._client:
            raise ConnectionError("OpenAI client is not initialized")
        
        try:
            logger.info(f"Retrieving model information for {model_id}")
            start_time = datetime.now(timezone.utc)
            
            model = await self._client.models.retrieve(model_id)
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Retrieved model information in {duration_ms:.2f}ms")
            
            # Convert Model object to dict
            model_data = {
                "id": model.id,
                "created": model.created,
                "owned_by": model.owned_by,
                "object": model.object,
            }
            
            return model_data
            
        except Exception as e:
            logger.error(f"Error retrieving model {model_id}: {e}", exc_info=True)
            raise
    
    # --- FILES API ---
    
    async def upload_file(self, file: BinaryIO, purpose: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload a file to the OpenAI API for use with features like fine-tuning or assistants.
        
        Args:
            file: File object to upload
            purpose: Purpose of the file (e.g., 'fine-tune', 'assistants')
            filename: Optional filename for the uploaded file
            
        Returns:
            Information about the uploaded file
        """
        if not self._client:
            raise ConnectionError("OpenAI client is not initialized")
        
        try:
            if filename:
                logger.info(f"Uploading file '{filename}' for purpose '{purpose}'")
            else:
                logger.info(f"Uploading file for purpose '{purpose}'")
                
            start_time = datetime.now(timezone.utc)
            
            # Prepare the optional filename parameter
            params = {
                "file": file,
                "purpose": purpose,
            }
            
            if filename:
                # Note: In the OpenAI v1 SDK, you might need to pass this differently depending on the library version
                # Some versions might expect a tuple of (filename, file) instead
                try:
                    params["filename"] = filename
                except:
                    # If that fails, we'll rely on the file object's name if available
                    logger.warning(f"Could not set filename parameter, using file object name if available")
            
            response = await self._client.files.create(**params)
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Uploaded file with ID {response.id} in {duration_ms:.2f}ms")
            
            # Convert FileObject to dict
            file_data = {
                "id": response.id,
                "bytes": response.bytes,
                "created_at": response.created_at,
                "filename": response.filename,
                "purpose": response.purpose,
                "status": response.status,
            }
            
            return file_data
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}", exc_info=True)
            raise
    
    async def list_files(self, purpose: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List files that have been uploaded to the OpenAI API.
        
        Args:
            purpose: Optional filter by purpose (e.g., 'fine-tune', 'assistants')
            
        Returns:
            List of file information
        """
        if not self._client:
            raise ConnectionError("OpenAI client is not initialized")
        
        try:
            if purpose:
                logger.info(f"Listing files with purpose '{purpose}'")
            else:
                logger.info("Listing all files")
                
            start_time = datetime.now(timezone.utc)
            
            # Prepare the optional purpose parameter
            params = {}
            if purpose:
                params["purpose"] = purpose
            
            response = await self._client.files.list(**params)
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Listed {len(response.data)} files in {duration_ms:.2f}ms")
            
            # Convert FileObject list to dict list
            files = []
            for file in response.data:
                file_data = {
                    "id": file.id,
                    "bytes": file.bytes,
                    "created_at": file.created_at,
                    "filename": file.filename,
                    "purpose": file.purpose,
                    "status": file.status,
                }
                files.append(file_data)
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files: {e}", exc_info=True)
            raise
    
    async def get_file(self, file_id: str) -> Dict[str, Any]:
        """
        Get information about a specific file.
        
        Args:
            file_id: The ID of the file to retrieve
            
        Returns:
            File information
        """
        if not self._client:
            raise ConnectionError("OpenAI client is not initialized")
        
        try:
            logger.info(f"Retrieving file information for ID {file_id}")
            start_time = datetime.now(timezone.utc)
            
            file = await self._client.files.retrieve(file_id)
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Retrieved file information in {duration_ms:.2f}ms")
            
            # Convert FileObject to dict
            file_data = {
                "id": file.id,
                "bytes": file.bytes,
                "created_at": file.created_at,
                "filename": file.filename,
                "purpose": file.purpose,
                "status": file.status,
            }
            
            return file_data
            
        except Exception as e:
            logger.error(f"Error retrieving file {file_id}: {e}", exc_info=True)
            raise
    
    async def get_file_content(self, file_id: str) -> bytes:
        """
        Get the content of a specific file.
        
        Args:
            file_id: The ID of the file to retrieve content for
            
        Returns:
            File content as bytes
        """
        if not self._client:
            raise ConnectionError("OpenAI client is not initialized")
        
        try:
            logger.info(f"Retrieving content for file ID {file_id}")
            start_time = datetime.now(timezone.utc)
            
            response = await self._client.files.content(file_id)
            content = response.read()
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Retrieved {len(content)} bytes of file content in {duration_ms:.2f}ms")
            
            return content
            
        except Exception as e:
            logger.error(f"Error retrieving file content for {file_id}: {e}", exc_info=True)
            raise
    
    async def delete_file(self, file_id: str) -> Dict[str, Any]:
        """
        Delete a file from the OpenAI API.
        
        Args:
            file_id: The ID of the file to delete
            
        Returns:
            Deletion confirmation
        """
        if not self._client:
            raise ConnectionError("OpenAI client is not initialized")
        
        try:
            logger.info(f"Deleting file ID {file_id}")
            start_time = datetime.now(timezone.utc)
            
            deletion = await self._client.files.delete(file_id)
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Deleted file in {duration_ms:.2f}ms")
            
            # Return deletion confirmation
            return {
                "id": deletion.id,
                "object": deletion.object,
                "deleted": deletion.deleted,
            }
            
        except Exception as e:
            logger.error(f"Error deleting file {file_id}: {e}", exc_info=True)
            raise
    
    # --- MODERATION API ---
    
    async def create_moderation(self, input_text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Create a moderation check for the provided text.
        
        Args:
            input_text: Text content to check, can be single string or list of strings
            
        Returns:
            Moderation results with category flags
        """
        if not self._client:
            raise ConnectionError("OpenAI client is not initialized")
        
        # Convert single string to list for consistent processing
        inputs = input_text if isinstance(input_text, list) else [input_text]
        
        try:
            logger.info(f"Creating moderation for {len(inputs)} text inputs")
            start_time = datetime.now(timezone.utc)
            
            # The moderation endpoint is the same for both OpenAI and Azure
            response = await self._client.moderations.create(input=inputs)
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Completed moderation in {duration_ms:.2f}ms")
            
            # Convert to dict for compatibility
            result = {
                "id": response.id,
                "model": response.model,
                "results": []
            }
            
            # Process each result
            for res in response.results:
                result_item = {
                    "flagged": res.flagged,
                    "categories": {k: v for k, v in res.categories.model_dump().items()},
                    "category_scores": {k: v for k, v in res.category_scores.model_dump().items()},
                }
                result["results"].append(result_item)
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating moderation: {e}", exc_info=True)
            raise

    # --- ASSISTANTS API ---
    
    async def create_assistant(self, model: str, name: Optional[str] = None, 
                        description: Optional[str] = None, 
                        instructions: Optional[str] = None,
                        tools: Optional[List[Dict[str, Any]]] = None, 
                        file_ids: Optional[List[str]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create an OpenAI Assistant.
        
        Args:
            model: ID of the model to use (e.g. gpt-4, gpt-3.5-turbo)
            name: Optional name for the assistant
            description: Optional description of the assistant
            instructions: System instructions for the assistant
            tools: List of tools the assistant can use
            file_ids: List of file IDs the assistant can access
            metadata: Additional metadata for the assistant
            
        Returns:
            The created assistant information
        """
        if not self._client:
            raise ConnectionError("OpenAI client is not initialized")
        
        if self.is_azure:
            # Assistants API may not be available on Azure
            logger.warning("Assistants API may not be supported on Azure OpenAI, fallback to standard OpenAI provider if this fails")
        
        try:
            logger.info(f"Creating assistant with model {model}")
            start_time = datetime.now(timezone.utc)
            
            # Prepare parameters
            params = {"model": model}
            
            # Add optional parameters if provided
            if name:
                params["name"] = name
            if description:
                params["description"] = description
            if instructions:
                params["instructions"] = instructions
            if tools:
                params["tools"] = tools
            if file_ids:
                params["file_ids"] = file_ids
            if metadata:
                params["metadata"] = metadata
            
            # Create assistant
            assistant = await self._client.beta.assistants.create(**params)
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Created assistant {assistant.id} in {duration_ms:.2f}ms")
            
            # Convert to dict
            assistant_data = {
                "id": assistant.id,
                "object": assistant.object,
                "created_at": assistant.created_at,
                "name": assistant.name,
                "description": assistant.description,
                "model": assistant.model,
                "instructions": assistant.instructions,
                "tools": [tool.model_dump() for tool in assistant.tools] if assistant.tools else [],
                "file_ids": assistant.file_ids,
                "metadata": assistant.metadata,
            }
            
            return assistant_data
            
        except Exception as e:
            logger.error(f"Error creating assistant: {e}", exc_info=True)
            raise
    
    async def get_assistant(self, assistant_id: str) -> Dict[str, Any]:
        """
        Get information about a specific assistant.
        
        Args:
            assistant_id: The ID of the assistant to retrieve
            
        Returns:
            Assistant information
        """
        if not self._client:
            raise ConnectionError("OpenAI client is not initialized")
        
        if self.is_azure:
            # Assistants API may not be available on Azure
            logger.warning("Assistants API may not be supported on Azure OpenAI, fallback to standard OpenAI provider if this fails")
        
        try:
            logger.info(f"Retrieving assistant {assistant_id}")
            start_time = datetime.now(timezone.utc)
            
            assistant = await self._client.beta.assistants.retrieve(assistant_id)
            
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(f"Retrieved assistant in {duration_ms:.2f}ms")
            
            # Convert to dict
            assistant_data = {
                "id": assistant.id,
                "object": assistant.object,
                "created_at": assistant.created_at,
                "name": assistant.name,
                "description": assistant.description,
                "model": assistant.model,
                "instructions": assistant.instructions,
                "tools": [tool.model_dump() for tool in assistant.tools] if assistant.tools else [],
                "file_ids": assistant.file_ids,
                "metadata": assistant.metadata,
            }
            
            return assistant_data
            
        except Exception as e:
            logger.error(f"Error retrieving assistant {assistant_id}: {e}", exc_info=True)
            raise




    class OpenAIModel(Enum):
        """Latest OpenAI models with their capabilities and pricing tiers."""
        # GPT-4 family
        GPT4 = "gpt-4"                    # Base GPT-4 (8K)
        GPT4_32K = "gpt-4-32k"            # Extended context GPT-4 (32K)
        GPT4_TURBO = "gpt-4-turbo"        # GPT-4 Turbo (128K)
        GPT4_VISION = "gpt-4-vision-preview"  # Vision-capable GPT-4
        GPT4O = "gpt-4o"                  # Latest GPT-4o (128K)
        
        # GPT-3.5 family
        GPT35_TURBO = "gpt-3.5-turbo"     # Latest GPT-3.5 (4K)
        GPT35_TURBO_16K = "gpt-3.5-turbo-16k"  # Extended context GPT-3.5 (16K)
        GPT35_INSTRUCT = "gpt-3.5-turbo-instruct"  # Instruct variant
        
        # Embedding models
        EMBEDDING_3_SMALL = "text-embedding-3-small"  # Cheaper, smaller dimensions
        EMBEDDING_3_LARGE = "text-embedding-3-large"  # Better quality, more dimensions
        
        # Image models
        DALL_E_2 = "dall-e-2"
        DALL_E_3 = "dall-e-3"
        
        # Audio models
        WHISPER = "whisper-1"
        TTS_1 = "tts-1"
        TTS_1_HD = "tts-1-hd"

    def has_vision_capability(model_id: str) -> bool:
        """
        Check if a model has vision capabilities.
        
        Args:
            model_id: The model identifier
            
        Returns:
            True if the model supports vision/image inputs
        """
        vision_models = [
            OpenAIModel.GPT4_VISION.value,
            OpenAIModel.GPT4O.value,
            # Add any new vision-capable models here
            "gpt-4-vision",
            "gpt-4o",
        ]
        
        # Case-insensitive check for model containing these strings
        return any(
            vision_model.lower() in model_id.lower() 
            for vision_model in vision_models
        )

    def has_parallel_function_calling(model_id: str) -> bool:
        """
        Check if a model supports parallel function calling.
        
        Args:
            model_id: The model identifier
            
        Returns:
            True if the model supports parallel function calling
        """
        parallel_function_models = [
            OpenAIModel.GPT4_TURBO.value,
            OpenAIModel.GPT4O.value,
            # Add any new parallel function calling models here
            "gpt-4-turbo",
            "gpt-4o",
        ]
        
        # Case-insensitive check
        return any(
            pfm.lower() in model_id.lower()
            for pfm in parallel_function_models
        )

    def has_json_mode(model_id: str) -> bool:
        """
        Check if a model supports JSON mode.
        
        Args:
            model_id: The model identifier
            
        Returns:
            True if the model supports JSON mode
        """
        # Most newer models support JSON mode
        json_mode_models = [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-3.5-turbo",
        ]
        
        # Check if model name contains any supported model string
        return any(
            jm.lower() in model_id.lower()
            for jm in json_mode_models
        )

    def configure_tool_choice(
        openai_params: Dict[str, Any],
        request: LLMRequest
    ) -> Dict[str, Any]:
        """
        Configure the tool_choice parameter for the OpenAI API request.
        
        Args:
            openai_params: Existing OpenAI parameters dict
            request: The LLM request with tool configuration
            
        Returns:
            The updated OpenAI parameters dict
        """
        # Check if tools exist in the request
        if not request.tools:
            return openai_params
        
        # Extract tool_choice from extra_params if present
        tool_choice = request.config.extra_params.get("tool_choice")
        
        if tool_choice is None:
            # Default to "auto" if not specified
            openai_params["tool_choice"] = "auto"
        elif tool_choice == "none":
            # Explicitly tell the model not to use tools
            openai_params["tool_choice"] = "none"
        elif tool_choice == "auto":
            # Use automatic tool selection
            openai_params["tool_choice"] = "auto"
        elif tool_choice == "required":
            # Force the model to use a tool
            openai_params["tool_choice"] = "required"
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            # Specific function request
            function_name = tool_choice.get("function", {}).get("name")
            if function_name:
                # Find the tool definition to validate
                tool_exists = any(
                    t.function.name == function_name
                    for t in request.tools
                )
                
                if tool_exists:
                    openai_params["tool_choice"] = tool_choice
                else:
                    logger.warning(
                        f"Requested tool '{function_name}' not found in available tools. "
                        f"Falling back to 'auto'."
                    )
                    openai_params["tool_choice"] = "auto"
        else:
            # Unrecognized format, default to auto
            logger.warning(f"Unrecognized tool_choice format: {tool_choice}. Using 'auto'.")
            openai_params["tool_choice"] = "auto"
        
        return openai_params

    def update_openai_client_for_latest_models(client_class, mappings_module):
        """
        Update the OpenAI client for latest model support.
        
        Args:
            client_class: The OpenAI client class to update
            mappings_module: The mappings module to update
        """
        # Update map_request to include tool_choice configuration
        original_map_request = mappings_module.map_request
        
        def enhanced_map_request(request: LLMRequest, model_id: str) -> Dict[str, Any]:
            """Enhanced request mapping with tool_choice and latest model support."""
            openai_params = original_map_request(request, model_id)
            
            # Add tool_choice configuration if tools are present
            if request.tools:
                openai_params = configure_tool_choice(openai_params, request)
                
            # Add parallel function calling parameters if supported
            if has_parallel_function_calling(model_id):
                openai_params["parallel_tool_calls"] = True
            
            # Add model-specific parameters based on capabilities
            if has_vision_capability(model_id):
                # Vision models might need additional configuration
                # Currently handled in content mapping, but could add here if needed
                pass
                
            return openai_params
        
        # Replace the mapping function
        mappings_module.map_request = enhanced_map_request
        
        # Add capability checks to the client class
        client_class.has_vision_capability = staticmethod(has_vision_capability)
        client_class.has_parallel_function_calling = staticmethod(has_parallel_function_calling)
        client_class.has_json_mode = staticmethod(has_json_mode)
        
    def extend_vision_api_support(client_class):
        """
        Extend the OpenAI client with improved vision API support.
        
        Args:
            client_class: The OpenAI client class to extend
        """
        # Add a convenience method for vision requests
        async def generate_with_images(
            self,
            text_prompt: str,
            image_urls: List[str],
            model: str = "gpt-4o",
            max_tokens: int = 1000,
            **kwargs
        ):
            """
            Generate a response with text and images using vision-capable models.
            
            Args:
                text_prompt: The text prompt
                image_urls: List of image URLs to include
                model: Vision-capable model to use
                max_tokens: Maximum tokens to generate
                **kwargs: Additional parameters for the request
                
            Returns:
                The generated response
            """
            from asf.medical.llm_gateway.core.models import (
                ContentItem,
                LLMRequest,
                LLMRequestConfig,
                RequestContext,
            )
            import uuid
            
            # Verify model supports vision
            if not has_vision_capability(model):
                raise ValueError(
                    f"Model '{model}' does not support vision capabilities. "
                    f"Use a vision-capable model like 'gpt-4o' or 'gpt-4-vision-preview'."
                )
            
            # Create content items for each image
            content_items = [
                ContentItem(
                    type=MCPContentType.TEXT,
                    text_content=text_prompt
                )
            ]
            
            for url in image_urls:
                content_items.append(
                    ContentItem(
                        type=MCPContentType.IMAGE,
                        mime_type="image/jpeg",  # Assumed, could be determined from URL
                        data={
                            "image": {
                                "source": {
                                    "type": "url",
                                    "url": url
                                }
                            }
                        }
                    )
                )
            
            # Create the request
            request = LLMRequest(
                version="1.0",
                prompt_content=content_items,
                config=LLMRequestConfig(
                    model_identifier=model,
                    max_tokens=max_tokens,
                    **{k: v for k, v in kwargs.items() if k in [
                        "temperature", "top_p", "presence_penalty", 
                        "frequency_penalty", "stop_sequences"
                    ]}
                ),
                initial_context=RequestContext(
                    request_id=str(uuid.uuid4()),
                    conversation_history=[]
                )
            )
            
            # Call the regular generate method
            return await self.generate(request)
    
    # Add the method to the client class
    client_class.generate_with_images = generate_with_images    
    # --- Additional convenience methods and wrappers can be added here ---
    
    # Examples:
    # - Thread management for Assistants API
    # - Message management for Assistants API  
    # - Run management for Assistants API
    # - Vector store operations
    # - Fine-tuning methods
    # - Etc.

# Enhanced exports
__all__ = ["OpenAIClient"]