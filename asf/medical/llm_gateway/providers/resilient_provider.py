"""
Resilient provider base class with integrated resilience patterns.

This module provides a base class for LLM providers with integrated
resilience patterns, including circuit breakers, retries, and timeouts.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, Optional, Any

from asf.medical.llm_gateway.providers.base import BaseProvider
from asf.medical.llm_gateway.core.models import (
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    StreamChunk,
    ErrorDetails,
    ErrorLevel,
    FinishReason
)

from asf.medical.llm_gateway.resilience.enhanced_circuit_breaker import (
    FailureCategory,
    get_circuit_breaker_registry,
    RecoveryStrategy
)
from asf.medical.llm_gateway.resilience.factory import get_resilience_factory
from asf.medical.llm_gateway.resilience.metrics import get_resilience_metrics
# Try to import traced decorators
try:
    from asf.medical.llm_gateway.resilience.traced_decorators import (
        with_traced_provider_circuit_breaker as with_provider_circuit_breaker,
        with_traced_retry as with_retry,
        with_traced_timeout as with_timeout
    )
    from asf.medical.llm_gateway.resilience.decorators import CircuitOpenError
    TRACING_AVAILABLE = True
except ImportError:
    from asf.medical.llm_gateway.resilience.decorators import (
        with_provider_circuit_breaker,
        with_retry,
        with_timeout,
        CircuitOpenError
    )
    TRACING_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResilientProvider(BaseProvider):
    """
    Resilient provider base class with integrated resilience patterns.

    This class extends the base provider with resilience patterns,
    including circuit breakers, retries, and timeouts.

    Features:
    - Circuit breaker pattern for preventing cascading failures
    - Retry pattern for handling transient failures
    - Timeout pattern for preventing long-running requests
    - Metrics integration for monitoring resilience patterns
    - Adaptive recovery based on failure patterns
    """

    def __init__(
        self,
        provider_config: ProviderConfig,
        circuit_breaker_config: Optional[Dict[str, Any]] = None,
        retry_config: Optional[Dict[str, Any]] = None,
        timeout_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the resilient provider.

        Args:
            provider_config: Provider configuration
            circuit_breaker_config: Circuit breaker configuration
            retry_config: Retry configuration
            timeout_config: Timeout configuration
        """
        super().__init__(provider_config)

        # Log tracing availability
        if TRACING_AVAILABLE:
            logger.info(f"Tracing is available for resilience patterns for provider: {self.provider_id}")
        else:
            logger.warning(f"Tracing is not available for resilience patterns for provider: {self.provider_id}")

        # Get resilience factory
        self.resilience_factory = get_resilience_factory()

        # Get resilience metrics
        self.resilience_metrics = get_resilience_metrics()

        # Get circuit breaker registry
        self.circuit_breaker_registry = get_circuit_breaker_registry()

        # Default circuit breaker config
        self.circuit_breaker_config = circuit_breaker_config or {
            "failure_threshold": 5,
            "recovery_timeout": 30,
            "half_open_max_calls": 1,
            "reset_timeout": 600,  # 10 minutes
            "recovery_strategy": RecoveryStrategy.EXPONENTIAL,
            "max_recovery_timeout": 1800,  # 30 minutes
            "min_recovery_timeout": 1,  # 1 second
            "jitter_factor": 0.2  # 20% jitter
        }

        # Default retry config
        self.retry_config = retry_config or {
            "max_attempts": 3,
            "backoff_factor": 0.5,
            "jitter": True,
            "max_backoff": 60.0,
            "retry_on_circuit_open": False
        }

        # Default timeout config
        self.timeout_config = timeout_config or {
            "generate_timeout": 60.0,  # 60 seconds
            "stream_timeout": 300.0,  # 5 minutes
            "embedding_timeout": 30.0  # 30 seconds
        }

        # Create circuit breakers
        self._create_circuit_breakers()

        logger.info(
            f"Initialized resilient provider: {self.provider_id}",
            provider_type=self.provider_config.provider_type
        )

    def _create_circuit_breakers(self) -> None:
        """Create circuit breakers for this provider."""
        # Main provider circuit breaker
        self.provider_circuit_breaker = self.resilience_factory.create_circuit_breaker(
            name=f"{self.provider_id}_main",
            provider_id=self.provider_id,
            **self.circuit_breaker_config
        )

        # Generate circuit breaker
        self.generate_circuit_breaker = self.resilience_factory.create_circuit_breaker(
            name=f"{self.provider_id}_generate",
            provider_id=self.provider_id,
            **self.circuit_breaker_config
        )

        # Stream circuit breaker
        self.stream_circuit_breaker = self.resilience_factory.create_circuit_breaker(
            name=f"{self.provider_id}_stream",
            provider_id=self.provider_id,
            **self.circuit_breaker_config
        )

        # Embedding circuit breaker
        self.embedding_circuit_breaker = self.resilience_factory.create_circuit_breaker(
            name=f"{self.provider_id}_embedding",
            provider_id=self.provider_id,
            **self.circuit_breaker_config
        )

    @with_provider_circuit_breaker(
        method_name="generate",
        failure_threshold=5,
        recovery_timeout=30,
        failure_exceptions=[Exception],
        failure_category_mapping={
            TimeoutError: FailureCategory.TIMEOUT,
            asyncio.TimeoutError: FailureCategory.TIMEOUT,
            ConnectionError: FailureCategory.CONNECTION,
            ValueError: FailureCategory.VALIDATION
        }
    )
    @with_retry(max_attempts=3, backoff_factor=0.5, jitter=True)
    @with_timeout(timeout_seconds=60.0)
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response for the given request with resilience patterns.

        Args:
            request: The request containing prompt, conversation history, and config

        Returns:
            A complete response with generated content or error details
        """
        try:
            # Call the provider-specific implementation
            return await self._generate_internal(request)
        except CircuitOpenError as e:
            # Create error response for circuit open
            return self._create_circuit_open_response(request, e)
        except Exception as e:
            # Log the error
            logger.error(
                f"Error generating response: {str(e)}",
                exc_info=True,
                provider_id=self.provider_id,
                request_id=request.initial_context.request_id
            )

            # Create error response
            return self._create_error_response(request, e)

    async def _generate_internal(self, request: LLMRequest) -> LLMResponse:
        """
        Internal method for generating a response.

        This method should be overridden by provider implementations
        with their specific generation logic.

        Args:
            request: The request containing prompt, conversation history, and config

        Returns:
            A complete response with generated content or error details
        """
        # Default implementation raises NotImplementedError
        # pylint: disable=unused-argument
        raise NotImplementedError("Provider must implement _generate_internal")

    @with_provider_circuit_breaker(
        method_name="generate_stream",
        failure_threshold=5,
        recovery_timeout=30,
        failure_exceptions=[Exception],
        failure_category_mapping={
            TimeoutError: FailureCategory.TIMEOUT,
            asyncio.TimeoutError: FailureCategory.TIMEOUT,
            ConnectionError: FailureCategory.CONNECTION,
            ValueError: FailureCategory.VALIDATION
        }
    )
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate a streaming response for the given request with resilience patterns.

        Args:
            request: The request containing prompt, conversation history, and config

        Yields:
            Chunks of the generated response
        """
        try:
            # Call the provider-specific implementation
            async for chunk in self._generate_stream_internal(request):
                yield chunk

            # Record success
            self.stream_circuit_breaker.record_success()
        except CircuitOpenError as e:
            # Yield error chunk for circuit open
            yield self._create_circuit_open_chunk(request.initial_context.request_id, e)
        except Exception as e:
            # Log the error
            logger.error(
                f"Error generating streaming response: {str(e)}",
                exc_info=True,
                provider_id=self.provider_id,
                request_id=request.initial_context.request_id
            )

            # Record failure
            self.stream_circuit_breaker.record_failure()

            # Yield error chunk
            yield self._create_error_chunk(request.initial_context.request_id, e)

    async def _generate_stream_internal(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Internal method for generating a streaming response.

        This method should be overridden by provider implementations
        with their specific streaming logic.

        Args:
            request: The request containing prompt, conversation history, and config

        Yields:
            Chunks of the generated response
        """
        # Default implementation raises NotImplementedError
        # pylint: disable=unused-argument
        raise NotImplementedError("Provider must implement _generate_stream_internal")

    def _create_circuit_open_response(self, request: LLMRequest, error: CircuitOpenError) -> LLMResponse:
        """
        Create an error response for circuit open.

        Args:
            request: The original request
            error: The circuit open error

        Returns:
            Error response
        """
        # Log the error
        logger.warning(
            f"Circuit open for provider: {self.provider_id}",
            provider_id=self.provider_id,
            request_id=request.initial_context.request_id,
            error=str(error)
        )

        # Create error details
        error_details = ErrorDetails(
            error_type="CIRCUIT_OPEN",
            error_message=str(error),
            error_level=ErrorLevel.WARNING,
            provider_error_code=None,
            provider_error_message=None,
            request_id=request.initial_context.request_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        # Create response
        return LLMResponse(
            final_context=request.initial_context,
            generated_content=None,
            finish_reason=FinishReason.ERROR,
            error_details=error_details,
            model_info={
                "provider_id": self.provider_id,
                "provider_type": self.provider_config.provider_type,
                "model_identifier": request.config.model_identifier
            },
            usage=None,
            performance_metrics=None,
            additional_info={
                "circuit_open": True,
                "provider_id": self.provider_id
            }
        )

    def _create_circuit_open_chunk(self, request_id: str, error: CircuitOpenError) -> StreamChunk:
        """
        Create an error chunk for circuit open.

        Args:
            request_id: The request ID
            error: The circuit open error

        Returns:
            Error chunk
        """
        # Log the error
        logger.warning(
            f"Circuit open for provider: {self.provider_id}",
            provider_id=self.provider_id,
            request_id=request_id,
            error=str(error)
        )

        # Create error details
        error_details = ErrorDetails(
            error_type="CIRCUIT_OPEN",
            error_message=str(error),
            error_level=ErrorLevel.WARNING,
            provider_error_code=None,
            provider_error_message=None,
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        # Create chunk
        return StreamChunk(
            request_id=request_id,
            content="",
            is_error=True,
            error_details=error_details,
            finish_reason=FinishReason.ERROR,
            model_info={
                "provider_id": self.provider_id,
                "provider_type": self.provider_config.provider_type
            },
            additional_info={
                "circuit_open": True,
                "provider_id": self.provider_id
            }
        )

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the provider is healthy and available.

        Returns:
            A dictionary with health check information
        """
        # Get circuit breaker health
        provider_health = self.provider_circuit_breaker.get_health()
        generate_health = self.generate_circuit_breaker.get_health()
        stream_health = self.stream_circuit_breaker.get_health()
        embedding_health = self.embedding_circuit_breaker.get_health()

        # Calculate overall health
        overall_health_score = min(
            provider_health["health_score"],
            generate_health["health_score"],
            stream_health["health_score"],
            embedding_health["health_score"]
        )

        # Get circuit breaker states
        circuit_breaker_states = {
            "provider": provider_health["state"],
            "generate": generate_health["state"],
            "stream": stream_health["state"],
            "embedding": embedding_health["state"]
        }

        # Check if any circuit breakers are open
        any_open = any(state == "open" for state in circuit_breaker_states.values())

        # Create health info
        health_info = {
            "provider_id": self.provider_id,
            "status": "unavailable" if any_open else "available",
            "created_at": self.created_at.isoformat(),
            "health_score": overall_health_score,
            "circuit_breaker_states": circuit_breaker_states,
            "circuit_breaker_health": {
                "provider": provider_health,
                "generate": generate_health,
                "stream": stream_health,
                "embedding": embedding_health
            },
            "message": "Provider is healthy" if not any_open else "One or more circuit breakers are open"
        }

        return health_info
