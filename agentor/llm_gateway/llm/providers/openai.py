import openai
from typing import Dict, Any, Optional
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from agentor.llm_gateway.llm.base import BaseLLM, LLMRequest, LLMResponse
from agentor.llm_gateway.utils.circuit_breaker import LLMCircuitBreaker, CircuitBreakerOpenError

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider."""

    def __init__(self, api_key: str, organization: Optional[str] = None):
        """Initialize the OpenAI LLM provider.

        Args:
            api_key: The OpenAI API key
            organization: The OpenAI organization ID (optional)
        """
        self.client = openai.AsyncOpenAI(api_key=api_key, organization=organization)
        self.circuit_breaker = LLMCircuitBreaker()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (openai.APIError, openai.APIConnectionError)
        )
    )
    async def _generate_internal(self, request: LLMRequest) -> LLMResponse:
        """Internal method to generate a response from the OpenAI API.

        This method is wrapped by the circuit breaker in the generate method.

        Args:
            request: The request to send to the OpenAI API

        Returns:
            The response from the OpenAI API
        """
        logger.info(f"Sending request to OpenAI API: {request.model}")

        response = await self.client.completions.create(
            model=request.model,
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop=request.stop_sequences
        )

        # Extract the response text
        text = response.choices[0].text

        # Extract usage information
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        logger.info(f"Received response from OpenAI API: {len(text)} chars")

        return LLMResponse(
            text=text,
            model=request.model,
            usage=usage,
            metadata={
                "request_id": response.id,
                "original_request": request.model_dump(exclude={"prompt"}),
            }
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the OpenAI API with circuit breaker protection.

        Args:
            request: The request to send to the OpenAI API

        Returns:
            The response from the OpenAI API

        Raises:
            CircuitBreakerOpenError: If the circuit breaker is open
            openai.RateLimitError: If the rate limit is exceeded
            openai.APIError: If there is an API error
            Exception: For any other unexpected error
        """
        try:
            # Use the circuit breaker to protect against service failures
            return await self.circuit_breaker.call_with_breaker(
                provider="openai",
                func=self._generate_internal,
                request=request
            )

        except CircuitBreakerOpenError as e:
            logger.error(f"Circuit breaker open: {e}")
            raise

        except openai.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
