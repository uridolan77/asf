# llm_gateway/core/client.py
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, List, Optional, Union, Any, Tuple

from asf.conexus.llm_gateway.core.models import (
    BatchLLMRequest, BatchLLMResponse, LLMRequest, LLMResponse, StreamChunk,
    GatewayConfig, InterventionContext, ErrorDetails, ErrorLevel,
    PerformanceMetrics, FinishReason
)
from asf.conexus.llm_gateway.core.factory import ProviderFactory, ProviderFactoryError
from asf.conexus.llm_gateway.core.manager import InterventionManager
from asf.conexus.llm_gateway.core.config_loader import ConfigLoader
from asf.conexus.llm_gateway.core.resource_init import setup_resource_layer, shutdown_resource_layer, get_resource_stats
from asf.conexus.llm_gateway.core.errors import ResourceError, CircuitBreakerError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class LLMGatewayClient:
    def __init__(self, config: GatewayConfig = None, provider_factory: Optional[ProviderFactory] = None, db: Session = None):
         self.db = db
         self.config_loader = ConfigLoader(db)
         self._resource_layer_initialized = False

         # Load configuration from database if not provided
         if config is None and db is not None:
             config_dict = self.config_loader.load_config()
             self.config = GatewayConfig(**config_dict)
         else:
             self.config = config

         if self.config is None:
             raise ValueError("Gateway configuration is required. Provide either config or db.")

         # Use a central factory instance
         self.provider_factory = provider_factory or ProviderFactory()
         # Pass the factory to the manager along with the database session
         self.intervention_manager = InterventionManager(self.provider_factory, self.config, self.db)
         # Configure semaphore from GatewayConfig
         batch_limit = self.config.additional_config.get("max_concurrent_batch_requests", 10)
         self._batch_semaphore = asyncio.Semaphore(batch_limit)
         logger.info(f"Initialized LLMGatewayClient with batch concurrency limit: {batch_limit}")

    async def initialize_resources(self):
        """Initialize the resource management layer."""
        if not self._resource_layer_initialized:
            await setup_resource_layer(register_signals=True)
            self._resource_layer_initialized = True
            logger.info("Resource management layer initialized")

    async def generate(self, request: LLMRequest) -> LLMResponse:
         """Process a single LLM request."""
         logger.info(f"Processing generate request: {request.initial_context.request_id}")
         try:
              if not self._resource_layer_initialized:
                  await self.initialize_resources()
              response = await self.intervention_manager.process_request(request)
              logger.info(f"Request successful: {request.initial_context.request_id}")
              return response
         except ProviderFactoryError as e:
              logger.error(f"Provider creation failed for request {request.initial_context.request_id}: {e}", exc_info=True)
              return self._create_error_response(request, e, "PROVIDER_INIT_FAILED")
         except ResourceError as e:
              logger.error(f"Resource error in request {request.initial_context.request_id}: {e}", exc_info=True)
              return self._create_error_response(request, e, "RESOURCE_ERROR", retryable=True)
         except CircuitBreakerError as e:
              logger.error(f"Circuit breaker tripped for request {request.initial_context.request_id}: {e}", exc_info=True)
              return self._create_error_response(request, e, "CIRCUIT_BREAKER_TRIPPED", retryable=True)
         except Exception as e:
              logger.error(f"Request processing failed: {request.initial_context.request_id}", exc_info=True)
              return self._create_error_response(request, e)

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """Process a streaming request."""
        request_id = request.initial_context.request_id
        logger.info(f"Processing stream request: {request_id}")
        try:
            if not self._resource_layer_initialized:
                await self.initialize_resources()
            async for chunk in self.intervention_manager.process_stream(request):
                yield chunk
            logger.info(f"Stream finished successfully: {request_id}")
        except ProviderFactoryError as e:
              logger.error(f"Provider creation failed for stream {request_id}: {e}", exc_info=True)
              yield self._create_error_chunk(request_id, e, "PROVIDER_INIT_FAILED")
        except ResourceError as e:
              logger.error(f"Resource error in stream {request_id}: {e}", exc_info=True)
              yield self._create_error_chunk(request_id, e, "RESOURCE_ERROR")
        except CircuitBreakerError as e:
              logger.error(f"Circuit breaker tripped for stream {request_id}: {e}", exc_info=True)
              yield self._create_error_chunk(request_id, e, "CIRCUIT_BREAKER_TRIPPED")
        except Exception as e:
            logger.error(f"Stream processing failed: {request_id}", exc_info=True)
            yield self._create_error_chunk(request_id, e)

    async def process_batch(self, batch_request: BatchLLMRequest) -> BatchLLMResponse:
        """Process a batch of LLM requests."""
        batch_start_time = datetime.now(timezone.utc)
        logger.info(f"Processing batch request: {batch_request.batch_id} ({len(batch_request.requests)} requests)")
        if not self._resource_layer_initialized:
            await self.initialize_resources()
        async with self._batch_semaphore:
            tasks = [
                self._process_single_request_in_batch(req)
                for req in batch_request.requests
            ]
            results_with_requests = await asyncio.gather(*tasks, return_exceptions=True)

            valid_responses: List[LLMResponse] = []
            for req, result in results_with_requests:
                if isinstance(result, Exception):
                     logger.warning(f"Sub-request {req.initial_context.request_id} in batch {batch_request.batch_id} failed.", exc_info=result)
                     valid_responses.append(self._create_error_response(req, result))
                else:
                     valid_responses.append(result)

            total_duration_ms = (datetime.now(timezone.utc) - batch_start_time).total_seconds() * 1000
            logger.info(f"Batch processing complete: {batch_request.batch_id}, Duration: {total_duration_ms:.2f}ms")
            return BatchLLMResponse(
                batch_id=batch_request.batch_id,
                responses=valid_responses,
                total_duration_ms=total_duration_ms
            )

    async def _process_single_request_in_batch(self, request: LLMRequest) -> Tuple[LLMRequest, LLMResponse]:
         """Wrapper for batch processing that returns request context on error."""
         try:
              response = await self.intervention_manager.process_request(request)
              return request, response
         except Exception as e:
              raise e

    async def close(self):
        """Clean up provider resources."""
        logger.info("Shutting down gateway client and cleaning up providers...")
        await self.provider_factory.cleanup_all()
        if self._resource_layer_initialized:
            try:
                await shutdown_resource_layer()
                self._resource_layer_initialized = False
                logger.info("Resource management layer shut down successfully")
            except Exception as e:
                logger.error(f"Error shutting down resource management layer: {e}", exc_info=True)
        logger.info("Gateway client shutdown complete.")

    def _create_error_response(self, request: Optional[LLMRequest], error: Exception, default_code: str = "CLIENT_PROCESSING_ERROR", retryable: bool = False) -> LLMResponse:
         """Create a standardized error response."""
         request_id = request.initial_context.request_id if request else f"error_{uuid.uuid4()}"
         context = request.initial_context if request else InterventionContext(request_id=request_id)

         error_code = default_code
         message = str(error)
         level = ErrorLevel.ERROR
         provider_details = {"exception_type": type(error).__name__, "details": str(error)}

         if isinstance(error, ResourceError):
             error_code = "RESOURCE_ERROR"
             retryable = True
             if hasattr(error, 'provider_id'):
                 provider_details['provider_id'] = error.provider_id
             if hasattr(error, 'resource_type'):
                 provider_details['resource_type'] = error.resource_type
             if hasattr(error, 'operation'):
                 provider_details['operation'] = error.operation
         elif isinstance(error, CircuitBreakerError):
             error_code = "CIRCUIT_BREAKER_TRIPPED"
             retryable = True
             level = ErrorLevel.WARNING
             if hasattr(error, 'provider_id'):
                 provider_details['provider_id'] = error.provider_id
             if hasattr(error, 'failure_count'):
                 provider_details['failure_count'] = error.failure_count
             if hasattr(error, 'reset_timeout_seconds'):
                 provider_details['reset_timeout_seconds'] = error.reset_timeout_seconds

         error_details = ErrorDetails(
             code=error_code,
             message=message,
             level=level,
             retryable=retryable,
             provider_error_details=provider_details
         )

         duration = (datetime.now(timezone.utc) - context.timestamp_start).total_seconds() * 1000 if context else 0

         return LLMResponse(
             request_id=request_id,
             generated_content=None,
             error_details=error_details,
             final_context=context,
             finish_reason=FinishReason.ERROR,
             performance_metrics=PerformanceMetrics(
                  total_duration_ms=duration,
             )
         )

    def _create_error_chunk(self, request_id: str, error: Exception, error_code: str = "STREAM_PROCESSING_ERROR") -> StreamChunk:
        """Create an error chunk for streaming responses."""
        retryable = isinstance(error, (ResourceError, CircuitBreakerError))
        error_details = ErrorDetails(
            code=error_code,
            message=str(error),
            level=ErrorLevel.ERROR,
            retryable=retryable,
            provider_error_details={"exception_type": type(error).__name__}
        )
        return StreamChunk(
            chunk_id=999,
            request_id=request_id,
            finish_reason=FinishReason.ERROR,
            provider_specific_data={"error": error_details.model_dump()}
        )

    async def health_check(self) -> Dict[str, Any]:
         """Check connectivity to all configured providers."""
         status = {}
         if self._resource_layer_initialized:
             try:
                 resource_stats = get_resource_stats()
                 status["resource_pools"] = resource_stats
             except Exception as e:
                 logger.error(f"Failed to get resource pool stats: {e}", exc_info=True)
                 status["resource_pools"] = {"error": str(e)}

         provider_configs = getattr(self.config, 'providers', {})
         if not provider_configs:
              logger.warning("No provider configurations found in GatewayConfig for health check.")
              provider_ids_to_check = self.config.allowed_providers
              if provider_ids_to_check and not provider_configs:
                  logger.warning("Checking allowed_providers without full configs.")
                  for provider_id in provider_ids_to_check:
                       status[provider_id] = {"status": "unknown", "error": "Missing provider configuration"}
                  return status
              elif not provider_ids_to_check:
                  return {"status": "no_providers_configured"}

         for provider_id, provider_conf in provider_configs.items():
              try:
                   provider = await self.provider_factory.get_provider(provider_id, provider_conf, self.config)
                   status[provider_id] = await provider.health_check()
              except Exception as e:
                   status[provider_id] = {"status": "unhealthy", "provider_id": provider_id, "error": f"Failed to get/check provider: {str(e)}"}
         return status

    async def warmup_providers(self):
         """Pre-initialize providers based on configuration.""" 
         await self.initialize_resources()
         preload_ids = getattr(self.config, 'preload_providers', [])
         provider_configs = getattr(self.config, 'providers', {})

         logger.info(f"Warming up providers: {preload_ids}")
         warmed_up = []
         for provider_id in preload_ids:
              provider_conf = provider_configs.get(provider_id)
              if provider_conf:
                   try:
                        await self.provider_factory.get_provider(provider_id, provider_conf, self.config)
                        warmed_up.append(provider_id)
                   except Exception as e:
                        logger.error(f"Failed to warm up provider {provider_id}: {e}")
              else:
                   logger.warning(f"Cannot warm up provider {provider_id}: Configuration not found.")
         logger.info(f"Successfully warmed up providers: {warmed_up}")

    async def get_resource_stats(self) -> Dict[str, Any]:
         """Get statistics about all resource pools."""
         if not self._resource_layer_initialized:
             return {"status": "resource_layer_not_initialized"}
         try:
             return get_resource_stats()
         except Exception as e:
             logger.error(f"Failed to get resource stats: {e}", exc_info=True)
             return {"status": "error", "message": str(e)}
