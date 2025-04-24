"""
Progress-tracking enhanced LLM Gateway client.

This module provides an enhanced version of the LLM Gateway client that
incorporates progress tracking for all LLM operations.
"""

import logging
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Union, Any, Tuple

from asf.medical.llm_gateway.core.client import LLMGatewayClient
from asf.medical.llm_gateway.core.models import (
    BatchLLMRequest, BatchLLMResponse, LLMRequest, LLMResponse, StreamChunk,
    GatewayConfig, InterventionContext, ErrorDetails, ErrorLevel,
    PerformanceMetrics, FinishReason
)
from asf.medical.llm_gateway.core.factory import ProviderFactory, ProviderFactoryError

from asf.medical.llm_gateway.progress import (
    track_llm_progress, get_progress_tracker, ProgressTracker,
    get_progress_registry, OperationType
)

logger = logging.getLogger(__name__)


class ProgressTrackingLLMClient(LLMGatewayClient):
    """
    Enhanced LLM Gateway client with integrated progress tracking.
    
    This class extends the base LLM Gateway client with progress tracking
    capabilities, automatically tracking the progress of all LLM operations.
    """
    
    def __init__(self, config: GatewayConfig = None, provider_factory: Optional[ProviderFactory] = None, db=None):
        """
        Initialize the progress tracking LLM client.
        
        Args:
            config: Gateway configuration
            provider_factory: Provider factory
            db: Database session
        """
        super().__init__(config, provider_factory, db)
        logger.info("Initializing progress tracking LLM client")
        
        # Initialize progress registry
        self.progress_registry = get_progress_registry()
    
    @track_llm_progress(operation_type=OperationType.LLM_REQUEST, total_steps=5)
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Process a single LLM request with progress tracking.
        
        Args:
            request: LLM request
            
        Returns:
            LLM response
        """
        # Get the current tracker
        tracker = get_progress_tracker()
        
        # Step 1: Initialize request
        tracker.update(1, "Initializing request")
        
        # Add request ID to logger context
        logger.info(f"Processing generate request with progress tracking: {request.initial_context.request_id}")
        
        try:
            # Step 2: Pre-processing
            tracker.update(2, "Pre-processing request")
            
            # Step 3: Provider selection
            tracker.update(3, "Selecting provider")
            
            # Step 4: Generating response
            tracker.update(4, "Generating response")
            
            # Delegate to intervention manager
            response = await self.intervention_manager.process_request(request)
            
            # Step 5: Post-processing
            tracker.update(5, "Post-processing response")
            
            logger.info(f"Request successful: {request.initial_context.request_id}")
            return response
        except ProviderFactoryError as e:
            # Handle factory errors specifically
            logger.error(f"Provider creation failed for request {request.initial_context.request_id}: {e}", exc_info=True)
            tracker.fail(f"Provider creation failed: {str(e)}")
            return self._create_error_response(request, e, "PROVIDER_INIT_FAILED")
        except Exception as e:
            # Handle other errors
            logger.error(f"Request processing failed: {request.initial_context.request_id}", exc_info=True)
            tracker.fail(f"Request processing failed: {str(e)}")
            return self._create_error_response(request, e)
    
    @track_llm_progress(operation_type=OperationType.LLM_STREAMING, total_steps=5)
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Process a streaming request with progress tracking.
        
        Args:
            request: LLM request
            
        Yields:
            Stream chunks
        """
        # Get the current tracker
        tracker = get_progress_tracker()
        
        # Step 1: Initialize request
        tracker.update(1, "Initializing streaming request")
        
        request_id = request.initial_context.request_id
        logger.info(f"Processing stream request with progress tracking: {request_id}")
        
        try:
            # Step 2: Pre-processing
            tracker.update(2, "Pre-processing streaming request")
            
            # Step 3: Provider selection
            tracker.update(3, "Selecting provider for streaming")
            
            # Step 4: Starting stream
            tracker.update(4, "Starting response stream")
            
            # Process the stream
            chunk_count = 0
            async for chunk in self.intervention_manager.process_stream(request):
                chunk_count += 1
                
                # Update progress periodically (every 10 chunks)
                if chunk_count % 10 == 0:
                    tracker.update(4, f"Streaming response (chunks: {chunk_count})")
                
                yield chunk
            
            # Step 5: Completing stream
            tracker.update(5, f"Stream completed (total chunks: {chunk_count})")
            
            logger.info(f"Stream finished successfully: {request_id}")
        except ProviderFactoryError as e:
            logger.error(f"Provider creation failed for stream {request_id}: {e}", exc_info=True)
            tracker.fail(f"Provider creation failed: {str(e)}")
            yield self._create_error_chunk(request_id, e, "PROVIDER_INIT_FAILED")
        except Exception as e:
            logger.error(f"Stream processing failed: {request_id}", exc_info=True)
            tracker.fail(f"Stream processing failed: {str(e)}")
            yield self._create_error_chunk(request_id, e)
    
    @track_llm_progress(operation_type=OperationType.LLM_BATCH, total_steps=6)
    async def generate_batch(self, batch_request: BatchLLMRequest) -> BatchLLMResponse:
        """
        Process a batch of LLM requests with progress tracking.
        
        Args:
            batch_request: Batch LLM request
            
        Returns:
            Batch LLM response
        """
        # Get the current tracker
        tracker = get_progress_tracker()
        
        # Step 1: Initialize batch request
        tracker.update(1, f"Initializing batch request with {len(batch_request.requests)} requests")
        
        # Add batch metadata to tracker
        tracker.metadata["batch_size"] = len(batch_request.requests)
        tracker.metadata["batch_id"] = batch_request.batch_id
        
        logger.info(f"Processing batch request with progress tracking: {batch_request.batch_id} ({len(batch_request.requests)} requests)")
        
        try:
            # Step 2: Validate batch
            tracker.update(2, "Validating batch requests")
            
            # Step 3: Pre-processing batch
            tracker.update(3, "Pre-processing batch requests")
            
            # Step 4: Processing individual requests
            tracker.update(4, "Processing individual requests")
            
            # Process each request in the batch
            tasks = []
            for i, request in enumerate(batch_request.requests):
                # Create a sub-tracker for this request
                request_id = request.initial_context.request_id
                sub_tracker = self.progress_registry.create_tracker(
                    operation_id=f"{batch_request.batch_id}_{request_id}",
                    operation_type=OperationType.LLM_REQUEST,
                    total_steps=5,
                    metadata={
                        "batch_id": batch_request.batch_id,
                        "request_index": i,
                        "request_id": request_id
                    }
                )
                
                # Update batch progress
                percent_complete = (i / len(batch_request.requests)) * 100
                tracker.update(
                    4,
                    f"Processing request {i+1}/{len(batch_request.requests)} ({percent_complete:.1f}%)",
                    {"current_request": i, "total_requests": len(batch_request.requests)}
                )
                
                # Add task
                tasks.append(self._process_single_request_in_batch(request))
            
            # Step 5: Gathering results
            tracker.update(5, "Gathering batch results")
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            responses = []
            errors = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Handle exception
                    request = batch_request.requests[i]
                    error_response = self._create_error_response(request, result)
                    responses.append(error_response)
                    errors.append({
                        "request_id": request.initial_context.request_id,
                        "error": str(result)
                    })
                else:
                    # Add successful response
                    request, response = result
                    responses.append(response)
            
            # Step 6: Finalizing batch response
            tracker.update(6, f"Batch completed: {len(responses)} responses generated")
            
            # Create batch response
            batch_response = BatchLLMResponse(
                batch_id=batch_request.batch_id,
                responses=responses,
                errors=errors
            )
            
            logger.info(f"Batch request completed: {batch_request.batch_id} ({len(responses)} responses)")
            return batch_response
        except Exception as e:
            logger.error(f"Batch processing failed: {batch_request.batch_id}", exc_info=True)
            tracker.fail(f"Batch processing failed: {str(e)}")
            
            # Create error response for each request
            responses = []
            for request in batch_request.requests:
                error_response = self._create_error_response(request, e, "BATCH_PROCESSING_ERROR")
                responses.append(error_response)
            
            # Create batch response with errors
            batch_response = BatchLLMResponse(
                batch_id=batch_request.batch_id,
                responses=responses,
                errors=[{"batch_error": str(e)}]
            )
            
            return batch_response
    
    @track_llm_progress(operation_type=OperationType.LLM_EMBEDDING, total_steps=4)
    async def generate_embedding(self, text: str, model: str = None, provider: str = None) -> Dict[str, Any]:
        """
        Generate embeddings for text with progress tracking.
        
        Args:
            text: Text to embed
            model: Model to use
            provider: Provider to use
            
        Returns:
            Embedding response
        """
        # Get the current tracker
        tracker = get_progress_tracker()
        
        # Step 1: Initialize embedding request
        tracker.update(1, "Initializing embedding request")
        
        # Add metadata
        tracker.metadata["text_length"] = len(text)
        tracker.metadata["model"] = model
        tracker.metadata["provider"] = provider
        
        logger.info(f"Processing embedding request with progress tracking")
        
        try:
            # Step 2: Selecting provider
            tracker.update(2, f"Selecting provider: {provider or 'default'}")
            
            # Step 3: Generating embedding
            tracker.update(3, f"Generating embedding with {model or 'default model'}")
            
            # Call the parent method (assuming it exists)
            # Note: This is a placeholder - you'll need to implement the actual embedding generation
            # or call the appropriate method in the parent class
            embedding = await self._generate_embedding_internal(text, model, provider)
            
            # Step 4: Post-processing embedding
            tracker.update(4, "Post-processing embedding")
            
            logger.info(f"Embedding generated successfully")
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
            tracker.fail(f"Embedding generation failed: {str(e)}")
            
            # Return error response
            return {
                "error": str(e),
                "status": "error",
                "embedding": None
            }
    
    async def _generate_embedding_internal(self, text: str, model: str = None, provider: str = None) -> Dict[str, Any]:
        """
        Internal method for generating embeddings.
        
        This is a placeholder - you'll need to implement the actual embedding generation
        or call the appropriate method in the parent class.
        
        Args:
            text: Text to embed
            model: Model to use
            provider: Provider to use
            
        Returns:
            Embedding response
        """
        # This is a placeholder - implement the actual embedding generation
        # or call the appropriate method in the parent class
        raise NotImplementedError("Embedding generation not implemented")
    
    async def _process_single_request_in_batch(self, request: LLMRequest) -> Tuple[LLMRequest, LLMResponse]:
        """
        Process a single request in a batch with progress tracking.
        
        Args:
            request: LLM request
            
        Returns:
            Tuple of request and response
        """
        # Get the sub-tracker for this request
        request_id = request.initial_context.request_id
        sub_tracker = self.progress_registry.get_tracker(f"{request.initial_context.batch_id}_{request_id}")
        
        if sub_tracker:
            # Step 1: Initialize request
            sub_tracker.update(1, "Initializing request")
        
        try:
            # Step 2: Pre-processing
            if sub_tracker:
                sub_tracker.update(2, "Pre-processing request")
            
            # Step 3: Provider selection
            if sub_tracker:
                sub_tracker.update(3, "Selecting provider")
            
            # Step 4: Generating response
            if sub_tracker:
                sub_tracker.update(4, "Generating response")
            
            # Process the request
            response = await self.intervention_manager.process_request(request)
            
            # Step 5: Post-processing
            if sub_tracker:
                sub_tracker.update(5, "Post-processing response")
                sub_tracker.complete("Request completed successfully")
            
            return request, response
        except Exception as e:
            # Mark as failed
            if sub_tracker:
                sub_tracker.fail(f"Request failed: {str(e)}")
            
            # Re-raise the exception
            raise e
