"""
Progress-tracking enhanced provider base class.

This module provides an enhanced version of the base provider that
incorporates progress tracking for all provider operations.
"""

import logging
import asyncio
from datetime import datetime
from typing import AsyncGenerator, Optional, Dict, Any

from asf.medical.llm_gateway.providers.base import BaseProvider
from asf.medical.llm_gateway.core.models import (
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    StreamChunk,
)

from asf.medical.llm_gateway.progress import (
    track_llm_progress, get_progress_tracker, ProgressTracker,
    get_progress_registry, OperationType
)

logger = logging.getLogger(__name__)


class ProgressTrackingProvider(BaseProvider):
    """
    Enhanced provider base class with integrated progress tracking.
    
    This class extends the base provider with progress tracking capabilities,
    automatically tracking the progress of all provider operations.
    """
    
    def __init__(self, provider_config: ProviderConfig):
        """
        Initialize the progress tracking provider.
        
        Args:
            provider_config: Provider configuration
        """
        super().__init__(provider_config)
        logger.info(f"Initializing progress tracking for provider: {self.provider_id}")
        
        # Initialize progress registry
        self.progress_registry = get_progress_registry()
    
    @track_llm_progress(operation_type=OperationType.PROVIDER_INITIALIZATION, total_steps=3)
    def _initialize_client(self) -> None:
        """
        Initialize the client for this provider with progress tracking.
        
        This method should be called by the provider implementation to
        track the initialization process.
        """
        # Get the current tracker
        tracker = get_progress_tracker()
        
        # Step 1: Loading configuration
        tracker.update(1, f"Loading configuration for provider: {self.provider_id}")
        
        # Step 2: Initializing client
        tracker.update(2, f"Initializing client for provider: {self.provider_id}")
        
        # Call the provider-specific initialization
        self._initialize_client_internal()
        
        # Step 3: Verifying connection
        tracker.update(3, f"Verifying connection for provider: {self.provider_id}")
        
        logger.info(f"Provider {self.provider_id} initialized successfully")
    
    def _initialize_client_internal(self) -> None:
        """
        Internal method for initializing the client.
        
        This method should be overridden by provider implementations
        with their specific initialization logic.
        """
        # Default implementation does nothing
        pass
    
    @track_llm_progress(operation_type=OperationType.LLM_REQUEST, total_steps=5)
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response for the given request with progress tracking.
        
        Args:
            request: The request containing prompt, conversation history, and config
            
        Returns:
            A complete response with generated content or error details
        """
        # Get the current tracker
        tracker = get_progress_tracker()
        
        # Step 1: Validating request
        tracker.update(1, "Validating request")
        
        # Add metadata
        tracker.metadata["request_id"] = request.initial_context.request_id
        tracker.metadata["model"] = request.config.model_identifier
        
        logger.info(f"Generating response for request: {request.initial_context.request_id}")
        
        try:
            # Step 2: Preparing request
            tracker.update(2, "Preparing request for provider")
            
            # Step 3: Sending request to API
            tracker.update(3, f"Sending request to {self.provider_id} API")
            
            # Step 4: Generating response
            tracker.update(4, "Generating response")
            
            # Call the provider-specific implementation
            response = await self._generate_internal(request)
            
            # Step 5: Processing response
            tracker.update(5, "Processing response")
            
            logger.info(f"Response generated successfully for request: {request.initial_context.request_id}")
            return response
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}", exc_info=True)
            tracker.fail(f"Response generation failed: {str(e)}")
            
            # Re-raise the exception
            raise e
    
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
        raise NotImplementedError("Provider must implement _generate_internal")
    
    @track_llm_progress(operation_type=OperationType.LLM_STREAMING, total_steps=5)
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate a streaming response for the given request with progress tracking.
        
        Args:
            request: The request containing prompt, conversation history, and config
            
        Yields:
            Chunks of the generated response
        """
        # Get the current tracker
        tracker = get_progress_tracker()
        
        # Step 1: Validating request
        tracker.update(1, "Validating streaming request")
        
        # Add metadata
        tracker.metadata["request_id"] = request.initial_context.request_id
        tracker.metadata["model"] = request.config.model_identifier
        tracker.metadata["streaming"] = True
        
        logger.info(f"Generating streaming response for request: {request.initial_context.request_id}")
        
        try:
            # Step 2: Preparing request
            tracker.update(2, "Preparing streaming request for provider")
            
            # Step 3: Sending request to API
            tracker.update(3, f"Sending streaming request to {self.provider_id} API")
            
            # Step 4: Starting stream
            tracker.update(4, "Starting response stream")
            
            # Call the provider-specific implementation
            chunk_count = 0
            async for chunk in self._generate_stream_internal(request):
                chunk_count += 1
                
                # Update progress periodically (every 10 chunks)
                if chunk_count % 10 == 0:
                    tracker.update(4, f"Streaming response (chunks: {chunk_count})")
                
                yield chunk
            
            # Step 5: Completing stream
            tracker.update(5, f"Stream completed (total chunks: {chunk_count})")
            
            logger.info(f"Streaming response completed for request: {request.initial_context.request_id}")
        except Exception as e:
            logger.error(f"Streaming response failed: {str(e)}", exc_info=True)
            tracker.fail(f"Streaming response failed: {str(e)}")
            
            # Re-raise the exception
            raise e
    
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
        raise NotImplementedError("Provider must implement _generate_stream_internal")
    
    @track_llm_progress(operation_type=OperationType.PROVIDER_CONNECTION, total_steps=3)
    async def cleanup(self) -> None:
        """
        Clean up resources used by this provider with progress tracking.
        
        This method should be called when the provider is no longer needed
        to ensure proper resource cleanup.
        """
        # Get the current tracker
        tracker = get_progress_tracker()
        
        # Step 1: Preparing cleanup
        tracker.update(1, f"Preparing cleanup for provider: {self.provider_id}")
        
        logger.info(f"Cleaning up resources for provider: {self.provider_id}")
        
        try:
            # Step 2: Cleaning up resources
            tracker.update(2, "Cleaning up provider resources")
            
            # Call the provider-specific implementation
            await self._cleanup_internal()
            
            # Step 3: Verifying cleanup
            tracker.update(3, "Verifying cleanup")
            
            logger.info(f"Provider {self.provider_id} cleaned up successfully")
        except Exception as e:
            logger.error(f"Provider cleanup failed: {str(e)}", exc_info=True)
            tracker.fail(f"Provider cleanup failed: {str(e)}")
            
            # Re-raise the exception
            raise e
    
    async def _cleanup_internal(self) -> None:
        """
        Internal method for cleaning up resources.
        
        This method should be overridden by provider implementations
        with their specific cleanup logic.
        """
        # Default implementation does nothing
        pass
    
    @track_llm_progress(operation_type=OperationType.PROVIDER_CONNECTION, total_steps=3)
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the provider is healthy and available with progress tracking.
        
        Returns:
            A dictionary with health check information
        """
        # Get the current tracker
        tracker = get_progress_tracker()
        
        # Step 1: Preparing health check
        tracker.update(1, f"Preparing health check for provider: {self.provider_id}")
        
        logger.info(f"Performing health check for provider: {self.provider_id}")
        
        try:
            # Step 2: Checking connection
            tracker.update(2, "Checking provider connection")
            
            # Call the provider-specific implementation
            health_info = await self._health_check_internal()
            
            # Step 3: Verifying health
            tracker.update(3, "Verifying provider health")
            
            logger.info(f"Health check completed for provider: {self.provider_id}")
            return health_info
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}", exc_info=True)
            tracker.fail(f"Health check failed: {str(e)}")
            
            # Return error information
            return {
                "provider_id": self.provider_id,
                "status": "unavailable",
                "created_at": self.created_at.isoformat(),
                "message": f"Health check failed: {str(e)}"
            }
    
    async def _health_check_internal(self) -> Dict[str, Any]:
        """
        Internal method for checking provider health.
        
        This method should be overridden by provider implementations
        with their specific health check logic.
        
        Returns:
            A dictionary with health check information
        """
        # Default implementation returns basic health info
        return {
            "provider_id": self.provider_id,
            "status": "available",
            "created_at": self.created_at.isoformat(),
            "message": "Provider health check not implemented"
        }
