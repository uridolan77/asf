"""
Integration of progress tracking with the LLM Gateway.

This module demonstrates how to integrate the progress tracking system
with the LLM Gateway client and providers.
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional

from ..core.client import LLMClient
from ..providers.base import BaseProvider
from .decorators import track_llm_progress, get_progress_tracker
from .models import OperationType

# Set up logging
logger = logging.getLogger(__name__)


class ProgressTrackingLLMClient(LLMClient):
    """
    LLM client with integrated progress tracking.
    
    This class extends the base LLM client with progress tracking capabilities,
    automatically tracking the progress of LLM operations.
    """
    
    @track_llm_progress(operation_type=OperationType.LLM_REQUEST, total_steps=4)
    async def generate_text(
        self,
        prompt: str,
        model: str = None,
        provider: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with progress tracking.
        
        Args:
            prompt: Text prompt
            model: Model to use
            provider: Provider to use
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        # Get the current tracker
        tracker = get_progress_tracker()
        
        # Step 1: Prepare request
        tracker.update(1, "Preparing request")
        
        # Step 2: Select provider
        tracker.update(2, f"Selecting provider: {provider or 'default'}")
        
        # Step 3: Send request to provider
        tracker.update(3, f"Sending request to {model or 'default model'}")
        
        # Call the parent method
        response = await super().generate_text(prompt, model, provider, **kwargs)
        
        # Step 4: Process response
        tracker.update(4, "Processing response")
        
        return response
    
    @track_llm_progress(operation_type=OperationType.LLM_STREAMING, total_steps=4)
    async def generate_text_stream(
        self,
        prompt: str,
        model: str = None,
        provider: str = None,
        **kwargs
    ) -> Any:
        """
        Generate streaming text with progress tracking.
        
        Args:
            prompt: Text prompt
            model: Model to use
            provider: Provider to use
            **kwargs: Additional parameters
            
        Returns:
            Streaming response
        """
        # Get the current tracker
        tracker = get_progress_tracker()
        
        # Step 1: Prepare request
        tracker.update(1, "Preparing streaming request")
        
        # Step 2: Select provider
        tracker.update(2, f"Selecting provider: {provider or 'default'}")
        
        # Step 3: Send request to provider
        tracker.update(3, f"Sending streaming request to {model or 'default model'}")
        
        # Call the parent method
        response = await super().generate_text_stream(prompt, model, provider, **kwargs)
        
        # Step 4: Stream started
        tracker.update(4, "Streaming response started")
        
        return response
    
    @track_llm_progress(operation_type=OperationType.LLM_BATCH, total_steps=5)
    async def batch_generate(
        self,
        prompts: List[str],
        model: str = None,
        provider: str = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts with progress tracking.
        
        Args:
            prompts: List of text prompts
            model: Model to use
            provider: Provider to use
            **kwargs: Additional parameters
            
        Returns:
            List of generated text responses
        """
        # Get the current tracker
        tracker = get_progress_tracker()
        
        # Update metadata with batch size
        tracker.metadata["batch_size"] = len(prompts)
        
        # Step 1: Prepare batch request
        tracker.update(1, f"Preparing batch request for {len(prompts)} prompts")
        
        # Step 2: Select provider
        tracker.update(2, f"Selecting provider: {provider or 'default'}")
        
        # Step 3: Send batch to provider
        tracker.update(3, f"Sending batch to {model or 'default model'}")
        
        # Process each prompt
        results = []
        for i, prompt in enumerate(prompts):
            # Update progress for each prompt
            percent_complete = (i / len(prompts)) * 100
            tracker.update(
                3,  # Still in step 3
                f"Processing prompt {i+1}/{len(prompts)} ({percent_complete:.1f}%)",
                {"current_prompt": i, "total_prompts": len(prompts)}
            )
            
            # Generate text for this prompt
            response = await super().generate_text(prompt, model, provider, **kwargs)
            results.append(response)
        
        # Step 4: Process batch results
        tracker.update(4, "Processing batch results")
        
        # Step 5: Complete batch
        tracker.update(5, f"Batch completed: {len(results)} responses generated")
        
        return results
    
    @track_llm_progress(operation_type=OperationType.LLM_EMBEDDING, total_steps=3)
    async def generate_embedding(
        self,
        text: str,
        model: str = None,
        provider: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate embedding with progress tracking.
        
        Args:
            text: Text to embed
            model: Model to use
            provider: Provider to use
            **kwargs: Additional parameters
            
        Returns:
            Embedding response
        """
        # Get the current tracker
        tracker = get_progress_tracker()
        
        # Step 1: Prepare request
        tracker.update(1, "Preparing embedding request")
        
        # Step 2: Send request to provider
        tracker.update(2, f"Generating embedding with {model or 'default model'}")
        
        # Call the parent method
        response = await super().generate_embedding(text, model, provider, **kwargs)
        
        # Step 3: Process response
        tracker.update(3, "Processing embedding response")
        
        return response


class ProgressTrackingProvider(BaseProvider):
    """
    Base provider with integrated progress tracking.
    
    This class extends the base provider with progress tracking capabilities,
    allowing for detailed tracking of provider operations.
    """
    
    @track_llm_progress(operation_type=OperationType.PROVIDER_CONNECTION, total_steps=3)
    async def connect(self) -> None:
        """
        Connect to the provider with progress tracking.
        """
        # Get the current tracker
        tracker = get_progress_tracker()
        
        # Step 1: Initializing connection
        tracker.update(1, f"Initializing connection to {self.name}")
        
        # Step 2: Authenticating
        tracker.update(2, "Authenticating with provider")
        
        # Call the parent method
        await super().connect()
        
        # Step 3: Connection established
        tracker.update(3, f"Connection to {self.name} established")
    
    @track_llm_progress(operation_type=OperationType.LLM_REQUEST, total_steps=5)
    async def generate_text(
        self,
        prompt: str,
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with progress tracking.
        
        Args:
            prompt: Text prompt
            model: Model to use
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        # Get the current tracker
        tracker = get_progress_tracker()
        
        # Step 1: Validate request
        tracker.update(1, "Validating request")
        
        # Step 2: Prepare request
        tracker.update(2, "Preparing provider request")
        
        # Step 3: Select model
        model = model or self.default_model
        tracker.update(3, f"Selected model: {model}")
        
        # Step 4: Send request to API
        tracker.update(4, f"Sending request to {self.name} API")
        
        # Call the parent method
        response = await super().generate_text(prompt, model, **kwargs)
        
        # Step 5: Process response
        tracker.update(5, "Processing provider response")
        
        return response


# Example usage
async def example_usage():
    """Example of using progress tracking with the LLM Gateway."""
    # Create a progress tracking client
    client = ProgressTrackingLLMClient()
    
    # Generate text
    response = await client.generate_text(
        prompt="Tell me about progress tracking",
        model="gpt-4",
        provider="openai"
    )
    
    print(f"Generated text: {response['text'][:100]}...")
    
    # Generate embeddings
    embedding = await client.generate_embedding(
        text="Progress tracking is important",
        model="text-embedding-ada-002",
        provider="openai"
    )
    
    print(f"Generated embedding: {len(embedding['embedding'])} dimensions")
    
    # Batch generate
    batch_responses = await client.batch_generate(
        prompts=[
            "What is progress tracking?",
            "Why is progress tracking important?",
            "How can I implement progress tracking?"
        ],
        model="gpt-4",
        provider="openai"
    )
    
    print(f"Generated {len(batch_responses)} batch responses")


if __name__ == "__main__":
    asyncio.run(example_usage())
