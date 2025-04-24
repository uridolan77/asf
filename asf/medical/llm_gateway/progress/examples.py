"""
Examples of using the progress tracking system in the LLM Gateway.

This module provides examples of how to use the progress tracking system,
including tracking progress of synchronous and asynchronous functions,
and accessing progress information.
"""

import time
import asyncio
import random
from typing import Dict, Any

from .tracker import ProgressTracker
from .registry import get_progress_registry
from .decorators import track_progress, track_llm_progress, get_progress_tracker
from .models import OperationType


# Example 1: Basic usage with manual tracking
def example_basic_usage():
    """Example of basic usage with manual tracking."""
    # Create a progress tracker
    tracker = ProgressTracker(
        operation_id="example-basic",
        operation_type=OperationType.GENERAL,
        total_steps=5
    )
    
    # Update progress
    for i in range(1, 6):
        # Simulate work
        time.sleep(0.5)
        
        # Update progress
        tracker.update(i, f"Step {i} completed")
        
        # Print progress
        progress = tracker.get_progress_details()
        print(f"Progress: {progress.percent_complete:.1f}% - {progress.message}")
    
    # Mark as completed
    tracker.complete("All steps completed")
    
    return tracker.get_progress_details()


# Example 2: Using the registry
def example_using_registry():
    """Example of using the progress registry."""
    # Get the registry
    registry = get_progress_registry()
    
    # Create a tracker through the registry
    tracker = registry.create_tracker(
        operation_id="example-registry",
        operation_type=OperationType.GENERAL,
        total_steps=3
    )
    
    # Update progress
    for i in range(1, 4):
        # Simulate work
        time.sleep(0.5)
        
        # Update progress
        tracker.update(i, f"Registry step {i} completed")
    
    # Mark as completed
    tracker.complete("All registry steps completed")
    
    # Get all trackers
    all_trackers = registry.get_all_trackers()
    print(f"Number of trackers in registry: {len(all_trackers)}")
    
    return tracker.get_progress_details()


# Example 3: Using the decorator with a synchronous function
@track_progress(operation_type=OperationType.GENERAL, total_steps=4)
def example_sync_function(iterations: int = 4):
    """Example of using the decorator with a synchronous function."""
    # Get the current tracker
    tracker = get_progress_tracker()
    
    # Process iterations
    for i in range(1, iterations + 1):
        # Simulate work
        time.sleep(0.5)
        
        # Update progress
        tracker.update(i, f"Sync iteration {i} completed")
        
        # Print progress
        progress = tracker.get_progress_details()
        print(f"Sync progress: {progress.percent_complete:.1f}% - {progress.message}")
    
    # Return result
    return {"iterations_completed": iterations}


# Example 4: Using the decorator with an asynchronous function
@track_progress(operation_type=OperationType.GENERAL, total_steps=4)
async def example_async_function(iterations: int = 4):
    """Example of using the decorator with an asynchronous function."""
    # Get the current tracker
    tracker = get_progress_tracker()
    
    # Process iterations
    for i in range(1, iterations + 1):
        # Simulate async work
        await asyncio.sleep(0.5)
        
        # Update progress
        tracker.update(i, f"Async iteration {i} completed")
        
        # Print progress
        progress = tracker.get_progress_details()
        print(f"Async progress: {progress.percent_complete:.1f}% - {progress.message}")
    
    # Return result
    return {"iterations_completed": iterations}


# Example 5: Using the LLM-specific decorator
@track_llm_progress(operation_type=OperationType.LLM_REQUEST, total_steps=3)
async def example_llm_function(prompt: str, model: str = "gpt-4", temperature: float = 0.7):
    """Example of using the LLM-specific decorator."""
    # Get the current tracker
    tracker = get_progress_tracker()
    
    # Step 1: Prepare request
    tracker.update(1, "Preparing LLM request")
    await asyncio.sleep(0.5)
    
    # Step 2: Send request to LLM
    tracker.update(2, "Sending request to LLM")
    await asyncio.sleep(1.0)
    
    # Step 3: Process response
    tracker.update(3, "Processing LLM response")
    await asyncio.sleep(0.5)
    
    # Generate a mock response
    response = f"This is a mock response to: {prompt}"
    
    # Return result
    return {"model": model, "prompt": prompt, "response": response}


# Example 6: Handling errors
@track_progress(operation_type=OperationType.GENERAL, total_steps=3)
def example_error_handling():
    """Example of handling errors in tracked functions."""
    # Get the current tracker
    tracker = get_progress_tracker()
    
    # Step 1: Start processing
    tracker.update(1, "Starting processing")
    time.sleep(0.5)
    
    # Step 2: Process data
    tracker.update(2, "Processing data")
    time.sleep(0.5)
    
    # Simulate an error
    raise ValueError("Simulated error in tracked function")
    
    # This code will not be reached
    tracker.update(3, "Finalizing processing")
    return {"status": "completed"}


# Run all examples
async def run_all_examples():
    """Run all examples."""
    print("Running Example 1: Basic usage with manual tracking")
    example_basic_usage()
    print("\n")
    
    print("Running Example 2: Using the registry")
    example_using_registry()
    print("\n")
    
    print("Running Example 3: Using the decorator with a synchronous function")
    example_sync_function()
    print("\n")
    
    print("Running Example 4: Using the decorator with an asynchronous function")
    await example_async_function()
    print("\n")
    
    print("Running Example 5: Using the LLM-specific decorator")
    await example_llm_function("Tell me about progress tracking")
    print("\n")
    
    print("Running Example 6: Handling errors")
    try:
        example_error_handling()
    except ValueError as e:
        print(f"Caught expected error: {e}")
    print("\n")
    
    # Print summary of all trackers
    registry = get_progress_registry()
    all_trackers = registry.get_all_trackers()
    print(f"Total trackers created: {len(all_trackers)}")
    
    for tracker in all_trackers:
        progress = tracker.get_progress_details()
        print(f"Operation {progress.operation_id}: {progress.status} - {progress.percent_complete:.1f}%")


# Entry point for running the examples
if __name__ == "__main__":
    asyncio.run(run_all_examples())
