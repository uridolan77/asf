"""
Decorators for progress tracking in the LLM Gateway.

This module provides decorators for easily adding progress tracking to functions,
including specialized decorators for LLM operations.
"""

import time
import functools
import hashlib
import inspect
import logging
from typing import Callable, Any, Dict, Optional, Union, TypeVar, cast

from .registry import get_progress_registry
from .models import OperationType

# Set up logging
logger = logging.getLogger(__name__)

# Type variables for better type hinting
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])


def track_progress(
    operation_type: Union[OperationType, str] = OperationType.GENERAL,
    total_steps: int = 100,
    operation_id: Optional[str] = None,
    metadata_fn: Optional[Callable[..., Dict[str, Any]]] = None
) -> Callable[[F], F]:
    """
    Decorator for tracking progress of a function.
    
    This decorator adds progress tracking to a function, creating a progress tracker
    and updating it as the function executes. It supports both synchronous and
    asynchronous functions.
    
    Args:
        operation_type: Type of operation being tracked
        total_steps: Total number of steps in the operation
        operation_id: Optional fixed operation ID (if not provided, one will be generated)
        metadata_fn: Optional function to extract metadata from function arguments
        
    Returns:
        Decorated function with progress tracking
        
    Example:
        ```python
        @track_progress(operation_type=OperationType.LLM_REQUEST, total_steps=3)
        async def process_request(request_data):
            # Function implementation with progress updates
            tracker = get_progress_tracker()
            tracker.update(1, "Processing request")
            # ...
            tracker.update(2, "Generating response")
            # ...
            tracker.complete("Request completed")
            return result
        ```
    """
    def decorator(func: F) -> F:
        is_async = inspect.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Generate operation ID if not provided
                nonlocal operation_id
                op_id = operation_id
                if op_id is None:
                    # Create a deterministic operation ID based on function name and arguments
                    func_name = func.__name__
                    arg_str = str(args) + str(sorted(kwargs.items()))
                    op_id = f"{func_name}:{hashlib.md5(arg_str.encode()).hexdigest()}"
                
                # Extract metadata if metadata_fn is provided
                metadata = None
                if metadata_fn:
                    try:
                        metadata = metadata_fn(*args, **kwargs)
                    except Exception as e:
                        logger.warning(f"Error extracting metadata: {str(e)}")
                
                # Get progress registry and create tracker
                registry = get_progress_registry()
                tracker = registry.create_tracker(
                    operation_id=op_id,
                    operation_type=operation_type,
                    total_steps=total_steps,
                    metadata=metadata
                )
                
                # Store tracker in context for access within the function
                _set_current_tracker(tracker)
                
                try:
                    # Call the function
                    result = await func(*args, **kwargs)
                    
                    # Mark as completed if not already
                    if tracker.status not in ("completed", "failed", "cancelled"):
                        tracker.complete("Operation completed successfully")
                    
                    return result
                except Exception as e:
                    # Mark as failed
                    tracker.fail(f"Operation failed: {str(e)}")
                    raise
                finally:
                    # Clear current tracker
                    _clear_current_tracker()
            
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Generate operation ID if not provided
                nonlocal operation_id
                op_id = operation_id
                if op_id is None:
                    # Create a deterministic operation ID based on function name and arguments
                    func_name = func.__name__
                    arg_str = str(args) + str(sorted(kwargs.items()))
                    op_id = f"{func_name}:{hashlib.md5(arg_str.encode()).hexdigest()}"
                
                # Extract metadata if metadata_fn is provided
                metadata = None
                if metadata_fn:
                    try:
                        metadata = metadata_fn(*args, **kwargs)
                    except Exception as e:
                        logger.warning(f"Error extracting metadata: {str(e)}")
                
                # Get progress registry and create tracker
                registry = get_progress_registry()
                tracker = registry.create_tracker(
                    operation_id=op_id,
                    operation_type=operation_type,
                    total_steps=total_steps,
                    metadata=metadata
                )
                
                # Store tracker in context for access within the function
                _set_current_tracker(tracker)
                
                try:
                    # Call the function
                    result = func(*args, **kwargs)
                    
                    # Mark as completed if not already
                    if tracker.status not in ("completed", "failed", "cancelled"):
                        tracker.complete("Operation completed successfully")
                    
                    return result
                except Exception as e:
                    # Mark as failed
                    tracker.fail(f"Operation failed: {str(e)}")
                    raise
                finally:
                    # Clear current tracker
                    _clear_current_tracker()
            
            return cast(F, sync_wrapper)
    
    return decorator


def track_llm_progress(
    operation_type: Union[OperationType, str] = OperationType.LLM_REQUEST,
    total_steps: int = 5,
    operation_id: Optional[str] = None
) -> Callable[[F], F]:
    """
    Specialized decorator for tracking progress of LLM operations.
    
    This decorator is a specialized version of track_progress for LLM operations,
    with defaults appropriate for LLM requests.
    
    Args:
        operation_type: Type of LLM operation being tracked
        total_steps: Total number of steps in the operation
        operation_id: Optional fixed operation ID (if not provided, one will be generated)
        
    Returns:
        Decorated function with progress tracking
        
    Example:
        ```python
        @track_llm_progress(operation_type=OperationType.LLM_STREAMING)
        async def generate_text_stream(self, prompt, **kwargs):
            # Function implementation with progress updates
            tracker = get_progress_tracker()
            tracker.update(1, "Preparing request")
            # ...
            return result
        ```
    """
    def llm_metadata_fn(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Extract metadata from LLM function arguments."""
        metadata = {}
        
        # Extract common LLM parameters
        for key in ["model", "provider", "max_tokens", "temperature"]:
            if key in kwargs:
                metadata[key] = kwargs[key]
        
        # Extract prompt or messages
        if "prompt" in kwargs:
            prompt = kwargs["prompt"]
            if isinstance(prompt, str):
                # Truncate long prompts
                metadata["prompt"] = (prompt[:100] + "...") if len(prompt) > 100 else prompt
                metadata["prompt_length"] = len(prompt)
        
        if "messages" in kwargs:
            messages = kwargs["messages"]
            if isinstance(messages, list):
                metadata["message_count"] = len(messages)
                if messages:
                    # Include role of last message
                    if isinstance(messages[-1], dict) and "role" in messages[-1]:
                        metadata["last_role"] = messages[-1]["role"]
        
        return metadata
    
    return track_progress(
        operation_type=operation_type,
        total_steps=total_steps,
        operation_id=operation_id,
        metadata_fn=llm_metadata_fn
    )


# Thread-local storage for current tracker
import threading
_thread_local = threading.local()


def _set_current_tracker(tracker):
    """Set the current tracker for this thread."""
    _thread_local.current_tracker = tracker


def _clear_current_tracker():
    """Clear the current tracker for this thread."""
    if hasattr(_thread_local, "current_tracker"):
        delattr(_thread_local, "current_tracker")


def get_progress_tracker():
    """
    Get the current progress tracker for this thread.
    
    Returns:
        Current progress tracker or None if not in a tracked function
        
    Example:
        ```python
        @track_progress()
        async def my_function():
            # Get the current tracker
            tracker = get_progress_tracker()
            tracker.update(1, "Step 1 completed")
            # ...
        ```
    """
    return getattr(_thread_local, "current_tracker", None)
