"""
Progress tracking module for the LLM Gateway.

This module provides components for tracking the progress of long-running operations,
such as complex LLM requests, batch processing, and model fine-tuning.

Classes:
    ProgressTracker: Base class for tracking operation progress.
    ProgressRegistry: Registry for managing multiple progress trackers.

Functions:
    get_progress_registry: Get the singleton instance of the ProgressRegistry.
    track_progress: Decorator for tracking progress of a function.
"""

from .tracker import (
    ProgressTracker,
    ProgressState,
    ProgressStep,
    ProgressError
)

from .registry import (
    ProgressRegistry,
    get_progress_registry
)

from .decorators import (
    track_progress,
    track_llm_progress
)

from .models import (
    ProgressDetails,
    OperationType,
    ProgressUpdate
)

__all__ = [
    'ProgressTracker',
    'ProgressState',
    'ProgressStep',
    'ProgressError',
    'ProgressRegistry',
    'get_progress_registry',
    'track_progress',
    'track_llm_progress',
    'ProgressDetails',
    'OperationType',
    'ProgressUpdate'
]
