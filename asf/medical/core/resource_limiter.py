"""
Resource Limiter module for the Medical Research Synthesizer.

This module provides functionality to limit resource usage for system stability
and to prevent resource exhaustion by applying throttling and quota mechanisms.

Classes:
    ResourceLimiter: Controls and limits usage of system resources.
    ResourceLimitExceededError: Exception raised when a resource limit is exceeded.

Functions:
    apply_resource_limits: Decorator to apply resource limits to a function.
"""
import contextlib
import os
import time
import logging
import threading
import psutil
from typing import Dict, Any, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ResourceLimitExceededError(Exception):
    """
    Exception raised when a resource limit is exceeded.

    Attributes:
        resource (str): The resource that was limited.
        limit (int): The limit that was exceeded.
        usage (int): The attempted usage that exceeded the limit.
        message (str): Explanation of the error.
    """

    def __init__(self, resource: str, limit: int, usage: int, message: str = None):
        """
        Initialize the ResourceLimitExceededError.

        Args:
            resource (str): The resource that was limited.
            limit (int): The limit that was exceeded.
            usage (int): The attempted usage that exceeded the limit.
            message (str, optional): Explanation of the error. Defaults to None.
        """
        self.resource = resource
        self.limit = limit
        self.usage = usage
        self.message = message or f"{resource} limit of {limit} exceeded with usage {usage}"
        super().__init__(self.message)

class ResourceLimiter:
    """
    Controls and limits usage of system resources.

    This class provides methods to define, track, and enforce limits on
    various system resources like memory, CPU, and network connections.

    Attributes:
        limits (Dict[str, int]): Resource limits configuration.
        usage (Dict[str, int]): Current resource usage.
        locks (Dict[str, threading.Lock]): Locks for thread-safe access to resource accounting.
    """

    def __init__(self, limits: Dict[str, int] = None):
        """
        Initialize the ResourceLimiter instance.

        Args:
            limits (Dict[str, int], optional): Resource limits configuration. Defaults to None.
        """
        self.limits = limits or {}
        self.usage = {resource: 0 for resource in self.limits}
        self.locks = {resource: threading.Lock() for resource in self.limits}

    def set_limit(self, resource: str, limit: int) -> None:
        """
        Set a limit for a specific resource.

        Args:
            resource (str): The resource to limit.
            limit (int): The maximum amount of the resource that can be used.
        """
        self.limits[resource] = limit
        if resource not in self.usage:
            self.usage[resource] = 0
        if resource not in self.locks:
            self.locks[resource] = threading.Lock()

    def get_limit(self, resource: str) -> int:
        """
        Get the limit for a specific resource.

        Args:
            resource (str): The resource to get the limit for.

        Returns:
            int: The limit for the resource.
        """
        return self.limits.get(resource, 0)

    def get_usage(self, resource: str) -> int:
        """
        Get the current usage for a specific resource.

        Args:
            resource (str): The resource to get the usage for.

        Returns:
            int: The current usage of the resource.
        """
        return self.usage.get(resource, 0)

    def check_limit(self, resource: str, amount: int = 1) -> bool:
        """
        Check if using a certain amount of a resource would exceed its limit.

        Args:
            resource (str): The resource to check.
            amount (int, optional): The amount of the resource to check. Defaults to 1.

        Returns:
            bool: True if the resource usage would be within limits, False otherwise.
        """
        limit = self.get_limit(resource)
        usage = self.get_usage(resource)
        return usage + amount <= limit

    def acquire(self, resource: str, amount: int = 1) -> bool:
        """
        Acquire a certain amount of a resource if it's within limits.

        Args:
            resource (str): The resource to acquire.
            amount (int, optional): The amount of the resource to acquire. Defaults to 1.

        Returns:
            bool: True if the resource was acquired, False otherwise.

        Raises:
            ResourceLimitExceededError: If the resource limit would be exceeded.
        """
        with self.locks[resource]:
            if not self.check_limit(resource, amount):
                raise ResourceLimitExceededError(resource, self.get_limit(resource), self.get_usage(resource) + amount)
            self.usage[resource] += amount
            return True

    def release(self, resource: str, amount: int = 1) -> None:
        """
        Release a certain amount of a previously acquired resource.

        Args:
            resource (str): The resource to release.
            amount (int, optional): The amount of the resource to release. Defaults to 1.
        """
        with self.locks[resource]:
            self.usage[resource] = max(0, self.usage[resource] - amount)

    @contextmanager
    def resource_context(self, resource: str, amount: int = 1):
        """
        Context manager for resource acquisition and release.

        Args:
            resource (str): The resource to acquire and release.
            amount (int, optional): The amount of the resource to acquire and release. Defaults to 1.

        Yields:
            None

        Raises:
            ResourceLimitExceededError: If the resource limit would be exceeded.
        """
        self.acquire(resource, amount)
        try:
            yield
        finally:
            self.release(resource, amount)

def apply_resource_limits(**resource_amounts):
    """
    Decorator to apply resource limits to a function.

    Args:
        **resource_amounts: Keyword arguments specifying the resource amounts to acquire.

    Returns:
        Callable: A decorated function that acquires resources before execution and releases them after.

    Raises:
        ResourceLimitExceededError: If any resource limit would be exceeded.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            limiter = ResourceLimiter()
            with contextlib.ExitStack() as stack:
                for resource, amount in resource_amounts.items():
                    stack.enter_context(limiter.resource_context(resource, amount))
                return func(*args, **kwargs)
        return wrapper
    return decorator