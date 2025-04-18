"""
Enhanced caching module for the Medical Research Synthesizer.

This module extends the base caching system with additional features specifically
designed for medical data caching, including:
- Domain-specific TTL strategies
- Medical data sensitivity handling
- Enhanced cache invalidation for medical research data
- Audit logging for cached medical data access

Classes:
    EnhancedCacheManager: Extended cache manager with medical-specific features.

Functions:
    enhanced_cached: Decorator for caching function results with enhanced features.
    get_enhanced_cache_manager: Get the singleton instance of the EnhancedCacheManager.
"""

import os
import json
import time
import logging
import functools
import hashlib
from typing import Dict, Any, Optional, List, Callable, TypeVar, Union, Tuple

# Import cache functionality
from .cache import CacheManager, get_cache_manager, get_cache_key

logger = logging.getLogger(__name__)
T = TypeVar('T')

# Domain-specific TTL values (in seconds)
DOMAIN_TTL = {
    "cardiology": 7200,        # 2 hours
    "oncology": 3600,          # 1 hour
    "neurology": 5400,         # 1.5 hours
    "pediatrics": 7200,        # 2 hours
    "psychiatry": 3600,        # 1 hour
    "default": 3600            # 1 hour
}

# Sensitivity levels for medical data
class DataSensitivity:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PHI = "phi"  # Protected Health Information

class EnhancedCacheManager:
    """
    Enhanced cache manager with medical-specific features.

    This class extends the base CacheManager with additional features for
    medical data caching, including domain-specific TTL strategies,
    medical data sensitivity handling, and enhanced cache invalidation.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EnhancedCacheManager, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        base_cache_manager: Optional[CacheManager] = None,
        domain: Optional[str] = None,
        sensitivity_level: str = DataSensitivity.LOW,
        audit_logging: bool = True
    ):
        """
        Initialize the enhanced cache manager.

        Args:
            base_cache_manager: Base cache manager to extend
            domain: Medical domain for domain-specific TTL
            sensitivity_level: Data sensitivity level
            audit_logging: Whether to enable audit logging
        """
        self.base_cache_manager = base_cache_manager or get_cache_manager()
        self.domain = domain
        self.sensitivity_level = sensitivity_level
        self.audit_logging = audit_logging
        self.access_log = []

    def get_domain_ttl(self, domain: Optional[str] = None) -> int:
        """
        Get the TTL for a specific medical domain.

        Args:
            domain: Medical domain

        Returns:
            TTL in seconds
        """
        domain_key = domain or self.domain or "default"
        return DOMAIN_TTL.get(domain_key.lower(), DOMAIN_TTL["default"])

    async def get(
        self,
        key: str,
        namespace: Optional[str] = None,
        data_type: Optional[str] = None,
        domain: Optional[str] = None,
        sensitivity_level: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get a value from the cache with enhanced features.

        Args:
            key: Cache key
            namespace: Optional namespace
            data_type: Optional data type
            domain: Optional medical domain
            sensitivity_level: Optional sensitivity level override

        Returns:
            Cached value or None if not found
        """
        # Log access for audit purposes
        if self.audit_logging:
            self._log_access("get", key, namespace, sensitivity_level or self.sensitivity_level)

        # Check if access is allowed based on sensitivity
        if not self._check_sensitivity_access(sensitivity_level or self.sensitivity_level):
            logger.warning(f"Access denied to cache key {key} due to sensitivity level")
            return None

        return await self.base_cache_manager.get(key, namespace, data_type)

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None,
        data_type: Optional[str] = None,
        domain: Optional[str] = None,
        sensitivity_level: Optional[str] = None
    ) -> bool:
        """
        Set a value in the cache with enhanced features.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            namespace: Optional namespace
            data_type: Optional data type
            domain: Optional medical domain
            sensitivity_level: Optional sensitivity level override

        Returns:
            True if successful, False otherwise
        """
        # Log access for audit purposes
        if self.audit_logging:
            self._log_access("set", key, namespace, sensitivity_level or self.sensitivity_level)

        # Use domain-specific TTL if not explicitly provided
        if ttl is None:
            ttl = self.get_domain_ttl(domain)

        # Adjust TTL based on sensitivity level
        ttl = self._adjust_ttl_for_sensitivity(ttl, sensitivity_level or self.sensitivity_level)

        return await self.base_cache_manager.set(
            key, value, ttl, namespace, data_type
        )

    async def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: Cache key
            namespace: Optional namespace

        Returns:
            True if successful, False otherwise
        """
        # Log access for audit purposes
        if self.audit_logging:
            self._log_access("delete", key, namespace, self.sensitivity_level)

        return await self.base_cache_manager.delete(key, namespace)

    async def clear(self, namespace: Optional[str] = None) -> int:
        """
        Clear the cache.

        Args:
            namespace: Optional namespace

        Returns:
            Number of keys cleared
        """
        # Log access for audit purposes
        if self.audit_logging:
            self._log_access("clear", "*", namespace, self.sensitivity_level)

        return await self.base_cache_manager.clear(namespace)

    async def invalidate_domain_data(self, domain: str) -> int:
        """
        Invalidate all cached data for a specific medical domain.

        Args:
            domain: Medical domain to invalidate

        Returns:
            Number of keys invalidated
        """
        # In a real implementation, this would use a more sophisticated
        # approach to identify domain-specific keys
        namespace = f"domain:{domain}"
        return await self.clear(namespace)

    def _log_access(
        self,
        operation: str,
        key: str,
        namespace: Optional[str],
        sensitivity_level: str
    ) -> None:
        """Log cache access for audit purposes."""
        timestamp = time.time()
        log_entry = {
            "timestamp": timestamp,
            "operation": operation,
            "key": key,
            "namespace": namespace,
            "sensitivity_level": sensitivity_level
        }
        self.access_log.append(log_entry)

        # Keep log size manageable
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-1000:]

        # Log high sensitivity access
        if sensitivity_level in (DataSensitivity.HIGH, DataSensitivity.PHI):
            logger.info(f"High sensitivity cache access: {operation} {key}")

    def _check_sensitivity_access(self, sensitivity_level: str) -> bool:
        """
        Check if access is allowed based on sensitivity level.

        In a real implementation, this would integrate with the application's
        security model to enforce access controls.
        """
        # For now, always allow access but log PHI access
        if sensitivity_level == DataSensitivity.PHI:
            logger.warning("PHI data accessed from cache")
        return True

    def _adjust_ttl_for_sensitivity(self, ttl: int, sensitivity_level: str) -> int:
        """Adjust TTL based on sensitivity level."""
        if sensitivity_level == DataSensitivity.HIGH:
            # Reduce TTL for high sensitivity data
            return min(ttl, 1800)  # Max 30 minutes
        elif sensitivity_level == DataSensitivity.PHI:
            # Minimal TTL for PHI data
            return min(ttl, 300)   # Max 5 minutes
        return ttl

    async def get_audit_log(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        operations: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the audit log with optional filtering.

        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            operations: Optional operations filter

        Returns:
            Filtered audit log entries
        """
        filtered_log = self.access_log

        if start_time is not None:
            filtered_log = [entry for entry in filtered_log if entry["timestamp"] >= start_time]

        if end_time is not None:
            filtered_log = [entry for entry in filtered_log if entry["timestamp"] <= end_time]

        if operations is not None:
            filtered_log = [entry for entry in filtered_log if entry["operation"] in operations]

        return filtered_log

def get_enhanced_cache_manager(**kwargs) -> EnhancedCacheManager:
    """
    Get the singleton instance of the EnhancedCacheManager.

    Args:
        **kwargs: Arguments to pass to the EnhancedCacheManager constructor

    Returns:
        EnhancedCacheManager instance
    """
    return EnhancedCacheManager(**kwargs)

def enhanced_cached(
    ttl: Optional[int] = None,
    namespace: Optional[str] = None,
    key_prefix: Optional[str] = None,
    cache_exceptions: bool = False,
    domain: Optional[str] = None,
    sensitivity_level: str = DataSensitivity.LOW
):
    """
    Enhanced decorator for caching function results.

    This decorator extends the base cached decorator with medical-specific
    features, including domain-specific TTL strategies and sensitivity handling.

    Args:
        ttl: Time-to-live in seconds
        namespace: Optional namespace
        key_prefix: Optional key prefix
        cache_exceptions: Whether to cache exceptions
        domain: Medical domain for domain-specific TTL
        sensitivity_level: Data sensitivity level

    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache manager
            cache_mgr = get_enhanced_cache_manager(domain=domain, sensitivity_level=sensitivity_level)

            # Generate cache key
            prefix = key_prefix or f"{func.__module__}.{func.__name__}"
            key = get_cache_key(prefix, *args, **kwargs)

            # Get data type from function return annotation if available
            data_type = None
            if hasattr(func, "__annotations__") and "return" in func.__annotations__:
                data_type = str(func.__annotations__["return"])

            # Try to get from cache
            cached_result = await cache_mgr.get(
                key, namespace, data_type, domain, sensitivity_level
            )

            if cached_result is not None:
                # Handle cached exceptions
                if isinstance(cached_result, dict) and "__exception__" in cached_result:
                    if cache_exceptions:
                        exception_class = cached_result["__exception__"]["class"]
                        exception_args = cached_result["__exception__"]["args"]
                        # Recreate the exception
                        exception = globals().get(exception_class)
                        if exception and issubclass(exception, Exception):
                            raise exception(*exception_args)
                    # If we can't recreate the exception, just return None
                    return None

                return cached_result

            # Call the function
            try:
                result = await func(*args, **kwargs)

                # Cache the result
                effective_ttl = ttl
                if effective_ttl is None and domain:
                    effective_ttl = cache_mgr.get_domain_ttl(domain)

                await cache_mgr.set(
                    key, result, effective_ttl, namespace, data_type, domain, sensitivity_level
                )

                return result
            except Exception as e:
                # Cache the exception if requested
                if cache_exceptions:
                    exception_data = {
                        "__exception__": {
                            "class": e.__class__.__name__,
                            "args": e.args
                        }
                    }
                    await cache_mgr.set(
                        key, exception_data, ttl, namespace, data_type, domain, sensitivity_level
                    )
                raise

        return wrapper

    return decorator

# Create a global instance of the enhanced cache manager - AFTER the class is defined
enhanced_cache_manager = get_enhanced_cache_manager()

