#!/usr/bin/env python3
"""
ClinicalTrials.gov API Client

A comprehensive client for interacting with the ClinicalTrials.gov Data API.
This client supports all major endpoints and includes features like pagination,
caching, error handling, and data transformation.

Official API documentation: https://clinicaltrials.gov/data-api/api-docs
"""

import os
import json
import time
import hashlib
import requests
import asyncio
import aiohttp
import random
from typing import Dict, List, Optional, Union, Any, Iterator, Tuple, Callable, Type
from functools import lru_cache, wraps
from datetime import datetime, timedelta
import logging
import sqlite3
from pathlib import Path
import pandas as pd
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("clinicaltrials")

# Constants
API_BASE_URL = "https://clinicaltrials.gov/api"
DEFAULT_CACHE_TTL = 86400  # 24 hours in seconds
DEFAULT_CACHE_SIZE = 1000  # Number of items to cache in memory
DEFAULT_REQUESTS_PER_SECOND = 5.0
DEFAULT_BURST_SIZE = 10
DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 0.5


class ErrorCatalog:
    """
    Centralized error catalog for ClinicalTrials.gov client with error codes and messages.
    
    Follows the pattern from the enhancement plan for standardized error handling.
    """
    
    # API errors
    API_CONNECTION_ERROR = ("CT001", "Failed to connect to ClinicalTrials.gov API: {reason}")
    API_TIMEOUT_ERROR = ("CT002", "ClinicalTrials.gov API request timed out after {timeout}s")
    API_RATE_LIMIT = ("CT003", "Rate limit exceeded. Try again in {retry_after}s")
    API_SERVICE_UNAVAILABLE = ("CT004", "ClinicalTrials.gov API service temporarily unavailable")
    
    # Data errors
    DATA_FORMAT_ERROR = ("CT101", "Invalid data format in API response: {reason}")
    DATA_PARSE_ERROR = ("CT102", "Failed to parse API response: {reason}")
    DATA_VALIDATION_ERROR = ("CT103", "Data validation failed: {reason}")
    
    # Request errors
    INVALID_QUERY = ("CT201", "Invalid query parameters: {reason}")
    INVALID_STUDY_ID = ("CT202", "Invalid study ID format: {id}")
    MISSING_REQUIRED = ("CT203", "Missing required parameter: {field}")
    
    # System errors
    CACHE_ERROR = ("CT301", "Cache operation failed: {reason}")
    CONFIG_ERROR = ("CT302", "Configuration error: {reason}")
    
    @classmethod
    def get_message(cls, error_code: str, **kwargs) -> str:
        """Get formatted error message for code with parameters."""
        for attr_name in dir(cls):
            if attr_name.isupper():
                code, message_template = getattr(cls, attr_name)
                if code == error_code:
                    return message_template.format(**kwargs)
        return f"Unknown error code: {error_code}"


class ClinicalTrialsClientError(Exception):
    """Base exception for ClinicalTrials client errors."""
    
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            code: Error code from ErrorCatalog
            message: Error message
            details: Additional error details
        """
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return a string representation of the error."""
        if self.details.get("status_code"):
            return f"{self.code}: {self.message} (Status code: {self.details['status_code']})"
        return f"{self.code}: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details
        }


class APIError(ClinicalTrialsClientError):
    """Error raised for API-related issues."""
    pass


class DataError(ClinicalTrialsClientError):
    """Error raised for data parsing and validation issues."""
    pass


class RequestError(ClinicalTrialsClientError):
    """Error raised for invalid request parameters."""
    pass


class SystemError(ClinicalTrialsClientError):
    """Error raised for system-related issues."""
    pass


def with_error_handling(
    fallback_func: Optional[Callable] = None,
    retry_count: int = DEFAULT_MAX_RETRIES,
    error_mapping: Optional[Dict[Type[Exception], Callable[[Exception], ClinicalTrialsClientError]]] = None
):
    """
    Decorator for standardized error handling with fallback and retry.
    
    Args:
        fallback_func: Function to call if all retries fail
        retry_count: Number of times to retry on failure
        error_mapping: Mapping from exception types to error conversion functions
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            error_mapping_local = error_mapping or {}
            last_error = None
            client_self = args[0] if args else None
            
            # Try with retries
            for attempt in range(retry_count + 1):
                try:
                    return await func(*args, **kwargs)
                except aiohttp.ClientError as e:
                    last_error = e
                    logger.warning(
                        f"API error in {func.__name__} (attempt {attempt+1}/{retry_count+1}): {str(e)}"
                    )
                    
                    # Add jitter to backoff
                    backoff_time = (DEFAULT_BACKOFF_FACTOR * (2 ** attempt)) * (0.5 + random.random())
                    
                    # Log the retry
                    if attempt < retry_count:
                        logger.info(f"Retrying in {backoff_time:.2f} seconds...")
                        await asyncio.sleep(backoff_time)
                        continue
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Error in {func.__name__} (attempt {attempt+1}/{retry_count+1}): {str(e)}"
                    )
                    
                    # If not an API error, only retry specific exceptions
                    if not isinstance(e, (json.JSONDecodeError, asyncio.TimeoutError)):
                        break
                        
                    # Add jitter to backoff
                    backoff_time = (DEFAULT_BACKOFF_FACTOR * (2 ** attempt)) * (0.5 + random.random())
                    
                    # Log the retry
                    if attempt < retry_count:
                        logger.info(f"Retrying in {backoff_time:.2f} seconds...")
                        await asyncio.sleep(backoff_time)
                        continue
            
            # On last attempt, use fallback if available
            if fallback_func is not None:
                try:
                    logger.info(f"Using fallback for {func.__name__}")
                    if client_self:
                        return await fallback_func(client_self, *args[1:], **kwargs)
                    else:
                        return await fallback_func(*args, **kwargs)
                except Exception as fallback_e:
                    logger.error(f"Fallback also failed: {str(fallback_e)}")
                    last_error = fallback_e
            
            # Convert error to appropriate type based on mapping
            for exc_type, converter in error_mapping_local.items():
                if isinstance(last_error, exc_type):
                    raise converter(last_error)
            
            # Default error conversion based on error type
            if isinstance(last_error, aiohttp.ClientConnectorError):
                raise APIError(*ErrorCatalog.API_CONNECTION_ERROR, 
                            details={"reason": str(last_error)})
            elif isinstance(last_error, asyncio.TimeoutError):
                raise APIError(*ErrorCatalog.API_TIMEOUT_ERROR, 
                            details={"timeout": DEFAULT_TIMEOUT})
            elif isinstance(last_error, aiohttp.ClientResponseError):
                if last_error.status == 429:
                    raise APIError(*ErrorCatalog.API_RATE_LIMIT, 
                                details={"retry_after": 60, "status_code": last_error.status})
                elif last_error.status >= 500:
                    raise APIError(*ErrorCatalog.API_SERVICE_UNAVAILABLE, 
                                details={"status_code": last_error.status})
                else:
                    raise APIError(
                        "CT005", f"API request failed with status {last_error.status}", 
                        details={"status_code": last_error.status, "reason": str(last_error)}
                    )
            elif isinstance(last_error, json.JSONDecodeError):
                raise DataError(*ErrorCatalog.DATA_PARSE_ERROR, 
                            details={"reason": str(last_error)})
            else:
                # Re-raise original error if no mapping found
                raise last_error
                
        return wrapper
    return decorator


class AsyncRateLimiter:
    """
    Asynchronous rate limiter using token bucket algorithm.
    Limits the number of requests that can be made per second.
    """
    
    def __init__(self, requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND, 
                 burst_size: int = DEFAULT_BURST_SIZE):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_second: Maximum number of requests per second
            burst_size: Maximum burst size (number of tokens that can be accumulated)
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_time = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """
        Acquire a token from the bucket. Blocks if no tokens are available.
        """
        async with self.lock:
            # Refill tokens based on time elapsed
            now = time.time()
            time_elapsed = now - self.last_time
            new_tokens = time_elapsed * self.requests_per_second
            self.tokens = min(self.burst_size, self.tokens + new_tokens)
            self.last_time = now
            
            # If we don't have a full token, we need to wait
            if self.tokens < 1:
                # Calculate wait time to get a token
                wait_time = (1 - self.tokens) / self.requests_per_second
                await asyncio.sleep(wait_time)
                self.tokens = 0
                self.last_time = time.time()
            else:
                # Consume a token
                self.tokens -= 1
    
    def set_rate_limit(self, requests_per_second: float, burst_size: Optional[int] = None) -> None:
        """
        Update the rate limit parameters.
        
        Args:
            requests_per_second: New maximum requests per second
            burst_size: New maximum burst size (if None, keeps current value)
        """
        self.requests_per_second = requests_per_second
        if burst_size is not None:
            self.burst_size = burst_size


class CtCache:
    """
    Cache for ClinicalTrials.gov API data to improve performance.
    
    This implementation aligns with the ASF unified caching architecture pattern,
    allowing for future integration with the broader caching strategy.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, ttl: int = DEFAULT_CACHE_TTL, 
                 memory_size: int = DEFAULT_CACHE_SIZE):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store persistent cache. If None, only memory cache is used.
            ttl: Time-to-live for cached items in seconds. Default is 24 hours.
            memory_size: Number of items to cache in memory.
        """
        self.ttl = ttl
        self.memory_size = memory_size
        self.memory_cache = {}
        self.disk_cache_enabled = cache_dir is not None
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "expirations": 0
        }
        
        if self.disk_cache_enabled:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize SQLite cache if disk cache is enabled
            self.db_path = self.cache_dir / "clinicaltrials_cache.db"
            self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database for persistent caching."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create cache table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_cache (
            key TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            namespace TEXT DEFAULT 'default',
            data_type TEXT DEFAULT 'general'
        )
        ''')
        
        # Create index on timestamp for cleanup
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON api_cache (created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_namespace ON api_cache (namespace)')
        
        conn.commit()
        conn.close()
    
    def _generate_key(self, prefix: str, *args) -> str:
        """Generate a unique cache key based on the function arguments."""
        key_str = prefix + ":" + json.dumps(args, sort_keys=True)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    def get(self, prefix: str, *args, namespace: str = "default") -> Optional[Dict]:
        """
        Get an item from the cache.
        
        Args:
            prefix: A string prefix to categorize the cache entry
            *args: Arguments to generate the unique key
            namespace: Optional namespace for the cache key
            
        Returns:
            The cached data or None if not found or expired
        """
        key = self._generate_key(prefix, *args)
        ns_key = f"{namespace}:{key}" if namespace != "default" else key
        
        # Check memory cache first
        if ns_key in self.memory_cache:
            entry = self.memory_cache[ns_key]
            if time.time() - entry["timestamp"] < self.ttl:
                # Record cache hit
                self.stats["hits"] += 1
                return entry["data"]
            else:
                # Remove expired item from memory cache
                del self.memory_cache[ns_key]
                self.stats["expirations"] += 1
        
        # If not in memory and disk cache is enabled, check disk
        if self.disk_cache_enabled:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get entry and check if it's expired
            cursor.execute(
                "SELECT data, created_at FROM api_cache WHERE key = ?", 
                (ns_key,)
            )
            result = cursor.fetchone()
            
            if result:
                data_json, created_at = result
                created_timestamp = datetime.fromisoformat(created_at).timestamp()
                
                if time.time() - created_timestamp < self.ttl:
                    data = json.loads(data_json)
                    
                    # Add to memory cache for faster access next time
                    self._add_to_memory_cache(ns_key, data)
                    
                    # Record cache hit
                    self.stats["hits"] += 1
                    
                    conn.close()
                    return data
                else:
                    # Remove expired entry
                    cursor.execute("DELETE FROM api_cache WHERE key = ?", (ns_key,))
                    conn.commit()
                    self.stats["expirations"] += 1
            
            conn.close()
        
        # Record cache miss
        self.stats["misses"] += 1
        return None
    
    def _add_to_memory_cache(self, key: str, data: Dict) -> None:
        """Add an item to the memory cache, managing cache size."""
        if len(self.memory_cache) >= self.memory_size:
            # Simple strategy: remove oldest item
            oldest_key = min(self.memory_cache.keys(), 
                            key=lambda k: self.memory_cache[k]["timestamp"])
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = {
            "data": data,
            "timestamp": time.time()
        }
    
    def set(self, prefix: str, data: Dict, *args, 
            namespace: str = "default", 
            data_type: str = "general", 
            ttl: Optional[int] = None) -> None:
        """
        Store an item in the cache.
        
        Args:
            prefix: A string prefix to categorize the cache entry
            data: The data to cache
            *args: Arguments to generate the unique key
            namespace: Optional namespace for the cache key
            data_type: Type of data for TTL selection
            ttl: Optional override for the default TTL
        """
        key = self._generate_key(prefix, *args)
        ns_key = f"{namespace}:{key}" if namespace != "default" else key
        actual_ttl = ttl if ttl is not None else self.ttl
        
        # Record cache set
        self.stats["sets"] += 1
        
        # Add to memory cache
        self._add_to_memory_cache(ns_key, data)
        
        # If disk cache is enabled, store there too
        if self.disk_cache_enabled:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store the data with the current timestamp
            now = datetime.now().isoformat()
            cursor.execute(
                "INSERT OR REPLACE INTO api_cache (key, data, created_at, namespace, data_type) VALUES (?, ?, ?, ?, ?)",
                (ns_key, json.dumps(data), now, namespace, data_type)
            )
            
            conn.commit()
            conn.close()
    
    def delete(self, prefix: str, *args, namespace: str = "default") -> bool:
        """
        Delete an item from the cache.
        
        Args:
            prefix: A string prefix to categorize the cache entry
            *args: Arguments to generate the unique key
            namespace: Optional namespace for the cache key
            
        Returns:
            True if the item was deleted, False otherwise
        """
        key = self._generate_key(prefix, *args)
        ns_key = f"{namespace}:{key}" if namespace != "default" else key
        deleted = False
        
        # Delete from memory cache
        if ns_key in self.memory_cache:
            del self.memory_cache[ns_key]
            deleted = True
        
        # Delete from disk cache if enabled
        if self.disk_cache_enabled:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM api_cache WHERE key = ?", (ns_key,))
            if cursor.rowcount > 0:
                deleted = True
            
            conn.commit()
            conn.close()
        
        if deleted:
            self.stats["deletes"] += 1
        
        return deleted
    
    def delete_pattern(self, pattern: str, namespace: str = "default") -> int:
        """
        Delete all items matching a pattern from the cache.
        
        Args:
            pattern: Pattern to match against cache keys
            namespace: Optional namespace for the cache key
            
        Returns:
            Number of items deleted
        """
        if namespace != "default":
            full_pattern = f"{namespace}:{pattern}"
        else:
            full_pattern = pattern
            
        count = 0
        
        # Delete from memory cache
        memory_keys_to_delete = [k for k in self.memory_cache.keys() if k.startswith(full_pattern)]
        for k in memory_keys_to_delete:
            del self.memory_cache[k]
            count += 1
        
        # Delete from disk cache if enabled
        if self.disk_cache_enabled:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # SQLite doesn't support LIKE with parameters, so we need to sanitize the pattern
            safe_pattern = full_pattern.replace("%", "\\%").replace("_", "\\_") + "%"
            cursor.execute("DELETE FROM api_cache WHERE key LIKE ? ESCAPE '\\'", (safe_pattern,))
            
            disk_count = cursor.rowcount
            count += disk_count
            
            conn.commit()
            conn.close()
        
        self.stats["deletes"] += count
        return count
    
    def clear_expired(self) -> int:
        """
        Clear expired items from the cache.
        
        Returns:
            Number of items removed
        """
        # Clear expired items from memory cache
        now = time.time()
        expired_keys = [k for k, v in self.memory_cache.items() 
                       if now - v["timestamp"] >= self.ttl]
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        memory_cleared = len(expired_keys)
        
        # Clear expired items from disk cache if enabled
        disk_cleared = 0
        if self.disk_cache_enabled:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate the expiration timestamp
            expiration_time = (datetime.now() - timedelta(seconds=self.ttl)).isoformat()
            
            # Delete expired entries
            cursor.execute(
                "DELETE FROM api_cache WHERE created_at < ?", 
                (expiration_time,)
            )
            
            disk_cleared = cursor.rowcount
            conn.commit()
            conn.close()
        
        total_cleared = memory_cleared + disk_cleared
        self.stats["expirations"] += total_cleared
        return total_cleared
    
    def clear_all(self) -> int:
        """
        Clear all items from the cache.
        
        Returns:
            Number of items removed
        """
        memory_cleared = len(self.memory_cache)
        self.memory_cache = {}
        
        disk_cleared = 0
        if self.disk_cache_enabled:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM api_cache")
            disk_cleared = cursor.rowcount
            
            conn.commit()
            conn.close()
        
        total_cleared = memory_cleared + disk_cleared
        self.stats["deletes"] += total_cleared
        return total_cleared
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        # Calculate hit rate
        total_operations = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_operations if total_operations > 0 else 0
        
        # Calculate memory cache size
        memory_size = len(self.memory_cache)
        memory_usage = sys.getsizeof(self.memory_cache)
        
        # Calculate disk cache size if enabled
        disk_size = 0
        if self.disk_cache_enabled:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM api_cache")
            disk_size = cursor.fetchone()[0]
            
            conn.close()
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "sets": self.stats["sets"],
            "deletes": self.stats["deletes"],
            "expirations": self.stats["expirations"],
            "hit_rate": hit_rate,
            "memory_size": memory_size,
            "memory_usage_bytes": memory_usage,
            "disk_size": disk_size,
            "ttl": self.ttl
        }


class ObservableClient:
    """
    Mixin to add observability features to the ClinicalTrials client.
    
    This implementation follows the ML observability pattern from the enhancement plan,
    adding metrics collection and performance tracking capabilities.
    """
    
    def __init__(self):
        """Initialize observability components."""
        self.metrics = {
            "requests": {
                "total": 0,
                "success": 0,
                "error": 0,
                "timeout": 0,
                "retry": 0
            },
            "latency": {
                "search": [],
                "study": [],
                "status": [],
                "other": []
            },
            "cache": {
                # Cache stats will come from the cache instance
            },
            "start_time": time.time()
        }
        self._last_metrics_reset = time.time()
    
    def _record_request(self, endpoint_type: str, success: bool, 
                       latency_ms: float, retried: bool = False, 
                       error_type: Optional[str] = None) -> None:
        """
        Record metrics for a request.
        
        Args:
            endpoint_type: Type of endpoint (search, study, status, other)
            success: Whether the request was successful
            latency_ms: Request latency in milliseconds
            retried: Whether the request was retried
            error_type: Type of error if the request failed
        """
        # Record request count
        self.metrics["requests"]["total"] += 1
        
        if success:
            self.metrics["requests"]["success"] += 1
        else:
            self.metrics["requests"]["error"] += 1
            
            if error_type == "timeout":
                self.metrics["requests"]["timeout"] += 1
        
        if retried:
            self.metrics["requests"]["retry"] += 1
        
        # Record latency
        if endpoint_type in self.metrics["latency"]:
            # Keep last 100 latencies for each endpoint type
            latencies = self.metrics["latency"][endpoint_type]
            if len(latencies) >= 100:
                latencies.pop(0)
            latencies.append(latency_ms)
        else:
            self.metrics["latency"]["other"].append(latency_ms)
    
    def get_metrics(self, include_cache_stats: bool = True, 
                   reset: bool = False) -> Dict[str, Any]:
        """
        Get client metrics.
        
        Args:
            include_cache_stats: Whether to include cache statistics
            reset: Whether to reset metrics after retrieval
            
        Returns:
            Dictionary with metrics
        """
        # Calculate derived metrics
        now = time.time()
        uptime = now - self.metrics["start_time"]
        time_since_reset = now - self._last_metrics_reset
        
        # Calculate request rate
        request_rate = self.metrics["requests"]["total"] / time_since_reset if time_since_reset > 0 else 0
        
        # Calculate success rate
        total_requests = self.metrics["requests"]["total"]
        success_rate = (self.metrics["requests"]["success"] / total_requests) if total_requests > 0 else 0
        
        # Calculate average latencies
        avg_latencies = {}
        for endpoint_type, latencies in self.metrics["latency"].items():
            if latencies:
                avg_latencies[endpoint_type] = sum(latencies) / len(latencies)
            else:
                avg_latencies[endpoint_type] = 0
        
        metrics_copy = {
            "requests": self.metrics["requests"].copy(),
            "request_rate": round(request_rate, 2),
            "success_rate": round(success_rate, 4),
            "avg_latency_ms": avg_latencies,
            "uptime_seconds": round(uptime, 1),
            "time_since_reset_seconds": round(time_since_reset, 1)
        }
        
        # Include cache stats if requested and available
        if include_cache_stats and hasattr(self, "cache"):
            metrics_copy["cache"] = self.cache.get_stats()
        
        # Reset metrics if requested
        if reset:
            self._reset_metrics()
        
        return metrics_copy
    
    def _reset_metrics(self) -> None:
        """Reset client metrics."""
        for k in self.metrics["requests"]:
            self.metrics["requests"][k] = 0
            
        for k in self.metrics["latency"]:
            self.metrics["latency"][k] = []
            
        self._last_metrics_reset = time.time()
        
    async def export_metrics(self, format: str = "json") -> Union[str, Dict]:
        """
        Export metrics in various formats.
        
        Args:
            format: Output format (json, prometheus, text)
            
        Returns:
            Metrics in the requested format
        """
        metrics = self.get_metrics()
        
        if format == "json":
            return metrics
        elif format == "prometheus":
            # Convert to Prometheus format
            lines = []
            
            # Request metrics
            for k, v in metrics["requests"].items():
                lines.append(f"# TYPE clinicaltrials_requests_{k} counter")
                lines.append(f"clinicaltrials_requests_{k} {v}")
            
            # Success rate
            lines.append(f"# TYPE clinicaltrials_success_rate gauge")
            lines.append(f"clinicaltrials_success_rate {metrics['success_rate']}")
            
            # Request rate
            lines.append(f"# TYPE clinicaltrials_request_rate gauge")
            lines.append(f"clinicaltrials_request_rate {metrics['request_rate']}")
            
            # Latency metrics
            for k, v in metrics["avg_latency_ms"].items():
                lines.append(f"# TYPE clinicaltrials_latency_ms gauge")
                lines.append(f'clinicaltrials_latency_ms{{endpoint="{k}"}} {v}')
            
            # Cache metrics if available
            if "cache" in metrics:
                for k, v in metrics["cache"].items():
                    if isinstance(v, (int, float)):
                        lines.append(f"# TYPE clinicaltrials_cache_{k} gauge")
                        lines.append(f"clinicaltrials_cache_{k} {v}")
            
            return "\n".join(lines)
        elif format == "text":
            # Convert to human-readable text
            lines = [
                "Clinical Trials Client Metrics",
                "==============================",
                f"Uptime: {metrics['uptime_seconds']} seconds",
                "",
                "Request Stats:",
                f"  Total: {metrics['requests']['total']}",
                f"  Success: {metrics['requests']['success']}",
                f"  Error: {metrics['requests']['error']}",
                f"  Timeout: {metrics['requests']['timeout']}",
                f"  Retry: {metrics['requests']['retry']}",
                f"  Success Rate: {metrics['success_rate'] * 100:.2f}%",
                f"  Request Rate: {metrics['request_rate']:.2f} req/sec",
                "",
                "Average Latency (ms):"
            ]
            
            for k, v in metrics["avg_latency_ms"].items():
                lines.append(f"  {k}: {v:.2f}")
            
            if "cache" in metrics:
                lines.extend([
                    "",
                    "Cache Stats:",
                    f"  Hits: {metrics['cache'].get('hits', 0)}",
                    f"  Misses: {metrics['cache'].get('misses', 0)}",
                    f"  Hit Rate: {metrics['cache'].get('hit_rate', 0) * 100:.2f}%",
                    f"  Memory Size: {metrics['cache'].get('memory_size', 0)}",
                    f"  Disk Size: {metrics['cache'].get('disk_size', 0)}"
                ])
            
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")


class ClinicalTrialsClient(ObservableClient):
    """
    Client for interacting with the ClinicalTrials.gov Data API.
    
    This enhanced implementation follows the architectural recommendations
    from the Medical Research Synthesizer enhancement plan, providing improved
    caching, error handling, and observability features.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: str = API_BASE_URL,
                 requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
                 cache_dir: Optional[str] = None,
                 cache_ttl: int = DEFAULT_CACHE_TTL,
                 timeout: float = DEFAULT_TIMEOUT):
        """
        Initialize the client.
        
        Args:
            api_key: Optional API key for authentication
            base_url: Base URL for the API
            requests_per_second: Maximum requests per second
            cache_dir: Directory for persistent cache
            cache_ttl: Cache TTL in seconds
            timeout: Request timeout in seconds
        """
        # Initialize observability components
        super().__init__()
        
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        
        # Set up rate limiter
        self.rate_limiter = AsyncRateLimiter(requests_per_second)
        
        # Set up cache
        self.cache = CtCache(cache_dir=cache_dir, ttl=cache_ttl)
        
        # Set up HTTP session
        self._session = None
        
        logger.info(f"ClinicalTrials client initialized with {requests_per_second} req/s rate limit")
        if cache_dir:
            logger.info(f"Cache directory: {cache_dir}, TTL: {cache_ttl}s")
    
    async def __aenter__(self):
        """Set up the HTTP session on entry."""
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the HTTP session on exit."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def _get_session(self):
        """Get the HTTP session, creating it if necessary."""
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self._session
    
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                           method: str = "GET", data: Optional[Dict] = None,
                           endpoint_type: str = "other", skip_cache: bool = False) -> Dict:
        """
        Make a request to the API with rate limiting, caching, and error handling.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            method: HTTP method
            data: Request body for POST requests
            endpoint_type: Type of endpoint for metrics
            skip_cache: Whether to skip the cache
            
        Returns:
            API response as a dictionary
        """
        # Check cache first unless skipping
        cache_key = f"request:{endpoint}"
        if not skip_cache and method == "GET":
            cached_data = self.cache.get(cache_key, params)
            if cached_data:
                return cached_data
        
        # Create full URL
        url = f"{self.base_url}/{endpoint}"
        
        # Set up headers
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Acquire rate limiting token
        await self.rate_limiter.acquire()
        
        # Start timing for metrics
        start_time = time.time()
        success = False
        retried = False
        error_type = None
        
        try:
            # Get session and make request
            session = await self._get_session()
            
            if method == "GET":
                async with session.get(url, params=params, headers=headers) as response:
                    response.raise_for_status()
                    data = await response.json()
            elif method == "POST":
                async with session.post(url, json=data, params=params, headers=headers) as response:
                    response.raise_for_status()
                    data = await response.json()
            else:
                async with session.request(method, url, json=data, params=params, headers=headers) as response:
                    response.raise_for_status()
                    data = await response.json()
            
            # Cache successful GET responses
            if method == "GET":
                self.cache.set(cache_key, data, params)
            
            success = True
            return data
            
        except (aiohttp.ClientResponseError, aiohttp.ClientConnectorError, 
                aiohttp.ClientOSError, asyncio.TimeoutError) as e:
            if isinstance(e, asyncio.TimeoutError):
                error_type = "timeout"
            elif isinstance(e, aiohttp.ClientResponseError):
                if e.status == 429:
                    error_type = "rate_limit"
                else:
                    error_type = "response"
            else:
                error_type = "connection"
            
            raise
        finally:
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            self._record_request(endpoint_type, success, latency_ms, 
                               retried=retried, error_type=error_type)
    
    @with_error_handling(retry_count=DEFAULT_MAX_RETRIES)
    async def search_studies(self, query: Optional[str] = None, 
                           fields: Optional[List[str]] = None,
                           count: int = 20, skip: int = 0,
                           advanced_query: Optional[Dict[str, Any]] = None,
                           skip_cache: bool = False) -> Dict:
        """
        Search for clinical studies based on the provided query.
        
        Args:
            query: Simple query string
            fields: Fields to include in the response
            count: Number of results to return
            skip: Number of results to skip (pagination)
            advanced_query: Advanced query parameters
            skip_cache: Whether to skip the cache
            
        Returns:
            Search results containing studies matching the query
        """
        # Validate parameters
        if query is None and advanced_query is None:
            raise RequestError(*ErrorCatalog.MISSING_REQUIRED, details={"field": "query or advanced_query"})
        
        # Set up parameters
        params = {
            "count": count,
            "skip": skip
        }
        
        if query:
            params["query"] = query
            
        if fields:
            params["fields"] = ",".join(fields)
        
        # Make different requests based on whether it's simple or advanced
        if advanced_query:
            # For advanced queries, we use the POST endpoint
            endpoint = "v2/studies"
            data = {"query": advanced_query}
            return await self._make_request(endpoint, params, method="POST", 
                                          data=data, endpoint_type="search",
                                          skip_cache=skip_cache)
        else:
            # For simple queries, we use the GET endpoint
            endpoint = "v2/studies"
            return await self._make_request(endpoint, params, endpoint_type="search",
                                          skip_cache=skip_cache)
    
    @with_error_handling(retry_count=DEFAULT_MAX_RETRIES)
    async def get_study(self, nct_id: str, skip_cache: bool = False) -> Dict:
        """
        Get details for a specific study by NCT ID.
        
        Args:
            nct_id: NCT identifier for the study
            skip_cache: Whether to skip the cache
            
        Returns:
            Study details
        """
        # Validate NCT ID format
        if not nct_id.startswith("NCT") or not len(nct_id) == 11:
            raise RequestError(*ErrorCatalog.INVALID_STUDY_ID, details={"id": nct_id})
        
        endpoint = f"v2/studies/{nct_id}"
        return await self._make_request(endpoint, endpoint_type="study", 
                                      skip_cache=skip_cache)
    
    @with_error_handling(retry_count=DEFAULT_MAX_RETRIES)
    async def get_study_fields(self, nct_id: str, fields: List[str], 
                             skip_cache: bool = False) -> Dict:
        """
        Get specific fields for a study.
        
        Args:
            nct_id: NCT identifier for the study
            fields: List of fields to retrieve
            skip_cache: Whether to skip the cache
            
        Returns:
            Study fields
        """
        # Validate NCT ID format
        if not nct_id.startswith("NCT") or not len(nct_id) == 11:
            raise RequestError(*ErrorCatalog.INVALID_STUDY_ID, details={"id": nct_id})
        
        # Validate fields
        if not fields:
            raise RequestError(*ErrorCatalog.MISSING_REQUIRED, details={"field": "fields"})
        
        endpoint = f"v2/studies/{nct_id}"
        params = {"fields": ",".join(fields)}
        return await self._make_request(endpoint, params, endpoint_type="study",
                                      skip_cache=skip_cache)
    
    @with_error_handling(retry_count=DEFAULT_MAX_RETRIES)
    async def get_api_info(self, skip_cache: bool = False) -> Dict:
        """
        Get information about the API.
        
        Args:
            skip_cache: Whether to skip the cache
            
        Returns:
            API information
        """
        endpoint = "v2/api-info"
        return await self._make_request(endpoint, endpoint_type="status",
                                      skip_cache=skip_cache)
    
    async def get_studies_batch(self, nct_ids: List[str], 
                              fields: Optional[List[str]] = None,
                              concurrency: int = 5,
                              skip_cache: bool = False) -> Dict[str, Dict]:
        """
        Get details for multiple studies in batch.
        
        Args:
            nct_ids: List of NCT identifiers
            fields: Optional list of fields to retrieve
            concurrency: Maximum number of concurrent requests
            skip_cache: Whether to skip the cache
            
        Returns:
            Dictionary mapping NCT IDs to study details
        """
        # Validate NCT IDs
        invalid_ids = [nct_id for nct_id in nct_ids 
                      if not nct_id.startswith("NCT") or not len(nct_id) == 11]
        if invalid_ids:
            raise RequestError(*ErrorCatalog.INVALID_STUDY_ID, 
                            details={"id": invalid_ids[0], "invalid_count": len(invalid_ids)})
        
        # Set up semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency)
        
        async def get_study_with_semaphore(nct_id: str) -> Tuple[str, Dict]:
            """Get a study with concurrency control."""
            async with semaphore:
                if fields:
                    study = await self.get_study_fields(nct_id, fields, skip_cache=skip_cache)
                else:
                    study = await self.get_study(nct_id, skip_cache=skip_cache)
                return nct_id, study
        
        # Execute requests in parallel
        tasks = [get_study_with_semaphore(nct_id) for nct_id in nct_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle errors
        studies = {}
        errors = []
        
        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
            else:
                nct_id, study = result
                studies[nct_id] = study
        
        if errors and len(errors) == len(nct_ids):
            # If all requests failed, raise the first error
            raise APIError("CT006", f"All batch requests failed", 
                        details={"error_count": len(errors), "first_error": errors[0]})
        
        # Return successful results even if some failed
        return studies
    
    async def search_all_pages(self, query: Optional[str] = None,
                              fields: Optional[List[str]] = None,
                              max_studies: int = 1000,
                              page_size: int = 100,
                              advanced_query: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Search for clinical studies and return all pages of results.
        
        Args:
            query: Simple query string
            fields: Fields to include in the response
            max_studies: Maximum number of studies to return
            page_size: Number of studies per page
            advanced_query: Advanced query parameters
            
        Returns:
            List of all studies matching the query
        """
        all_studies = []
        skip = 0
        
        while skip < max_studies:
            # Get a page of results
            count = min(page_size, max_studies - skip)
            result = await self.search_studies(
                query=query,
                fields=fields,
                count=count,
                skip=skip,
                advanced_query=advanced_query
            )
            
            # Process results
            studies = result.get("studies", [])
            if not studies:
                break
                
            all_studies.extend(studies)
            
            # Check if we've reached the end
            if len(studies) < count:
                break
                
            # Move to next page
            skip += count
        
        return all_studies
    
    async def export_to_dataframe(self, studies: List[Dict]) -> pd.DataFrame:
        """
        Convert study data to a pandas DataFrame for analysis.
        
        Args:
            studies: List of study dictionaries
            
        Returns:
            DataFrame with study data
        """
        # Extract common fields from studies
        data = []
        
        for study in studies:
            # Extract protocol section
            protocol = study.get("protocolSection", {})
            
            # Extract identification
            identification = protocol.get("identificationModule", {})
            
            # Extract study information
            item = {
                "nct_id": identification.get("nctId"),
                "title": identification.get("officialTitle") or identification.get("briefTitle"),
                "status": protocol.get("statusModule", {}).get("overallStatus"),
                "phase": protocol.get("designModule", {}).get("phases", [None])[0],
                "enrollment": protocol.get("designModule", {}).get("enrollmentInfo", {}).get("count"),
                "study_type": protocol.get("designModule", {}).get("studyType"),
                "conditions": [c.get("name") for c in protocol.get("conditionsModule", {}).get("conditions", [])],
                "interventions": [i.get("name") for i in protocol.get("armsInterventionsModule", {}).get("interventions", [])],
                "start_date": protocol.get("statusModule", {}).get("startDateStruct", {}).get("date"),
                "completion_date": protocol.get("statusModule", {}).get("completionDateStruct", {}).get("date"),
                "sponsors": [s.get("name") for s in protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", [])]
            }
            
            data.append(item)
        
        return pd.DataFrame(data)
    
    async def close(self):
        """Close the client and release resources."""
        if self._session:
            await self._session.close()
            self._session = None
            logger.info("ClinicalTrials client session closed")
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_stats()
    
    async def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear the cache.
        
        Args:
            pattern: Optional pattern to match against cache keys
            
        Returns:
            Number of items removed
        """
        if pattern:
            return self.cache.delete_pattern(pattern)
        else:
            return self.cache.clear_all()
    
    def set_rate_limit(self, requests_per_second: float, burst_size: Optional[int] = None) -> None:
        """
        Update the rate limit parameters.
        
        Args:
            requests_per_second: New maximum requests per second
            burst_size: New maximum burst size (if None, keeps current value)
        """
        self.rate_limiter.set_rate_limit(requests_per_second, burst_size)


class StudyAnalyzer:
    """
    Analyzer for clinical trial study data.
    
    This class provides methods for analyzing clinical trial data,
    extracting insights, and creating visualizations of study characteristics.
    """
    
    def __init__(self, client: 'ClinicalTrialsClient'):
        """
        Initialize the study analyzer.
        
        Args:
            client: An initialized ClinicalTrialsClient instance
        """
        self.client = client
        self.logger = logging.getLogger("clinicaltrials.analyzer")
    
    async def get_phase_distribution(self, query: str, max_results: int = 500) -> Dict[str, Any]:
        """
        Analyze the distribution of clinical trial phases for a given query.
        
        Args:
            query: The search query
            max_results: Maximum number of results to analyze
            
        Returns:
            Dictionary with phase distribution analysis
        """
        # Search for studies matching the query
        studies = await self.client.search_all_pages(
            query=query,
            fields=["NCTId", "BriefTitle", "Phase", "OverallStatus", "EnrollmentCount"],
            max_studies=max_results
        )
        
        # Count studies by phase
        phase_counts = {}
        enrollment_by_phase = {}
        status_by_phase = {}
        
        for study in studies:
            # Extract phase
            phase = study.get("phase", "Not Specified")
            if not phase:
                phase = "Not Specified"
                
            # Count by phase
            if phase not in phase_counts:
                phase_counts[phase] = 0
                enrollment_by_phase[phase] = []
                status_by_phase[phase] = {}
                
            phase_counts[phase] += 1
            
            # Track enrollment
            enrollment = study.get("enrollmentCount")
            if enrollment:
                try:
                    enrollment_by_phase[phase].append(int(enrollment))
                except (ValueError, TypeError):
                    pass
            
            # Track status
            status = study.get("overallStatus", "Unknown")
            if status not in status_by_phase[phase]:
                status_by_phase[phase][status] = 0
            status_by_phase[phase][status] += 1
        
        # Calculate average enrollment by phase
        avg_enrollment = {}
        for phase, enrollments in enrollment_by_phase.items():
            if enrollments:
                avg_enrollment[phase] = sum(enrollments) / len(enrollments)
            else:
                avg_enrollment[phase] = 0
        
        # Prepare results
        return {
            "query": query,
            "total_studies": len(studies),
            "phase_distribution": phase_counts,
            "average_enrollment": avg_enrollment,
            "status_distribution": status_by_phase
        }
    
    async def analyze_study_timeline(self, nct_id: str) -> Dict[str, Any]:
        """
        Analyze the timeline of a clinical trial study.
        
        Args:
            nct_id: NCT identifier for the study
            
        Returns:
            Dictionary with timeline analysis
        """
        # Get study details
        study = await self.client.get_study(nct_id)
        
        # Extract protocol section
        protocol = study.get("data", {}).get("study", {}).get("protocolSection", {})
        
        # Extract status module for dates
        status_module = protocol.get("statusModule", {})
        
        # Extract key dates
        start_date_struct = status_module.get("startDateStruct", {})
        primary_completion_date_struct = status_module.get("primaryCompletionDateStruct", {})
        completion_date_struct = status_module.get("completionDateStruct", {})
        
        # Extract study design info
        design_module = protocol.get("designModule", {})
        
        # Create timeline
        timeline = {
            "study_start": {
                "date": start_date_struct.get("date"),
                "type": start_date_struct.get("type")
            },
            "primary_completion": {
                "date": primary_completion_date_struct.get("date"),
                "type": primary_completion_date_struct.get("type")
            },
            "study_completion": {
                "date": completion_date_struct.get("date"),
                "type": completion_date_struct.get("type")
            },
            "current_status": status_module.get("overallStatus"),
            "why_stopped": status_module.get("whyStopped"),
            "study_type": design_module.get("studyType"),
            "phases": design_module.get("phases", []),
            "enrollment": design_module.get("enrollmentInfo", {}).get("count")
        }
        
        return {
            "nct_id": nct_id,
            "title": protocol.get("identificationModule", {}).get("briefTitle"),
            "timeline": timeline
        }
    
    async def get_outcome_measures(self, nct_id: str) -> Dict[str, Any]:
        """
        Get and analyze outcome measures for a study.
        
        Args:
            nct_id: NCT identifier for the study
            
        Returns:
            Dictionary with outcome measures analysis
        """
        # Get study details
        study = await self.client.get_study(nct_id)
        
        # Extract protocol section
        protocol = study.get("data", {}).get("study", {}).get("protocolSection", {})
        
        # Extract outcome measures
        outcome_module = protocol.get("outcomesModule", {})
        primary_outcomes = outcome_module.get("primaryOutcomes", [])
        secondary_outcomes = outcome_module.get("secondaryOutcomes", [])
        other_outcomes = outcome_module.get("otherOutcomes", [])
        
        # Process outcome measures
        def process_outcomes(outcomes, outcome_type):
            return [{
                "type": outcome_type,
                "measure": outcome.get("measure"),
                "time_frame": outcome.get("timeFrame"),
                "description": outcome.get("description")
            } for outcome in outcomes]
        
        all_outcomes = []
        all_outcomes.extend(process_outcomes(primary_outcomes, "Primary"))
        all_outcomes.extend(process_outcomes(secondary_outcomes, "Secondary"))
        all_outcomes.extend(process_outcomes(other_outcomes, "Other"))
        
        # Get outcome counts by type
        outcome_counts = {
            "Primary": len(primary_outcomes),
            "Secondary": len(secondary_outcomes),
            "Other": len(other_outcomes),
            "Total": len(primary_outcomes) + len(secondary_outcomes) + len(other_outcomes)
        }
        
        return {
            "nct_id": nct_id,
            "title": protocol.get("identificationModule", {}).get("briefTitle"),
            "outcome_counts": outcome_counts,
            "outcomes": all_outcomes
        }
    
    async def compare_studies(self, nct_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple studies side by side.
        
        Args:
            nct_ids: List of NCT identifiers for the studies
            
        Returns:
            Dictionary with comparative analysis
        """
        # Get multiple studies
        studies = await self.client.get_studies_batch(
            nct_ids, 
            fields=["briefTitle", "officialTitle", "overallStatus", "phase", 
                   "designModule.studyType", "designModule.enrollmentInfo",
                   "startDateStruct", "completionDateStruct", "conditions",
                   "interventions", "outcomesModule.primaryOutcomes"]
        )
        
        # Prepare comparison data
        comparison = {
            "studies": [],
            "common_features": {
                "phases": set(),
                "statuses": set(),
                "conditions": set(),
                "interventions": set(),
                "outcome_measures": set()
            }
        }
        
        # Process each study
        for nct_id, study_data in studies.items():
            study = study_data.get("data", {}).get("study", {})
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            
            # Extract key information
            study_info = {
                "nct_id": nct_id,
                "title": identification.get("briefTitle"),
                "status": protocol.get("statusModule", {}).get("overallStatus"),
                "phase": protocol.get("designModule", {}).get("phases", [None])[0],
                "enrollment": protocol.get("designModule", {}).get("enrollmentInfo", {}).get("count"),
                "study_type": protocol.get("designModule", {}).get("studyType"),
                "start_date": protocol.get("statusModule", {}).get("startDateStruct", {}).get("date"),
                "completion_date": protocol.get("statusModule", {}).get("completionDateStruct", {}).get("date"),
                "conditions": [c.get("name") for c in protocol.get("conditionsModule", {}).get("conditions", [])],
                "interventions": [i.get("name") for i in protocol.get("armsInterventionsModule", {}).get("interventions", [])],
                "primary_outcomes": [o.get("measure") for o in protocol.get("outcomesModule", {}).get("primaryOutcomes", [])]
            }
            
            # Add to comparison
            comparison["studies"].append(study_info)
            
            # Update common features
            if study_info["phase"]:
                comparison["common_features"]["phases"].add(study_info["phase"])
            if study_info["status"]:
                comparison["common_features"]["statuses"].add(study_info["status"])
            
            for condition in study_info["conditions"]:
                comparison["common_features"]["conditions"].add(condition)
            
            for intervention in study_info["interventions"]:
                comparison["common_features"]["interventions"].add(intervention)
            
            for outcome in study_info["primary_outcomes"]:
                comparison["common_features"]["outcome_measures"].add(outcome)
        
        # Convert sets to lists for JSON serialization
        for key, value in comparison["common_features"].items():
            comparison["common_features"][key] = list(value)
        
        return comparison
    
    async def get_intervention_distribution(self, query: str, max_results: int = 500) -> Dict[str, Any]:
        """
        Analyze the distribution of intervention types for a given query.
        
        Args:
            query: The search query
            max_results: Maximum number of results to analyze
            
        Returns:
            Dictionary with intervention distribution analysis
        """
        # Search for studies matching the query
        studies = await self.client.search_all_pages(
            query=query,
            fields=["NCTId", "BriefTitle", "InterventionType", "InterventionName", "Phase"],
            max_studies=max_results
        )
        
        # Count studies by intervention type
        intervention_counts = {}
        interventions_by_type = {}
        phases_by_intervention_type = {}
        
        for study in studies:
            # Extract interventions
            interventions = study.get("interventions", [])
            
            if not interventions:
                # If no interventions, categorize as "Not Specified"
                if "Not Specified" not in intervention_counts:
                    intervention_counts["Not Specified"] = 0
                    interventions_by_type["Not Specified"] = []
                    phases_by_intervention_type["Not Specified"] = {}
                
                intervention_counts["Not Specified"] += 1
                
                # Track phase
                phase = study.get("phase", "Not Specified")
                if not phase:
                    phase = "Not Specified"
                
                if phase not in phases_by_intervention_type["Not Specified"]:
                    phases_by_intervention_type["Not Specified"][phase] = 0
                phases_by_intervention_type["Not Specified"][phase] += 1
            else:
                # Process each intervention
                for intervention in interventions:
                    int_type = intervention.get("type", "Not Specified")
                    int_name = intervention.get("name", "Not Specified")
                    
                    if not int_type:
                        int_type = "Not Specified"
                    
                    # Count by type
                    if int_type not in intervention_counts:
                        intervention_counts[int_type] = 0
                        interventions_by_type[int_type] = []
                        phases_by_intervention_type[int_type] = {}
                    
                    intervention_counts[int_type] += 1
                    interventions_by_type[int_type].append(int_name)
                    
                    # Track phase
                    phase = study.get("phase", "Not Specified")
                    if not phase:
                        phase = "Not Specified"
                    
                    if phase not in phases_by_intervention_type[int_type]:
                        phases_by_intervention_type[int_type][phase] = 0
                    phases_by_intervention_type[int_type][phase] += 1
        
        # Get top interventions by type
        top_interventions = {}
        for int_type, interventions in interventions_by_type.items():
            # Count frequency of each intervention name
            name_counts = {}
            for name in interventions:
                if name not in name_counts:
                    name_counts[name] = 0
                name_counts[name] += 1
            
            # Sort by frequency and get top 5
            sorted_names = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)
            top_interventions[int_type] = [{"name": name, "count": count} for name, count in sorted_names[:5]]
        
        # Prepare results
        return {
            "query": query,
            "total_studies": len(studies),
            "intervention_distribution": intervention_counts,
            "top_interventions": top_interventions,
            "phase_distribution": phases_by_intervention_type
        }
    
    async def analyze_sponsors(self, query: str, max_results: int = 500) -> Dict[str, Any]:
        """
        Analyze the distribution of sponsors for a given query.
        
        Args:
            query: The search query
            max_results: Maximum number of results to analyze
            
        Returns:
            Dictionary with sponsor analysis
        """
        # Search for studies matching the query
        studies = await self.client.search_all_pages(
            query=query,
            fields=["NCTId", "BriefTitle", "LeadSponsorName", "LeadSponsorClass", 
                   "CollaboratorName", "CollaboratorClass", "Phase", "OverallStatus"],
            max_studies=max_results
        )
        
        # Count studies by sponsor type
        sponsor_type_counts = {}
        sponsors_by_type = {}
        phases_by_sponsor_type = {}
        
        for study in studies:
            # Extract lead sponsor
            lead_sponsor = study.get("leadSponsor", {})
            sponsor_name = lead_sponsor.get("name", "Not Specified")
            sponsor_type = lead_sponsor.get("class", "Not Specified")
            
            if not sponsor_type:
                sponsor_type = "Not Specified"
            
            # Count by type
            if sponsor_type not in sponsor_type_counts:
                sponsor_type_counts[sponsor_type] = 0
                sponsors_by_type[sponsor_type] = []
                phases_by_sponsor_type[sponsor_type] = {}
            
            sponsor_type_counts[sponsor_type] += 1
            sponsors_by_type[sponsor_type].append(sponsor_name)
            
            # Track phase
            phase = study.get("phase", "Not Specified")
            if not phase:
                phase = "Not Specified"
            
            if phase not in phases_by_sponsor_type[sponsor_type]:
                phases_by_sponsor_type[sponsor_type][phase] = 0
            phases_by_sponsor_type[sponsor_type][phase] += 1
        
        # Get top sponsors by type
        top_sponsors = {}
        for sponsor_type, sponsors in sponsors_by_type.items():
            # Count frequency of each sponsor name
            name_counts = {}
            for name in sponsors:
                if name not in name_counts:
                    name_counts[name] = 0
                name_counts[name] += 1
            
            # Sort by frequency and get top 5
            sorted_names = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)
            top_sponsors[sponsor_type] = [{"name": name, "count": count} for name, count in sorted_names[:5]]
        
        # Prepare results
        return {
            "query": query,
            "total_studies": len(studies),
            "sponsor_type_distribution": sponsor_type_counts,
            "top_sponsors": top_sponsors,
            "phase_distribution": phases_by_sponsor_type
        }

    async def get_enrollment_statistics(self, query: str, max_results: int = 500) -> Dict[str, Any]:
        """
        Analyze enrollment statistics for studies matching a query.
        
        Args:
            query: The search query
            max_results: Maximum number of results to analyze
            
        Returns:
            Dictionary with enrollment statistics
        """
        # Search for studies matching the query
        studies = await self.client.search_all_pages(
            query=query,
            fields=["NCTId", "BriefTitle", "EnrollmentCount", "EnrollmentType", 
                   "Phase", "OverallStatus", "StartDate", "CompletionDate"],
            max_studies=max_results
        )
        
        # Process enrollment data
        enrollments = []
        enrollment_by_phase = {}
        enrollment_by_status = {}
        enrollment_by_year = {}
        
        for study in studies:
            # Extract enrollment count
            enrollment = study.get("enrollmentCount")
            if not enrollment:
                continue
                
            try:
                enrollment = int(enrollment)
            except (ValueError, TypeError):
                continue
                
            enrollments.append(enrollment)
            
            # Track by phase
            phase = study.get("phase", "Not Specified")
            if not phase:
                phase = "Not Specified"
                
            if phase not in enrollment_by_phase:
                enrollment_by_phase[phase] = []
            enrollment_by_phase[phase].append(enrollment)
            
            # Track by status
            status = study.get("overallStatus", "Not Specified")
            if not status:
                status = "Not Specified"
                
            if status not in enrollment_by_status:
                enrollment_by_status[status] = []
            enrollment_by_status[status].append(enrollment)
            
            # Track by start year
            start_date = study.get("startDate", "")
            start_year = None
            
            if start_date and len(start_date) >= 4:
                try:
                    start_year = start_date[:4]
                except (IndexError, TypeError):
                    pass
            
            if start_year:
                if start_year not in enrollment_by_year:
                    enrollment_by_year[start_year] = []
                enrollment_by_year[start_year].append(enrollment)
        
        # Calculate statistics
        def calculate_stats(values):
            if not values:
                return {"min": 0, "max": 0, "mean": 0, "median": 0, "total": 0, "count": 0}
                
            values.sort()
            n = len(values)
            
            if n % 2 == 0:
                median = (values[n//2 - 1] + values[n//2]) / 2
            else:
                median = values[n//2]
                
            return {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / n,
                "median": median,
                "total": sum(values),
                "count": n
            }
        
        overall_stats = calculate_stats(enrollments)
        
        # Calculate stats by category
        phase_stats = {phase: calculate_stats(values) for phase, values in enrollment_by_phase.items()}
        status_stats = {status: calculate_stats(values) for status, values in enrollment_by_status.items()}
        year_stats = {year: calculate_stats(values) for year, values in enrollment_by_year.items()}
        
        # Prepare results
        return {
            "query": query,
            "total_studies_with_enrollment": len(enrollments),
            "overall_statistics": overall_stats,
            "by_phase": phase_stats,
            "by_status": status_stats,
            "by_year": year_stats
        }