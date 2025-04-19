from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary
)
import time
import functools
import logging
import asyncio

logger = logging.getLogger(__name__)

# Define metrics
LLM_REQUESTS = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['provider', 'model', 'status']
)

LLM_LATENCY = Histogram(
    'llm_request_latency_seconds',
    'LLM request latency',
    ['provider', 'model'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

AGENT_EXECUTIONS = Counter(
    'agent_executions_total',
    'Agent executions',
    ['agent_name', 'status']
)

AGENT_LATENCY = Histogram(
    'agent_execution_latency_seconds',
    'Agent execution latency',
    ['agent_name'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

TOKEN_USAGE = Counter(
    'token_usage_total',
    'Token usage',
    ['provider', 'model', 'type']
)

CACHE_OPERATIONS = Counter(
    'cache_operations_total',
    'Cache operations',
    ['operation', 'result']
)

ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Number of active requests',
    ['endpoint']
)

REQUEST_SIZE = Summary(
    'request_size_bytes',
    'Request size in bytes',
    ['endpoint']
)

RESPONSE_SIZE = Summary(
    'response_size_bytes',
    'Response size in bytes',
    ['endpoint']
)


def track_latency(metric, labels):
    """Decorator to track the latency of a function.

    Args:
        metric: The Histogram metric to update
        labels: The labels for the metric
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric.labels(**labels).observe(duration)
                logger.debug(f"Function {func.__name__} took {duration:.2f} seconds")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric.labels(**labels).observe(duration)
                logger.debug(f"Function {func.__name__} took {duration:.2f} seconds")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def track_llm_request(provider, model):
    """Decorator to track an LLM request.

    Args:
        provider: The LLM provider
        model: The LLM model
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                LLM_REQUESTS.labels(provider=provider, model=model, status="success").inc()

                # Track token usage
                if hasattr(result, 'usage'):
                    for token_type, count in result.usage.items():
                        TOKEN_USAGE.labels(
                            provider=provider,
                            model=model,
                            type=token_type
                        ).inc(count)

                return result
            except Exception as e:
                LLM_REQUESTS.labels(provider=provider, model=model, status="error").inc()
                raise
            finally:
                duration = time.time() - start_time
                LLM_LATENCY.labels(provider=provider, model=model).observe(duration)

        return wrapper

    return decorator


def track_agent_execution(agent_name):
    """Decorator to track an agent execution.

    Args:
        agent_name: The name of the agent
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                AGENT_EXECUTIONS.labels(agent_name=agent_name, status="success").inc()
                return result
            except Exception as e:
                AGENT_EXECUTIONS.labels(agent_name=agent_name, status="error").inc()
                raise
            finally:
                duration = time.time() - start_time
                AGENT_LATENCY.labels(agent_name=agent_name).observe(duration)

        return wrapper

    return decorator
