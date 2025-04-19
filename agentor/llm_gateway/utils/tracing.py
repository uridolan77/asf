import logging
import uuid
from typing import Dict, Any, Optional, List, Callable, Awaitable, TypeVar
import functools
import time
import asyncio

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace import SpanKind, Status, StatusCode
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from fastapi import Request, Response

logger = logging.getLogger(__name__)

T = TypeVar('T')


def setup_tracing(service_name: str, endpoint: Optional[str] = None):
    """Set up tracing for the application.
    
    Args:
        service_name: The name of the service
        endpoint: The endpoint to send traces to (optional)
    """
    # Create a tracer provider
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)
    
    # If an endpoint is provided, set up the exporter
    if endpoint:
        # Create an exporter
        exporter = OTLPSpanExporter(endpoint=endpoint)
        
        # Create a processor
        processor = BatchSpanProcessor(exporter)
        
        # Add the processor to the provider
        tracer_provider.add_span_processor(processor)
    
    # Return the tracer
    return trace.get_tracer(service_name)


class TracingMiddleware(BaseHTTPMiddleware):
    """Middleware for tracing HTTP requests."""
    
    def __init__(self, app: ASGIApp, tracer=None):
        """Initialize the middleware.
        
        Args:
            app: The ASGI application
            tracer: The tracer to use (optional)
        """
        super().__init__(app)
        self.tracer = tracer or trace.get_tracer("llm_gateway")
        self.propagator = TraceContextTextMapPropagator()
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process the request.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response
        """
        # Extract the trace context from the request headers
        context = self.propagator.extract(request.headers)
        
        # Start a new span
        with self.tracer.start_as_current_span(
            f"{request.method} {request.url.path}",
            context=context,
            kind=SpanKind.SERVER,
            attributes={
                "http.method": request.method,
                "http.url": str(request.url),
                "http.host": request.url.hostname,
                "http.scheme": request.url.scheme,
                "http.target": request.url.path,
            }
        ) as span:
            # Add a request ID
            request_id = str(uuid.uuid4())
            span.set_attribute("request.id", request_id)
            
            # Process the request
            start_time = time.time()
            
            try:
                response = await call_next(request)
                
                # Add response attributes
                span.set_attribute("http.status_code", response.status_code)
                
                # Set the span status
                if response.status_code >= 400:
                    span.set_status(Status(StatusCode.ERROR))
                else:
                    span.set_status(Status(StatusCode.OK))
                
                return response
            
            except Exception as e:
                # Set the span status
                span.set_status(Status(StatusCode.ERROR, str(e)))
                
                # Record the exception
                span.record_exception(e)
                
                # Re-raise the exception
                raise
            
            finally:
                # Record the latency
                latency = time.time() - start_time
                span.set_attribute("http.latency_ms", latency * 1000)


def trace_function(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """Decorator to trace a function.
    
    Args:
        name: The name of the span (optional)
        attributes: Attributes to add to the span (optional)
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get the tracer
            tracer = trace.get_tracer("llm_gateway")
            
            # Start a span
            with tracer.start_as_current_span(
                name or func.__name__,
                attributes=attributes
            ) as span:
                # Add the function arguments as attributes
                for i, arg in enumerate(args):
                    if isinstance(arg, (str, int, float, bool)):
                        span.set_attribute(f"arg.{i}", arg)
                
                for key, value in kwargs.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f"kwarg.{key}", value)
                
                # Call the function
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Add the result as an attribute
                    if isinstance(result, (str, int, float, bool)):
                        span.set_attribute("result", result)
                    
                    return result
                
                except Exception as e:
                    # Set the span status
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    
                    # Record the exception
                    span.record_exception(e)
                    
                    # Re-raise the exception
                    raise
                
                finally:
                    # Record the latency
                    latency = time.time() - start_time
                    span.set_attribute("latency_ms", latency * 1000)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get the tracer
            tracer = trace.get_tracer("llm_gateway")
            
            # Start a span
            with tracer.start_as_current_span(
                name or func.__name__,
                attributes=attributes
            ) as span:
                # Add the function arguments as attributes
                for i, arg in enumerate(args):
                    if isinstance(arg, (str, int, float, bool)):
                        span.set_attribute(f"arg.{i}", arg)
                
                for key, value in kwargs.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f"kwarg.{key}", value)
                
                # Call the function
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Add the result as an attribute
                    if isinstance(result, (str, int, float, bool)):
                        span.set_attribute("result", result)
                    
                    return result
                
                except Exception as e:
                    # Set the span status
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    
                    # Record the exception
                    span.record_exception(e)
                    
                    # Re-raise the exception
                    raise
                
                finally:
                    # Record the latency
                    latency = time.time() - start_time
                    span.set_attribute("latency_ms", latency * 1000)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
