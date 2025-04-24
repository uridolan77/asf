"""
Resource Management Layer Initialization

This module handles initialization and shutdown of the resource management layer
and provides convenient access to the different resource pools.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Set, Callable, Awaitable, List, TypeVar

# Import resource management components
from .resource_manager import resource_manager, initialize_resource_manager, shutdown_resource_manager
from .models import ResourceType, ResourcePoolConfig, ResourceLimits

# Import transport pools
from ..transport.http_pool import http_transport_manager, initialize_http_transport, shutdown_http_transport
from ..transport.grpc_pool import grpc_transport_manager, initialize_grpc_transport, shutdown_grpc_transport

# Import MCP session pool
from ..mcp.resource_managed_session_pool import create_session_pool

logger = logging.getLogger(__name__)


async def initialize_resource_layer():
    """
    Initialize the resource management layer.
    
    This initializes the core resource manager and all resource pools.
    """
    logger.info("Initializing resource management layer")
    
    # Initialize core resource manager
    await initialize_resource_manager()
    
    # Initialize specific transport pools
    await initialize_http_transport()
    await initialize_grpc_transport()
    
    logger.info("Resource management layer initialized")


async def shutdown_resource_layer():
    """
    Shutdown the resource management layer.
    
    This closes all resource pools and shuts down the resource manager.
    """
    logger.info("Shutting down resource management layer")
    
    # Shutdown specific transport pools first
    await shutdown_http_transport()
    await shutdown_grpc_transport()
    
    # Shutdown core resource manager last
    await shutdown_resource_manager()
    
    logger.info("Resource management layer shut down")


def get_resource_stats() -> Dict[str, Any]:
    """
    Get statistics about all resource pools.
    
    Returns:
        Dictionary with statistics for each resource pool
    """
    # This is an async function, but we provide a sync wrapper for convenience
    async def _get_stats():
        stats = await resource_manager.get_stats()
        return {
            pool_id: pool_stats.dict() 
            for pool_id, pool_stats in stats.items()
        }
    
    # Run in an event loop
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Create a new task
        future = asyncio.run_coroutine_threadsafe(_get_stats(), loop)
        return future.result()
    else:
        # Run the coroutine directly
        return loop.run_until_complete(_get_stats())


class ResourceLayerShutdownHandler:
    """
    Resource Management Layer shutdown handler.
    
    This class can be used to ensure proper shutdown of the resource layer
    when the application exits, including when terminated by signals.
    """
    
    def __init__(self):
        """Initialize the shutdown handler."""
        self._shutdown_complete = False
        self._shutdown_in_progress = False
        self._original_handlers = {}
    
    async def shutdown(self):
        """Perform shutdown of the resource layer."""
        if self._shutdown_complete or self._shutdown_in_progress:
            # Avoid multiple concurrent shutdown attempts
            return
            
        try:
            self._shutdown_in_progress = True
            logger.info("Shutting down resource layer due to application exit")
            await shutdown_resource_layer()
            self._shutdown_complete = True
        except Exception as e:
            logger.error(f"Error during resource layer shutdown: {e}")
        finally:
            self._shutdown_in_progress = False
            
            # Restore original signal handlers to prevent repeated signals
            self._restore_signal_handlers()
    
    def _restore_signal_handlers(self):
        """Restore original signal handlers after shutdown."""
        try:
            import signal
            for sig, handler in self._original_handlers.items():
                signal.signal(sig, handler)
        except (ImportError, NotImplementedError):
            pass
    
    def register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        try:
            import signal
            loop = asyncio.get_event_loop()
            
            # Define signal handler
            def signal_handler(sig, frame):
                logger.info(f"Received signal {sig}, shutting down resource layer")
                if not self._shutdown_in_progress and not self._shutdown_complete:
                    # Only create the shutdown task if not already shutting down
                    loop.create_task(self.shutdown())
            
            # Store original handlers
            self._original_handlers = {
                signal.SIGTERM: signal.getsignal(signal.SIGTERM),
                signal.SIGINT: signal.getsignal(signal.SIGINT)
            }
            
            # Register for common termination signals
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            
            logger.info("Registered signal handlers for resource layer shutdown")
            
        except (ImportError, NotImplementedError):
            # Signal module might not be available on all platforms
            logger.warning("Could not register signal handlers for resource layer shutdown")


# Create a global shutdown handler
shutdown_handler = ResourceLayerShutdownHandler()


# Convenience function for application startup
async def setup_resource_layer(register_signals: bool = True):
    """
    Set up the resource management layer.
    
    Args:
        register_signals: Whether to register signal handlers for graceful shutdown
    """
    await initialize_resource_layer()
    
    if register_signals:
        shutdown_handler.register_signal_handlers()
    
    logger.info("Resource management layer setup complete")