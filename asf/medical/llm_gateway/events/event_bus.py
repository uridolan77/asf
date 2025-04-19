"""
Event bus implementation for LLM Gateway.

This module provides an event bus for publishing and subscribing to events
in the LLM Gateway. The event bus is the central component of the event-driven
architecture, allowing decoupled communication between components.
"""

import asyncio
import logging
import threading
import weakref
from typing import Dict, List, Type, Callable, Awaitable, TypeVar, Generic, Set, Any, Optional

from asf.medical.llm_gateway.events.events import Event

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Event)


class EventBus:
    """
    Event bus for publishing and subscribing to events.
    
    This class provides a central hub for event-driven communication
    between components in the LLM Gateway.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        # Mapping from event type to list of subscribers
        self._subscribers: Dict[Type[Event], List[Callable[[Event], Awaitable[None]]]] = {}
        # Lock for synchronizing access to subscribers dictionary
        self._lock = asyncio.Lock()
        # Thread lock for synchronizing access to subscribers dictionary from non-async contexts
        self._thread_lock = threading.RLock()
        # Event loop accessor
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        
    async def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: The event to publish
        """
        event_type = type(event)
        subscribers = []
        
        # Get subscribers for this event type
        async with self._lock:
            for cls in event_type.__mro__:
                if cls in self._subscribers:
                    subscribers.extend(self._subscribers[cls])
        
        if not subscribers:
            logger.debug(f"No subscribers for event type {event_type.__name__}")
            return
        
        # Call all subscribers
        logger.debug(f"Publishing {event_type.__name__} to {len(subscribers)} subscribers")
        
        # Create tasks for all subscribers
        tasks = [subscriber(event) for subscriber in subscribers]
        
        # Wait for all subscribers to process the event
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log errors from subscribers
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error in subscriber {subscribers[i].__qualname__}: {str(result)}", exc_info=result)
    
    def sync_publish(self, event: Event) -> None:
        """
        Synchronously publish an event to all subscribers.
        
        This method is useful for publishing events from synchronous code paths.
        It will use the current event loop if available, or create a new one if needed.
        
        Args:
            event: The event to publish
        """
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create a new event loop if none exists in this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # If the loop is already running, we need to use run_coroutine_threadsafe
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self.publish(event), loop)
            try:
                # Wait for the result with a timeout
                future.result(timeout=5.0)
            except Exception as e:
                logger.error(f"Error in sync_publish: {str(e)}", exc_info=True)
        else:
            # Otherwise, we can just run the coroutine directly
            try:
                loop.run_until_complete(self.publish(event))
            except Exception as e:
                logger.error(f"Error in sync_publish: {str(e)}", exc_info=True)
    
    async def subscribe(self, event_type: Type[T], handler: Callable[[T], Awaitable[None]]) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: The type of events to subscribe to
            handler: The handler function to call when an event is published
        """
        async with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)
                logger.debug(f"Added subscriber {handler.__qualname__} for {event_type.__name__}")
            else:
                logger.warning(f"Handler {handler.__qualname__} already subscribed to {event_type.__name__}")
    
    async def unsubscribe(self, event_type: Type[T], handler: Callable[[T], Awaitable[None]]) -> None:
        """
        Unsubscribe from events of a specific type.
        
        Args:
            event_type: The type of events to unsubscribe from
            handler: The handler function to remove
        """
        async with self._lock:
            if event_type in self._subscribers and handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
                logger.debug(f"Removed subscriber {handler.__qualname__} for {event_type.__name__}")
                
                # Clean up empty subscriber lists
                if not self._subscribers[event_type]:
                    del self._subscribers[event_type]
            else:
                logger.warning(f"Handler {handler.__qualname__} not subscribed to {event_type.__name__}")
    
    def sync_subscribe(self, event_type: Type[T], handler: Callable[[T], Awaitable[None]]) -> None:
        """
        Synchronously subscribe to events of a specific type.
        
        This method is useful for subscribing from synchronous code paths.
        
        Args:
            event_type: The type of events to subscribe to
            handler: The handler function to call when an event is published
        """
        with self._thread_lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)
                logger.debug(f"Added subscriber {handler.__qualname__} for {event_type.__name__}")
            else:
                logger.warning(f"Handler {handler.__qualname__} already subscribed to {event_type.__name__}")
    
    def sync_unsubscribe(self, event_type: Type[T], handler: Callable[[T], Awaitable[None]]) -> None:
        """
        Synchronously unsubscribe from events of a specific type.
        
        This method is useful for unsubscribing from synchronous code paths.
        
        Args:
            event_type: The type of events to unsubscribe from
            handler: The handler function to remove
        """
        with self._thread_lock:
            if event_type in self._subscribers and handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
                logger.debug(f"Removed subscriber {handler.__qualname__} for {event_type.__name__}")
                
                # Clean up empty subscriber lists
                if not self._subscribers[event_type]:
                    del self._subscribers[event_type]
            else:
                logger.warning(f"Handler {handler.__qualname__} not subscribed to {event_type.__name__}")
    
    def get_subscriber_count(self, event_type: Type[Event] = None) -> int:
        """
        Get the number of subscribers for an event type or all event types.
        
        Args:
            event_type: Optional event type to count subscribers for
            
        Returns:
            Number of subscribers
        """
        if event_type:
            return len(self._subscribers.get(event_type, []))
        else:
            return sum(len(subscribers) for subscribers in self._subscribers.values())
    
    def get_subscribed_event_types(self) -> Set[Type[Event]]:
        """
        Get the set of event types that have subscribers.
        
        Returns:
            Set of event types
        """
        return set(self._subscribers.keys())


# Global event bus singleton
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """
    Get the global event bus instance.
    
    Returns:
        The global event bus instance
    """
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def set_event_bus(event_bus: EventBus) -> None:
    """
    Set the global event bus instance.
    
    Args:
        event_bus: The event bus instance to set as global
    """
    global _event_bus
    _event_bus = event_bus