"""
Event subscriber base class for LLM Gateway.

This module provides the base class for components that want to subscribe
to events from the event bus.
"""

import abc
import asyncio
import logging
from typing import List, Dict, Any, Type, TypeVar, Generic, Optional, Set, Callable, Awaitable

from asf.medical.llm_gateway.events.event_bus import EventBus, get_event_bus
from asf.medical.llm_gateway.events.events import Event

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=Event)


class EventSubscriber(abc.ABC):
    """
    Abstract base class for event subscribers.
    
    Classes that want to subscribe to events should inherit from this class
    and implement the handle_event method for each event type they're interested in.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the event subscriber.
        
        Args:
            event_bus: Optional event bus to use. If not provided, will use the global instance.
        """
        self.event_bus = event_bus or get_event_bus()
        self._subscriptions: Dict[Type[Event], Callable[[Event], Awaitable[None]]] = {}
        self._is_subscribed = False
        
    async def subscribe(self) -> None:
        """
        Subscribe to events.
        
        This method subscribes the handler methods to the appropriate event types
        based on their annotations.
        """
        if self._is_subscribed:
            logger.warning(f"{self.__class__.__name__} is already subscribed to events")
            return
        
        # Find all methods with event handling annotations
        for attribute_name in dir(self):
            if attribute_name.startswith("_"):
                continue
                
            attribute = getattr(self, attribute_name)
            if not callable(attribute):
                continue
                
            # Check if this method is annotated to handle specific event types
            event_types = getattr(attribute, "_event_types", None)
            if event_types:
                for event_type in event_types:
                    method = attribute
                    wrapper = lambda event, method=method: method(event)
                    await self.event_bus.subscribe(event_type, wrapper)
                    self._subscriptions[event_type] = wrapper
        
        self._is_subscribed = True
        logger.debug(f"{self.__class__.__name__} subscribed to {len(self._subscriptions)} event types")
    
    async def unsubscribe(self) -> None:
        """
        Unsubscribe from all events.
        """
        if not self._is_subscribed:
            return
            
        # Unsubscribe from all event types
        for event_type, handler in self._subscriptions.items():
            await self.event_bus.unsubscribe(event_type, handler)
            
        self._subscriptions.clear()
        self._is_subscribed = False
        logger.debug(f"{self.__class__.__name__} unsubscribed from all events")
    
    def sync_subscribe(self) -> None:
        """
        Synchronously subscribe to events.
        
        This is a convenience method for subscribing from synchronous code.
        """
        if self._is_subscribed:
            logger.warning(f"{self.__class__.__name__} is already subscribed to events")
            return
            
        # Find all methods with event handling annotations
        for attribute_name in dir(self):
            if attribute_name.startswith("_"):
                continue
                
            attribute = getattr(self, attribute_name)
            if not callable(attribute):
                continue
                
            # Check if this method is annotated to handle specific event types
            event_types = getattr(attribute, "_event_types", None)
            if event_types:
                for event_type in event_types:
                    method = attribute
                    wrapper = lambda event, method=method: method(event)
                    self.event_bus.sync_subscribe(event_type, wrapper)
                    self._subscriptions[event_type] = wrapper
        
        self._is_subscribed = True
        logger.debug(f"{self.__class__.__name__} subscribed to {len(self._subscriptions)} event types")
    
    def sync_unsubscribe(self) -> None:
        """
        Synchronously unsubscribe from all events.
        
        This is a convenience method for unsubscribing from synchronous code.
        """
        if not self._is_subscribed:
            return
            
        # Unsubscribe from all event types
        for event_type, handler in self._subscriptions.items():
            self.event_bus.sync_unsubscribe(event_type, handler)
            
        self._subscriptions.clear()
        self._is_subscribed = False
        logger.debug(f"{self.__class__.__name__} unsubscribed from all events")
    
    @property
    def subscription_count(self) -> int:
        """
        Get the number of event types this subscriber is subscribed to.
        
        Returns:
            Number of subscribed event types
        """
        return len(self._subscriptions)
    
    @property
    def subscribed_event_types(self) -> Set[Type[Event]]:
        """
        Get the set of event types this subscriber is subscribed to.
        
        Returns:
            Set of subscribed event types
        """
        return set(self._subscriptions.keys())


def handles_event(event_type: Type[T]):
    """
    Decorator for methods that handle specific event types.
    
    Args:
        event_type: The event type to handle
        
    Returns:
        Decorated method
    """
    def decorator(method):
        if not hasattr(method, "_event_types"):
            method._event_types = []
        method._event_types.append(event_type)
        return method
    return decorator