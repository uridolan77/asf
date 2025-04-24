"""
Events component for the Enhanced LLM Service.

This module provides event functionality for the Enhanced LLM Service,
including event publishing and subscription.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Awaitable

from asf.medical.llm_gateway.events.event_bus import EventBus

logger = logging.getLogger(__name__)

class EventsComponent:
    """
    Events component for the Enhanced LLM Service.
    
    This class provides event functionality for the Enhanced LLM Service,
    including event publishing and subscription.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None, enabled: bool = True):
        """
        Initialize the events component.
        
        Args:
            event_bus: Optional event bus to use
            enabled: Whether events are enabled
        """
        self.event_bus = event_bus
        self.enabled = enabled
        
        # Store event handlers for manual event handling if no event bus is available
        self._event_handlers: Dict[str, List[Callable[[Dict[str, Any]], Awaitable[None]]]] = {}
        
        # Store recent events for inspection
        self._recent_events: List[Dict[str, Any]] = []
        self._max_recent_events = 100
    
    async def publish_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """
        Publish an event.
        
        Args:
            event_type: Type of event
            payload: Event payload
        """
        if not self.enabled:
            return
        
        try:
            # Create event object
            event = {
                "type": event_type,
                "payload": payload
            }
            
            # Store in recent events
            self._add_recent_event(event)
            
            # Log event
            logger.debug(f"Published event {event_type}: {payload}")
            
            # Forward to event bus if available
            if self.event_bus:
                await self.event_bus.publish(event_type, payload)
            else:
                # Manual event handling
                await self._handle_event(event_type, payload)
        except Exception as e:
            logger.error(f"Error publishing event: {str(e)}")
    
    async def subscribe_to_events(self,
                                 event_type: str,
                                 handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """
        Subscribe to events.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Event handler function
        """
        if not self.enabled:
            return
        
        try:
            # Register with event bus if available
            if self.event_bus:
                await self.event_bus.subscribe(event_type, handler)
            
            # Register with manual event handling
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            
            self._event_handlers[event_type].append(handler)
            
            # Log subscription
            logger.debug(f"Subscribed to event type {event_type}")
        except Exception as e:
            logger.error(f"Error subscribing to events: {str(e)}")
    
    async def _handle_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """
        Handle an event manually.
        
        Args:
            event_type: Type of event
            payload: Event payload
        """
        if event_type not in self._event_handlers:
            return
        
        for handler in self._event_handlers[event_type]:
            try:
                await handler(payload)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {str(e)}")
    
    def _add_recent_event(self, event: Dict[str, Any]) -> None:
        """
        Add an event to the recent events list.
        
        Args:
            event: Event to add
        """
        self._recent_events.append(event)
        
        # Trim list if it gets too long
        if len(self._recent_events) > self._max_recent_events:
            self._recent_events = self._recent_events[-self._max_recent_events:]
    
    def get_recent_events(self, event_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent events.
        
        Args:
            event_type: Optional event type to filter by
            limit: Maximum number of events to return
            
        Returns:
            List of recent events
        """
        if not self.enabled:
            return []
        
        # Filter by event type if provided
        if event_type:
            filtered_events = [e for e in self._recent_events if e.get("type") == event_type]
        else:
            filtered_events = self._recent_events
        
        # Return the most recent events up to the limit
        return filtered_events[-limit:]
    
    def clear_recent_events(self) -> None:
        """
        Clear the recent events list.
        """
        if not self.enabled:
            return
        
        self._recent_events = []
