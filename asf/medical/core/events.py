Event system for the Medical Research Synthesizer.
This module provides an event-driven architecture for the application,
allowing components to communicate through events rather than direct dependencies.
import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Optional, Type, TypeVar, Generic, Union
from datetime import datetime
from asf.medical.core.logging_config import get_logger
logger = get_logger(__name__)
T = TypeVar('T')
class Event:
    Base class for all events.
    Events are used to communicate between different parts of the application
    without creating direct dependencies.
    def __init__(self, event_type: str, data: Dict[str, Any], source: str = None):
        """
        Initialize an event.
        Args:
            event_type: Type of the event
            data: Event data
            source: Source of the event (default: None)
        """
        self.event_type = event_type
        self.data = data
        self.source = source
        self.timestamp = datetime.utcnow()
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event to a dictionary.
        Returns:
            Dictionary representation of the event
        """
        return {
            "event_type": self.event_type,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """
        Create an event from a dictionary.
        Args:
            data: Dictionary representation of the event
        Returns:
            Event instance
        """
        event = cls(
            event_type=data["event_type"],
            data=data["data"],
            source=data.get("source"),
        )
        event.timestamp = datetime.fromisoformat(data["timestamp"])
        return event
    def __str__(self) -> str:
        """
        Get a string representation of the event.
        Returns:
            String representation
        """
        return f"Event({self.event_type}, source={self.source}, timestamp={self.timestamp})"
class EventHandler(Generic[T]):
    Handler for events.
    Event handlers are registered with the event bus to handle specific event types.
    def __init__(self, handler_func: Callable[[T], Any], event_type: Type[T] = None):
        """
        Initialize an event handler.
        Args:
            handler_func: Function to handle the event
            event_type: Type of events to handle (default: inferred from type hints)
        """
        self.handler_func = handler_func
        # If event_type is not provided, try to infer it from the handler function
        if event_type is None:
            sig = inspect.signature(handler_func)
            if len(sig.parameters) != 1:
                raise ValueError("Event handler must take exactly one parameter")
            param = list(sig.parameters.values())[0]
            if param.annotation == inspect.Parameter.empty:
                raise ValueError("Event handler parameter must have a type annotation")
            event_type = param.annotation
        self.event_type = event_type
    async def __call__(self, event: T) -> Any:
        """
        Call the handler function with the event.
        Args:
            event: Event to handle
        Returns:
            Result of the handler function
        """
        return await self.handler_func(event)
class EventBus:
    Event bus for the application.
    The event bus allows components to publish events and subscribe to event types.
    def __init__(self):
        Initialize the event bus.
        
        Args:
        
        Subscribe to events of a specific type.
        Args:
            event_type: Type of events to subscribe to
            handler: Function or EventHandler to handle the events
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        # Convert function to EventHandler if needed
        if not isinstance(handler, EventHandler):
            handler = EventHandler(handler, event_type)
        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed handler to event type: {event_type.__name__}")
    def unsubscribe(self, event_type: Type[T], handler: Union[Callable[[T], Any], EventHandler[T]]) -> None:
        """
        Unsubscribe from events of a specific type.
        Args:
            event_type: Type of events to unsubscribe from
            handler: Function or EventHandler to unsubscribe
        """
        if event_type not in self._handlers:
            return
        # Convert function to EventHandler if needed
        if not isinstance(handler, EventHandler):
            handler = EventHandler(handler, event_type)
        self._handlers[event_type] = [h for h in self._handlers[event_type] if h != handler]
        logger.debug(f"Unsubscribed handler from event type: {event_type.__name__}")
    async def publish(self, event: T) -> None:
        """
        Publish an event.
        Args:
            event: Event to publish
        """
        await self._queue.put(event)
        logger.debug(f"Published event: {event}")
    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                event = await self._queue.get()
                # Find handlers for this event type
                handlers = []
                for event_type, type_handlers in self._handlers.items():
                    if isinstance(event, event_type):
                        handlers.extend(type_handlers)
                # Call handlers
                for handler in handlers:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(
                            f"Error in event handler: {str(e)}",
                            extra={"event": str(event), "handler": str(handler)},
                            exc_info=e
                        )
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {str(e)}", exc_info=e)
    def start(self) -> None:
        Start the event bus.
        
        Args:
        
        
        Returns:
            Description of return value
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Stopped event bus")
class EventBroker(ABC):
    Abstract base class for event brokers.
    Event brokers handle communication between different instances of the application,
    allowing events to be published and subscribed to across process boundaries.
    @abstractmethod
    async def connect(self) -> None:
        """
        Connect to the event broker.
        Raises:
            Exception: If connection fails
        """
        pass
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the event broker.
        Raises:
            Exception: If disconnection fails
        """
        pass
    @abstractmethod
    async def publish(self, topic: str, event: Event) -> None:
        """
        Publish an event to a topic.
        Args:
            topic: Topic to publish to
            event: Event to publish
        Raises:
            Exception: If publishing fails
        """
        pass
    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable[[Event], Any]) -> None:
        """
        Subscribe to events on a topic.
        Args:
            topic: Topic to subscribe to
            handler: Function to handle events
        Raises:
            Exception: If subscription fails
        """
        pass
    @abstractmethod
    async def unsubscribe(self, topic: str, handler: Callable[[Event], Any]) -> None:
        """
        Unsubscribe from events on a topic.
        Args:
            topic: Topic to unsubscribe from
            handler: Function to unsubscribe
        Raises:
            Exception: If unsubscription fails
        """
        pass
# Create a global event bus
event_bus = EventBus()
# Start the event bus
event_bus.start()
# Define common event types
class UserEvent(Event):
    Base class for user-related events.
        Initialize a user event.
        Args:
            user_id: ID of the user
            data: Event data
            source: Source of the event (default: None)
        """
        super().__init__(f"user.{self.__class__.__name__.lower()}", data, source)
        self.user_id = user_id
        self.data["user_id"] = user_id
class UserCreatedEvent(UserEvent):
    Event fired when a user is created.
        Initialize a user created event.
        Args:
            user_id: ID of the user
            email: Email of the user
            role: Role of the user
            source: Source of the event (default: None)
        """
        super().__init__(user_id, {"email": email, "role": role}, source)
class UserUpdatedEvent(UserEvent):
    Event fired when a user is updated.
        Initialize a user updated event.
        Args:
            user_id: ID of the user
            updated_fields: Fields that were updated
            source: Source of the event (default: None)
        """
        super().__init__(user_id, {"updated_fields": updated_fields}, source)
class UserDeletedEvent(UserEvent):
    Event fired when a user is deleted.
        Initialize a user deleted event.
        Args:
            user_id: ID of the user
            source: Source of the event (default: None)
        """
        super().__init__(user_id, {}, source)
class SearchEvent(Event):
    Base class for search-related events.
        Initialize a search event.
        Args:
            query: Search query
            data: Event data
            source: Source of the event (default: None)
        """
        super().__init__(f"search.{self.__class__.__name__.lower()}", data, source)
        self.query = query
        self.data["query"] = query
class SearchPerformedEvent(SearchEvent):
    Event fired when a search is performed.
        Initialize a search performed event.
        Args:
            query: Search query
            filters: Search filters
            user_id: ID of the user who performed the search (default: None)
            source: Source of the event (default: None)
        """
        super().__init__(query, {"filters": filters, "user_id": user_id}, source)
class SearchCompletedEvent(SearchEvent):
    Event fired when a search is completed.
        Initialize a search completed event.
        Args:
            query: Search query
            result_count: Number of results
            duration_ms: Duration of the search in milliseconds
            user_id: ID of the user who performed the search (default: None)
            source: Source of the event (default: None)
        """
        super().__init__(
            query,
            {
                "result_count": result_count,
                "duration_ms": duration_ms,
                "user_id": user_id
            },
            source
        )
class AnalysisEvent(Event):
    Base class for analysis-related events.
        Initialize an analysis event.
        Args:
            analysis_id: ID of the analysis
            data: Event data
            source: Source of the event (default: None)
        """
        super().__init__(f"analysis.{self.__class__.__name__.lower()}", data, source)
        self.analysis_id = analysis_id
        self.data["analysis_id"] = analysis_id
class AnalysisStartedEvent(AnalysisEvent):
    Event fired when an analysis is started.
        Initialize an analysis started event.
        Args:
            analysis_id: ID of the analysis
            analysis_type: Type of analysis
            user_id: ID of the user who started the analysis (default: None)
            source: Source of the event (default: None)
        """
        super().__init__(
            analysis_id,
            {
                "analysis_type": analysis_type,
                "user_id": user_id
            },
            source
        )
class AnalysisCompletedEvent(AnalysisEvent):
    Event fired when an analysis is completed.
        Initialize an analysis completed event.
        Args:
            analysis_id: ID of the analysis
            result: Analysis result
            duration_ms: Duration of the analysis in milliseconds
            source: Source of the event (default: None)
        """
        super().__init__(
            analysis_id,
            {
                "result": result,
                "duration_ms": duration_ms
            },
            source
        )
class AnalysisFailedEvent(AnalysisEvent):
    Event fired when an analysis fails.
        Initialize an analysis failed event.
        Args:
            analysis_id: ID of the analysis
            error: Error message
            source: Source of the event (default: None)
        """
        super().__init__(
            analysis_id,
            {
                "error": error
            },
            source
        )
class TaskEvent(Event):
    Base class for task-related events.
        Initialize a task event.
        Args:
            task_id: ID of the task
            data: Event data
            source: Source of the event (default: None)
        """
        super().__init__(f"task.{self.__class__.__name__.lower()}", data, source)
        self.task_id = task_id
        self.data["task_id"] = task_id
class TaskStartedEvent(TaskEvent):
    Event fired when a task is started.
        Initialize a task started event.
        Args:
            task_id: ID of the task
            task_type: Type of task
            params: Task parameters
            source: Source of the event (default: None)
        """
        super().__init__(
            task_id,
            {
                "task_type": task_type,
                "params": params
            },
            source
        )
class TaskProgressEvent(TaskEvent):
    Event fired when a task makes progress.
        Initialize a task progress event.
        Args:
            task_id: ID of the task
            progress: Progress percentage (0-100)
            message: Progress message
            source: Source of the event (default: None)
        """
        super().__init__(
            task_id,
            {
                "progress": progress,
                "message": message
            },
            source
        )
class TaskCompletedEvent(TaskEvent):
    Event fired when a task is completed.
        Initialize a task completed event.
        Args:
            task_id: ID of the task
            result: Task result
            duration_ms: Duration of the task in milliseconds
            source: Source of the event (default: None)
        """
        super().__init__(
            task_id,
            {
                "result": result,
                "duration_ms": duration_ms
            },
            source
        )
class TaskFailedEvent(TaskEvent):
    Event fired when a task fails.
        Initialize a task failed event.
        Args:
            task_id: ID of the task
            error: Error message
            source: Source of the event (default: None)
        """
        super().__init__(
            task_id,
            {
                "error": error
            },
            source
        )