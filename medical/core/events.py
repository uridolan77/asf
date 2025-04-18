"""
Event system for the Medical Research Synthesizer.

This module provides an event-driven architecture for the application,
allowing components to communicate through events rather than direct dependencies.

Classes:
    Event: Base class for all events.
    EventHandler: Handler for events.
    EventBus: Event bus for the application.
    EventBroker: Abstract base class for event brokers.
    UserEvent: Base class for user-related events.
    UserCreatedEvent: Event fired when a user is created.
    UserUpdatedEvent: Event fired when a user is updated.
    UserDeletedEvent: Event fired when a user is deleted.
    SearchEvent: Base class for search-related events.
    SearchPerformedEvent: Event fired when a search is performed.
    SearchCompletedEvent: Event fired when a search is completed.
    AnalysisEvent: Base class for analysis-related events.
    AnalysisStartedEvent: Event fired when an analysis is started.
    AnalysisCompletedEvent: Event fired when an analysis is completed.
    AnalysisFailedEvent: Event fired when an analysis fails.
    TaskEvent: Base class for task-related events.
    TaskStartedEvent: Event fired when a task is started.
    TaskProgressEvent: Event fired when a task makes progress.
    TaskCompletedEvent: Event fired when a task is completed.
    TaskFailedEvent: Event fired when a task fails.
"""

import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Optional, Type, TypeVar, Generic, Union
from datetime import datetime
from .logging_config import get_logger

logger = get_logger(__name__)
T = TypeVar('T')

class Event:
    """
    Base class for all events.

    Events are used to communicate between different parts of the application
    without creating direct dependencies.

    Attributes:
        event_type (str): Type of the event.
        data (Dict[str, Any]): Data associated with the event.
        source (Optional[str]): Source of the event.
        timestamp (datetime): Time when the event was created.
    """

    def __init__(self, event_type: str, data: Dict[str, Any], source: str = None):
        """
        Initialize the Event instance.

        Args:
            event_type (str): Type of the event.
            data (Dict[str, Any]): Data associated with the event.
            source (str, optional): Source of the event. Defaults to None.
        """
        self.event_type = event_type
        self.data = data
        self.source = source
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the event.
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
            data (Dict[str, Any]): Dictionary containing event data.

        Returns:
            Event: Event instance created from the dictionary.
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
            str: String representation of the event.
        """
        return f"Event({self.event_type}, source={self.source}, timestamp={self.timestamp})"

class EventHandler(Generic[T]):
    """
    Handler for events.

    Event handlers are registered with the event bus to handle specific event types.

    Attributes:
        handler_func (Callable[[T], Any]): Function to handle the event.
        event_type (Type[T]): Type of events to handle.
    """

    def __init__(self, handler_func: Callable[[T], Any], event_type: Type[T] = None):
        """
        Initialize an event handler.

        Args:
            handler_func (Callable[[T], Any]): Function to handle the event.
            event_type (Type[T], optional): Type of events to handle. Defaults to None.
        """
        self.handler_func = handler_func
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
            event (T): Event to handle.

        Returns:
            Any: Result of the handler function.
        """
        return await self.handler_func(event)

class EventBus:
    """
    Event bus for the application.

    The event bus allows components to publish events and subscribe to event types.

    Attributes:
        _handlers (Dict[Type[T], List[Union[Callable[[T], Any], EventHandler[T]]]]): Registered event handlers.
        _queue (asyncio.Queue): Queue for events to be processed.
        _task (Optional[asyncio.Task]): Background task for processing events.
        _running (bool): Whether the event bus is running.
    """

    def __init__(self):
        """
        Initialize the event bus.
        """
        self._handlers: Dict[Type[T], List[Union[Callable[[T], Any], EventHandler[T]]]] = {}
        self._queue = asyncio.Queue()
        self._task = None
        self._running = False

    def subscribe(self, event_type: Type[T], handler: Union[Callable[[T], Any], EventHandler[T]]) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type (Type[T]): Type of events to subscribe to.
            handler (Union[Callable[[T], Any], EventHandler[T]]): Function or EventHandler to handle the events.
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        if not isinstance(handler, EventHandler):
            handler = EventHandler(handler, event_type)
        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed handler to event type: {event_type.__name__}")

    def unsubscribe(self, event_type: Type[T], handler: Union[Callable[[T], Any], EventHandler[T]]) -> None:
        """
        Unsubscribe from events of a specific type.

        Args:
            event_type (Type[T]): Type of events to unsubscribe from.
            handler (Union[Callable[[T], Any], EventHandler[T]]): Function or EventHandler to unsubscribe.
        """
        if event_type not in self._handlers:
            return
        if not isinstance(handler, EventHandler):
            handler = EventHandler(handler, event_type)
        self._handlers[event_type] = [h for h in self._handlers[event_type] if h != handler]
        logger.debug(f"Unsubscribed handler from event type: {event_type.__name__}")

    async def publish(self, event: T) -> None:
        """
        Publish an event.

        Args:
            event (T): Event to publish.
        """
        await self._queue.put(event)
        logger.debug(f"Published event: {event}")

    async def _process_events(self) -> None:
        """
        Process events from the queue.
        """
        while self._running:
            try:
                event = await self._queue.get()
                handlers = []
                for event_type, type_handlers in self._handlers.items():
                    if isinstance(event, event_type):
                        handlers.extend(type_handlers)
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
        """
        Start the event bus.
        """
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._process_events())
        logger.info("Started event bus")

    def stop(self) -> None:
        """
        Stop the event bus.
        """
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                asyncio.run(self._task)
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Stopped event bus")

class EventBroker(ABC):
    """
    Abstract base class for event brokers.

    Event brokers handle communication between different instances of the application,
    allowing events to be published and subscribed to across process boundaries.
    """

    @abstractmethod
    async def connect(self) -> None:
        """
        Connect to the event broker.

        Raises:
            Exception: If connection fails.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the event broker.

        Raises:
            Exception: If disconnection fails.
        """
        pass

    @abstractmethod
    async def publish(self, topic: str, event: Event) -> None:
        """
        Publish an event to a topic.

        Args:
            topic (str): Topic to publish to.
            event (Event): Event to publish.

        Raises:
            Exception: If publishing fails.
        """
        pass

    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable[[Event], Any]) -> None:
        """
        Subscribe to events on a topic.

        Args:
            topic (str): Topic to subscribe to.
            handler (Callable[[Event], Any]): Function to handle events.

        Raises:
            Exception: If subscription fails.
        """
        pass

    @abstractmethod
    async def unsubscribe(self, topic: str, handler: Callable[[Event], Any]) -> None:
        """
        Unsubscribe from events on a topic.

        Args:
            topic (str): Topic to unsubscribe from.
            handler (Callable[[Event], Any]): Function to unsubscribe.

        Raises:
            Exception: If unsubscription fails.
        """
        pass

# Create a global event bus
event_bus = EventBus()

# Start the event bus
event_bus.start()

# Define common event types
class UserEvent(Event):
    """
    Base class for user-related events.

    Attributes:
        user_id (str): ID of the user.
    """

    def __init__(self, user_id: str, data: Dict[str, Any], source: str = None):
        """
        Initialize a user event.

        Args:
            user_id (str): ID of the user.
            data (Dict[str, Any]): Event data.
            source (str, optional): Source of the event. Defaults to None.
        """
        super().__init__(f"user.{self.__class__.__name__.lower()}", data, source)
        self.user_id = user_id
        self.data["user_id"] = user_id

class UserCreatedEvent(UserEvent):
    """
    Event fired when a user is created.

    Attributes:
        email (str): Email of the user.
        role (str): Role of the user.
    """

    def __init__(self, user_id: str, email: str, role: str, source: str = None):
        """
        Initialize a user created event.

        Args:
            user_id (str): ID of the user.
            email (str): Email of the user.
            role (str): Role of the user.
            source (str, optional): Source of the event. Defaults to None.
        """
        super().__init__(user_id, {"email": email, "role": role}, source)

class UserUpdatedEvent(UserEvent):
    """
    Event fired when a user is updated.

    Attributes:
        updated_fields (Dict[str, Any]): Fields that were updated.
    """

    def __init__(self, user_id: str, updated_fields: Dict[str, Any], source: str = None):
        """
        Initialize a user updated event.

        Args:
            user_id (str): ID of the user.
            updated_fields (Dict[str, Any]): Fields that were updated.
            source (str, optional): Source of the event. Defaults to None.
        """
        super().__init__(user_id, {"updated_fields": updated_fields}, source)

class UserDeletedEvent(UserEvent):
    """
    Event fired when a user is deleted.
    """

    def __init__(self, user_id: str, source: str = None):
        """
        Initialize a user deleted event.

        Args:
            user_id (str): ID of the user.
            source (str, optional): Source of the event. Defaults to None.
        """
        super().__init__(user_id, {}, source)

class SearchEvent(Event):
    """
    Base class for search-related events.

    Attributes:
        query (str): Search query.
    """

    def __init__(self, query: str, data: Dict[str, Any], source: str = None):
        """
        Initialize a search event.

        Args:
            query (str): Search query.
            data (Dict[str, Any]): Event data.
            source (str, optional): Source of the event. Defaults to None.
        """
        super().__init__(f"search.{self.__class__.__name__.lower()}", data, source)
        self.query = query
        self.data["query"] = query

class SearchPerformedEvent(SearchEvent):
    """
    Event fired when a search is performed.

    Attributes:
        filters (Dict[str, Any]): Search filters.
        user_id (Optional[str]): ID of the user who performed the search.
    """

    def __init__(self, query: str, filters: Dict[str, Any], user_id: str = None, source: str = None):
        """
        Initialize a search performed event.

        Args:
            query (str): Search query.
            filters (Dict[str, Any]): Search filters.
            user_id (str, optional): ID of the user who performed the search. Defaults to None.
            source (str, optional): Source of the event. Defaults to None.
        """
        super().__init__(query, {"filters": filters, "user_id": user_id}, source)

class SearchCompletedEvent(SearchEvent):
    """
    Event fired when a search is completed.

    Attributes:
        result_count (int): Number of results.
        duration_ms (float): Duration of the search in milliseconds.
        user_id (Optional[str]): ID of the user who performed the search.
    """

    def __init__(self, query: str, result_count: int, duration_ms: float, user_id: str = None, source: str = None):
        """
        Initialize a search completed event.

        Args:
            query (str): Search query.
            result_count (int): Number of results.
            duration_ms (float): Duration of the search in milliseconds.
            user_id (str, optional): ID of the user who performed the search. Defaults to None.
            source (str, optional): Source of the event. Defaults to None.
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
    """
    Base class for analysis-related events.

    Attributes:
        analysis_id (str): ID of the analysis.
    """

    def __init__(self, analysis_id: str, data: Dict[str, Any], source: str = None):
        """
        Initialize an analysis event.

        Args:
            analysis_id (str): ID of the analysis.
            data (Dict[str, Any]): Event data.
            source (str, optional): Source of the event. Defaults to None.
        """
        super().__init__(f"analysis.{self.__class__.__name__.lower()}", data, source)
        self.analysis_id = analysis_id
        self.data["analysis_id"] = analysis_id

class AnalysisStartedEvent(AnalysisEvent):
    """
    Event fired when an analysis is started.

    Attributes:
        analysis_type (str): Type of analysis.
        user_id (Optional[str]): ID of the user who started the analysis.
    """

    def __init__(self, analysis_id: str, analysis_type: str, user_id: str = None, source: str = None):
        """
        Initialize an analysis started event.

        Args:
            analysis_id (str): ID of the analysis.
            analysis_type (str): Type of analysis.
            user_id (str, optional): ID of the user who started the analysis. Defaults to None.
            source (str, optional): Source of the event. Defaults to None.
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
    """
    Event fired when an analysis is completed.

    Attributes:
        result (Any): Analysis result.
        duration_ms (float): Duration of the analysis in milliseconds.
    """

    def __init__(self, analysis_id: str, result: Any, duration_ms: float, source: str = None):
        """
        Initialize an analysis completed event.

        Args:
            analysis_id (str): ID of the analysis.
            result (Any): Analysis result.
            duration_ms (float): Duration of the analysis in milliseconds.
            source (str, optional): Source of the event. Defaults to None.
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
    """
    Event fired when an analysis fails.

    Attributes:
        error (str): Error message.
    """

    def __init__(self, analysis_id: str, error: str, source: str = None):
        """
        Initialize an analysis failed event.

        Args:
            analysis_id (str): ID of the analysis.
            error (str): Error message.
            source (str, optional): Source of the event. Defaults to None.
        """
        super().__init__(
            analysis_id,
            {
                "error": error
            },
            source
        )

class TaskEvent(Event):
    """
    Base class for task-related events.

    Attributes:
        task_id (str): ID of the task.
    """

    def __init__(self, task_id: str, data: Dict[str, Any], source: str = None):
        """
        Initialize a task event.

        Args:
            task_id (str): ID of the task.
            data (Dict[str, Any]): Event data.
            source (str, optional): Source of the event. Defaults to None.
        """
        super().__init__(f"task.{self.__class__.__name__.lower()}", data, source)
        self.task_id = task_id
        self.data["task_id"] = task_id

class TaskStartedEvent(TaskEvent):
    """
    Event fired when a task is started.

    Attributes:
        task_type (str): Type of task.
        params (Dict[str, Any]): Task parameters.
    """

    def __init__(self, task_id: str, task_type: str, params: Dict[str, Any], source: str = None):
        """
        Initialize a task started event.

        Args:
            task_id (str): ID of the task.
            task_type (str): Type of task.
            params (Dict[str, Any]): Task parameters.
            source (str, optional): Source of the event. Defaults to None.
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
    """
    Event fired when a task makes progress.

    Attributes:
        progress (float): Progress percentage (0-100).
        message (str): Progress message.
    """

    def __init__(self, task_id: str, progress: float, message: str, source: str = None):
        """
        Initialize a task progress event.

        Args:
            task_id (str): ID of the task.
            progress (float): Progress percentage (0-100).
            message (str): Progress message.
            source (str, optional): Source of the event. Defaults to None.
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
    """
    Event fired when a task is completed.

    Attributes:
        result (Any): Task result.
        duration_ms (float): Duration of the task in milliseconds.
    """

    def __init__(self, task_id: str, result: Any, duration_ms: float, source: str = None):
        """
        Initialize a task completed event.

        Args:
            task_id (str): ID of the task.
            result (Any): Task result.
            duration_ms (float): Duration of the task in milliseconds.
            source (str, optional): Source of the event. Defaults to None.
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
    """
    Event fired when a task fails.

    Attributes:
        error (str): Error message.
    """

    def __init__(self, task_id: str, error: str, source: str = None):
        """
        Initialize a task failed event.

        Args:
            task_id (str): ID of the task.
            error (str): Error message.
            source (str, optional): Source of the event. Defaults to None.
        """
        super().__init__(
            task_id,
            {
                "error": error
            },
            source
        )