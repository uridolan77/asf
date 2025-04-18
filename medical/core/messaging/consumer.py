"""
Message consumer for the Medical Research Synthesizer.
This module provides a message consumer for consuming messages from the message broker.
"""
from typing import Dict, Any, List, Set, TypeVar
from abc import ABC, abstractmethod
from aio_pika.abc import AbstractIncomingMessage
from ..logging_config import get_logger
from .rabbitmq_broker import RabbitMQBroker, get_rabbitmq_broker
from .producer import MessageProducer, get_message_producer
logger = get_logger(__name__)
T = TypeVar('T')
class MessageHandler(ABC):
    """
    Base class for message handlers.
    
    Message handlers process messages from the message broker.
    They can be registered with a consumer to handle specific message types.
    """
    @abstractmethod
    async def handle(self, message: Dict[str, Any], properties: Dict[str, Any]) -> Any:
        """
        Handle a message.
        Args:
            message: Message data
            properties: Message properties
        Returns:
            Handler result
        """
        pass
class EventHandler(MessageHandler):
    """
    Base class for event handlers.
    
    Event handlers process events from the message broker.
    They can be registered with a consumer to handle specific event types.
    """
    @abstractmethod
    async def handle_event(self, event_type: str, event_data: Dict[str, Any], properties: Dict[str, Any]) -> Any:
        """
        Handle an event.
        Args:
            event_type: Event type
            event_data: Event data
            properties: Event properties
        Returns:
            Handler result
        """
        pass
    async def handle(self, message: Dict[str, Any], properties: Dict[str, Any]) -> Any:
        """
        Handle a message.
        Args:
            message: Message data
            properties: Message properties
        Returns:
            Handler result
        """
        event_type = message.get("type")
        event_data = message.get("data", {})
        return await self.handle_event(event_type, event_data, properties)
class TaskHandler(MessageHandler):
    """
    Base class for task handlers.
    
    Task handlers process tasks from the message broker.
    They can be registered with a consumer to handle specific task types.
    """
    @abstractmethod
    async def handle_task(self, task_id: str, task_type: str, task_data: Dict[str, Any], properties: Dict[str, Any]) -> Any:
        """
        Handle a task.
        Args:
            task_id: Task ID
            task_type: Task type
            task_data: Task data
            properties: Task properties
        Returns:
            Handler result
        """
        pass
    async def handle(self, message: Dict[str, Any], properties: Dict[str, Any]) -> Any:
        """
        Handle a message.
        Args:
            message: Message data
            properties: Message properties
        Returns:
            Handler result
        """
        task_id = message.get("id")
        task_type = message.get("type")
        task_data = message.get("data", {})
        return await self.handle_task(task_id, task_type, task_data, properties)
class CommandHandler(MessageHandler):
    """
    Base class for command handlers.
    
    Command handlers process commands from the message broker.
    They can be registered with a consumer to handle specific command types.
    """
    def __init__(self, producer: MessageProducer = None):
        """
        Initialize the command handler.
        Args:
            producer: Message producer for sending replies (default: global instance)
        """
        self.producer = producer or get_message_producer()
    @abstractmethod
    async def handle_command(
        self,
        command_id: str,
        command_type: str,
        command_data: Dict[str, Any],
        properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle a command.
        Args:
            command_id: Command ID
            command_type: Command type
            command_data: Command data
            properties: Command properties
        Returns:
            Command result
        """
        pass
    async def handle(self, message: Dict[str, Any], properties: Dict[str, Any]) -> Any:
        """
        Handle a message.
        Args:
            message: Message data
            properties: Message properties
        Returns:
            Handler result
        """
        command_id = message.get("id")
        command_type = message.get("type")
        command_data = message.get("data", {})
        # Handle the command
        result = await self.handle_command(command_id, command_type, command_data, properties)
        # Send a reply if reply_to is provided
        if properties.get("reply_to") and properties.get("correlation_id"):
            await self.producer.publish_reply(
                reply_data=result,
                correlation_id=properties["correlation_id"],
                reply_to=properties["reply_to"]
            )
        return result
class MessageConsumer:
    """
    Message consumer for consuming messages from the message broker.
    
    This class provides methods for consuming messages from queues
    and routing them to registered handlers based on message type.
    """
    def __init__(self, broker: RabbitMQBroker = None, producer: MessageProducer = None):
        """
        Initialize the message consumer.
        Args:
            broker: RabbitMQ broker instance (default: global instance)
            producer: Message producer for sending replies (default: global instance)
        """
        self.broker = broker or get_rabbitmq_broker()
        self.producer = producer or get_message_producer()
        self.handlers: Dict[str, List[MessageHandler]] = {}
        self.consumer_tags: Set[str] = set()
        self._running = False
        logger.info("Initialized message consumer")
    def register_handler(self, routing_key: str, handler: MessageHandler) -> None:
        """
        Register a handler for a routing key.
        Args:
            routing_key: Routing key to handle
            handler: Message handler
        """
        if routing_key not in self.handlers:
            self.handlers[routing_key] = []
        self.handlers[routing_key].append(handler)
        logger.debug(f"Registered handler for routing key: {routing_key}")
    def register_event_handler(self, event_type: str, handler: EventHandler) -> None:
        """
        Register a handler for an event type.
        Args:
            event_type: Event type to handle
            handler: Event handler
        """
        self.register_handler(event_type, handler)
        logger.debug(f"Registered event handler for event type: {event_type}")
    def register_task_handler(self, task_type: str, handler: TaskHandler) -> None:
        """
        Register a handler for a task type.
        Args:
            task_type: Task type to handle
            handler: Task handler
        """
        self.register_handler(task_type, handler)
        logger.debug(f"Registered task handler for task type: {task_type}")
    def register_command_handler(self, command_type: str, target: str, handler: CommandHandler) -> None:
        """
        Register a handler for a command type.
        Args:
            command_type: Command type to handle
            target: Target service
            handler: Command handler
        """
        routing_key = f"{target}.{command_type}"
        self.register_handler(routing_key, handler)
        logger.debug(f"Registered command handler for command type: {command_type}, target: {target}")
    async def _message_callback(self, message: Dict[str, Any], original_message: AbstractIncomingMessage) -> None:
        """
        Callback for handling messages.
        Args:
            message: Deserialized message
            original_message: Original message object
        """
        # Extract message properties
        properties = {
            "message_id": original_message.message_id,
            "correlation_id": original_message.correlation_id,
            "reply_to": original_message.reply_to,
            "routing_key": original_message.routing_key,
            "exchange": original_message.exchange,
            "headers": original_message.headers or {},
            "delivery_tag": original_message.delivery_tag,
            "redelivered": original_message.redelivered,
            "priority": original_message.priority,
            "timestamp": original_message.timestamp,
        }
        # Find handlers for this routing key
        routing_key = original_message.routing_key
        handlers = self.handlers.get(routing_key, [])
        # Also check for wildcard handlers
        for pattern, pattern_handlers in self.handlers.items():
            if "*" in pattern:
                # Convert pattern to regex
                regex_pattern = pattern.replace(".", "\\.").replace("*", "[^.]+")
                if routing_key.match(regex_pattern):
                    handlers.extend(pattern_handlers)
        if not handlers:
            logger.warning(f"No handlers found for routing key: {routing_key}")
            return
        # Call all handlers
        for handler in handlers:
            try:
                await handler.handle(message, properties)
            except Exception as e:
                logger.error(
                    f"Error in message handler: {str(e)}",
                    extra={
                        "routing_key": routing_key,
                        "message_id": properties["message_id"],
                        "handler": handler.__class__.__name__
                    },
                    exc_info=e
                )
    async def start(self) -> None:
        """
        Start consuming messages.
        Raises:
            ConnectionError: If connection fails
            MessageBrokerError: If consumption fails
        """
        if self._running:
            return
        self._running = True
        # Declare exchanges
        await self.broker.declare_exchange("events", "topic")
        await self.broker.declare_exchange("tasks", "topic")
        await self.broker.declare_exchange("commands", "topic")
        # Declare and bind queues for each handler
        for routing_key in self.handlers:
            queue_name = f"asf-medical.{routing_key}"
            # Determine the exchange based on the routing key
            if "." in routing_key:
                # Commands have the format "target.command_type"
                exchange_name = "commands"
            elif routing_key.startswith("task."):
                exchange_name = "tasks"
            else:
                exchange_name = "events"
            # Declare the queue
            await self.broker.declare_queue(queue_name)
            # Bind the queue to the exchange
            await self.broker.bind_queue(queue_name, exchange_name, routing_key)
            # Start consuming
            consumer_tag = await self.broker.consume(queue_name, self._message_callback)
            self.consumer_tags.add(consumer_tag)
        logger.info("Started message consumer")
    async def stop(self) -> None:
        """
        Stop consuming messages.
        Raises:
            ConnectionError: If connection fails
            MessageBrokerError: If cancellation fails
        """
        if not self._running:
            return
        self._running = False
        # Cancel all consumers
        for consumer_tag in self.consumer_tags:
            try:
                await self.broker.cancel_consumer(consumer_tag)
            except Exception as e:
                logger.error(f"Error cancelling consumer {consumer_tag}: {str(e)}", exc_info=e)
        self.consumer_tags.clear()
        logger.info("Stopped message consumer")
# Create a global message consumer instance
message_consumer = MessageConsumer()
# Function to get the global message consumer instance
def get_message_consumer() -> MessageConsumer:
    """
    Get the global message consumer instance.
    Returns:
        Global message consumer instance
    """
    return message_consumer