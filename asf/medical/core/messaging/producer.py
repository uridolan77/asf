"""
Message producer for the Medical Research Synthesizer.

This module provides a message producer for publishing messages to the message broker.
"""
import uuid
from typing import Dict, Any, Optional, Union
from datetime import datetime

from aio_pika import DeliveryMode

from ..logging_config import get_logger
from .rabbitmq_broker import (
    RabbitMQBroker, get_rabbitmq_broker, MessagePriority
)

logger = get_logger(__name__)

class MessageProducer:
    """
    Message producer for publishing messages to the message broker.
    
    This class provides methods for publishing messages to exchanges
    with various options for routing, delivery, and message properties.
    """

    def __init__(self, broker: RabbitMQBroker = None):
        """
        Initialize the message producer.

        Args:
            broker: RabbitMQ broker instance (default: global instance)
        """
        self.broker = broker or get_rabbitmq_broker()
        logger.info("Initialized message producer")

    async def publish(
        self,
        exchange_name: str,
        routing_key: str,
        message: Any,
        message_id: str = None,
        correlation_id: str = None,
        reply_to: str = None,
        expiration: int = None,
        priority: Union[int, MessagePriority] = MessagePriority.NORMAL,
        persistent: bool = True,
        headers: Dict[str, Any] = None
    ) -> str:
        """
        Publish a message to an exchange.

        Args:
            exchange_name: Exchange name
            routing_key: Routing key
            message: Message to publish
            message_id: Message ID (default: generated UUID)
            correlation_id: Correlation ID for request-reply pattern
            reply_to: Queue name for replies
            expiration: Message expiration in milliseconds
            priority: Message priority
            persistent: Whether the message should be persistent
            headers: Message headers

        Returns:
            Message ID

        Raises:
            ConnectionError: If connection fails
            MessageError: If serialization fails
            MessageBrokerError: If publishing fails
        """
        # Generate a message ID if not provided
        message_id = message_id or str(uuid.uuid4())

        # Add timestamp to headers if not present
        headers = headers or {}
        if "timestamp" not in headers:
            headers["timestamp"] = datetime.utcnow().isoformat()

        # Set delivery mode based on persistence
        delivery_mode = DeliveryMode.PERSISTENT if persistent else DeliveryMode.TRANSIENT

        # Publish the message
        await self.broker.publish(
            exchange_name=exchange_name,
            routing_key=routing_key,
            message=message,
            message_id=message_id,
            correlation_id=correlation_id,
            reply_to=reply_to,
            expiration=expiration,
            priority=priority,
            delivery_mode=delivery_mode,
            headers=headers
        )

        return message_id

    async def publish_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        routing_key: str = None,
        correlation_id: str = None,
        priority: Union[int, MessagePriority] = MessagePriority.NORMAL,
        headers: Dict[str, Any] = None
    ) -> str:
        """
        Publish an event to the events exchange.

        Args:
            event_type: Event type
            event_data: Event data
            routing_key: Routing key (default: event type)
            correlation_id: Correlation ID
            priority: Event priority
            headers: Event headers

        Returns:
            Event ID

        Raises:
            ConnectionError: If connection fails
            MessageError: If serialization fails
            MessageBrokerError: If publishing fails
        """
        # Use event type as routing key if not provided
        routing_key = routing_key or event_type

        # Create the event message
        event = {
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Add source to headers if not present
        headers = headers or {}
        if "source" not in headers:
            headers["source"] = "asf-medical"

        # Publish the event
        return await self.publish(
            exchange_name="events",
            routing_key=routing_key,
            message=event,
            correlation_id=correlation_id,
            priority=priority,
            headers=headers
        )

    async def publish_task(
        self,
        task_type: str,
        task_data: Dict[str, Any],
        task_id: str = None,
        correlation_id: str = None,
        priority: Union[int, MessagePriority] = MessagePriority.NORMAL,
        headers: Dict[str, Any] = None
    ) -> str:
        """
        Publish a task to the tasks exchange.

        Args:
            task_type: Task type
            task_data: Task data
            task_id: Task ID (default: generated UUID)
            correlation_id: Correlation ID
            priority: Task priority
            headers: Task headers

        Returns:
            Task ID

        Raises:
            ConnectionError: If connection fails
            MessageError: If serialization fails
            MessageBrokerError: If publishing fails
        """
        # Generate a task ID if not provided
        task_id = task_id or str(uuid.uuid4())

        # Create the task message
        task = {
            "id": task_id,
            "type": task_type,
            "data": task_data,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Add source to headers if not present
        headers = headers or {}
        if "source" not in headers:
            headers["source"] = "asf-medical"

        # Publish the task
        await self.publish(
            exchange_name="tasks",
            routing_key=task_type,
            message=task,
            message_id=task_id,
            correlation_id=correlation_id,
            priority=priority,
            headers=headers
        )

        return task_id

    async def publish_command(
        self,
        command_type: str,
        command_data: Dict[str, Any],
        target: str,
        command_id: str = None,
        correlation_id: str = None,
        reply_to: str = None,
        priority: Union[int, MessagePriority] = MessagePriority.NORMAL,
        headers: Dict[str, Any] = None
    ) -> str:
        """
        Publish a command to the commands exchange.

        Args:
            command_type: Command type
            command_data: Command data
            target: Target service
            command_id: Command ID (default: generated UUID)
            correlation_id: Correlation ID
            reply_to: Queue name for replies
            priority: Command priority
            headers: Command headers

        Returns:
            Command ID

        Raises:
            ConnectionError: If connection fails
            MessageError: If serialization fails
            MessageBrokerError: If publishing fails
        """
        # Generate a command ID if not provided
        command_id = command_id or str(uuid.uuid4())

        # Create the command message
        command = {
            "id": command_id,
            "type": command_type,
            "data": command_data,
            "target": target,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Add source to headers if not present
        headers = headers or {}
        if "source" not in headers:
            headers["source"] = "asf-medical"

        # Publish the command
        await self.publish(
            exchange_name="commands",
            routing_key=f"{target}.{command_type}",
            message=command,
            message_id=command_id,
            correlation_id=correlation_id,
            reply_to=reply_to,
            priority=priority,
            headers=headers
        )

        return command_id

    async def publish_raw_message(
        self,
        exchange: str,
        routing_key: str,
        message: Dict[str, Any],
        headers: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        priority: Union[int, MessagePriority] = MessagePriority.NORMAL,
        persistent: bool = True
    ) -> str:
        """
        Publish a raw message to an exchange.

        This method is primarily used for reprocessing dead letter messages.

        Args:
            exchange: Exchange name
            routing_key: Routing key
            message: Message content
            headers: Message headers
            message_id: Message ID (default: generated UUID)
            correlation_id: Correlation ID
            priority: Message priority
            persistent: Whether the message should be persistent

        Returns:
            Message ID

        Raises:
            ConnectionError: If connection fails
            MessageError: If serialization fails
            MessageBrokerError: If publishing fails
        """
        # Generate a message ID if not provided
        message_id = message_id or str(uuid.uuid4())

        # Set delivery mode based on persistence
        delivery_mode = DeliveryMode.PERSISTENT if persistent else DeliveryMode.TRANSIENT

        # Publish the message
        await self.broker.publish(
            exchange_name=exchange,
            routing_key=routing_key,
            message=message,
            message_id=message_id,
            correlation_id=correlation_id,
            priority=priority,
            delivery_mode=delivery_mode,
            headers=headers
        )

        return message_id

    async def publish_reply(
        self,
        reply_data: Dict[str, Any],
        correlation_id: str,
        reply_to: str,
        reply_id: str = None,
        priority: Union[int, MessagePriority] = MessagePriority.NORMAL,
        headers: Dict[str, Any] = None
    ) -> str:
        """
        Publish a reply to a command or request.

        Args:
            reply_data: Reply data
            correlation_id: Correlation ID of the original message
            reply_to: Queue name for the reply
            reply_id: Reply ID (default: generated UUID)
            priority: Reply priority
            headers: Reply headers

        Returns:
            Reply ID

        Raises:
            ConnectionError: If connection fails
            MessageError: If serialization fails
            MessageBrokerError: If publishing fails
        """
        # Generate a reply ID if not provided
        reply_id = reply_id or str(uuid.uuid4())

        # Create the reply message
        reply = {
            "id": reply_id,
            "data": reply_data,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Add source to headers if not present
        headers = headers or {}
        if "source" not in headers:
            headers["source"] = "asf-medical"

        # Publish the reply
        await self.publish(
            exchange_name="",  # Default exchange for direct routing to queues
            routing_key=reply_to,
            message=reply,
            message_id=reply_id,
            correlation_id=correlation_id,
            priority=priority,
            headers=headers
        )

        return reply_id


# Create a global message producer instance
message_producer = MessageProducer()

# Function to get the global message producer instance
def get_message_producer() -> MessageProducer:
    """
    Get the global message producer instance.

    Returns:
        Global message producer instance
    """
    return message_producer
