"""
RabbitMQ message broker for the Medical Research Synthesizer.
This module provides a RabbitMQ-based implementation of the message broker interface,
allowing services to publish and consume messages in an event-driven architecture.
"""

import json
import asyncio
import time
from typing import Dict, Any, Optional, Callable, TypeVar, Union
from enum import Enum
import aio_pika
from aio_pika import Message, DeliveryMode, ExchangeType, connect_robust
from aio_pika.abc import AbstractIncomingMessage, AbstractRobustConnection, AbstractRobustChannel
from ..config import settings
from ..logging_config import get_logger
from ..exceptions import MessageBrokerError, ConnectionError, MessageError
logger = get_logger(__name__)
T = TypeVar('T')
class MessagePriority(int, Enum):
    """
    Message priority levels.
    
    Defines standard priority levels for messages in the messaging system.
    Higher values indicate higher priority.
    """
    LOWEST = 0
    LOW = 3
    NORMAL = 5
    HIGH = 7
    HIGHEST = 10

class MessageSerializer:
    """
    Serializer for message data.
    
    Handles conversion between Python objects and bytes for message transport.
    Uses JSON serialization with UTF-8 encoding.
    """
    @staticmethod
    def serialize(data: Any) -> bytes:
        """
        Serialize data to bytes.
        
        Args:
            data: Data to serialize
            
        Returns:
            Serialized data as bytes
            
        Raises:
            MessageError: If serialization fails
        """
        try:
            return json.dumps(data).encode('utf-8')
        except (TypeError, ValueError) as e:
            logger.error(f"Error serializing message: {str(e)}", exc_info=e)
            raise MessageError("serialize", f"Failed to serialize message: {str(e)}")

    @staticmethod
    def deserialize(data: bytes) -> Any:
        """
        Deserialize bytes to data.
        
        Args:
            data: Bytes to deserialize
            
        Returns:
            Deserialized data
            
        Raises:
            MessageError: If deserialization fails
        """
        try:
            return json.loads(data.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.error(f"Error deserializing message: {str(e)}", exc_info=e)
            raise MessageError("deserialize", f"Failed to deserialize message: {str(e)}")

class CircuitBreaker:
    """
    Circuit breaker for handling connection failures.
    
    This class implements the circuit breaker pattern to prevent repeated
    connection attempts when the broker is unavailable.
    """
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        reset_timeout: float = 60.0
    ):
        """
        Initialize the circuit breaker.
        Args:
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds to wait before attempting recovery
            reset_timeout: Time in seconds to wait before resetting the circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
    def record_failure(self) -> None:
        """
        Record a connection failure.
        
        Increments the failure count and updates the circuit state if needed.
        If the failure threshold is reached, opens the circuit.
        
        Returns:
            None
        """
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == "CLOSED" and self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} consecutive failures"
            )
        
        logger.debug(
            f"Circuit breaker recorded failure (count: {self.failure_count}, state: {self.state})"
        )
    def reset(self) -> None:
        """
        Reset the circuit breaker to its initial state.
        
        Resets the failure count and changes the state to CLOSED.
        
        Returns:
            None
        """
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"
        logger.debug("Circuit breaker reset to initial closed state")
    def allow_request(self) -> bool:
        """
        Check if requests are allowed based on the circuit state.
        
        Returns:
            True if requests are allowed, False otherwise
        """
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF-OPEN"
                logger.debug("Circuit breaker moved to half-open state")
                return True
            return False
        elif self.state == "HALF-OPEN":
            return True
        return False
    def record_success(self) -> None:
        """
        Record a successful connection attempt.
        
        Resets the failure count and closes the circuit if in half-open state.
        
        Returns:
            None
        """
        if self.state == "HALF-OPEN":
            self.reset()
            logger.info("Circuit breaker closed after successful attempt")
        elif self.state == "CLOSED":
            self.failure_count = 0
class RabbitMQBroker:
    """
    RabbitMQ message broker for the Medical Research Synthesizer.
    
    This class provides methods to connect to RabbitMQ, declare exchanges and queues,
    publish and consume messages, and manage the broker connection.
    """
    def __init__(
        self,
        host: str = None,
        port: int = None,
        username: str = None,
        password: str = None,
        vhost: str = None,
        connection_name: str = None
    ):
        """
        Initialize the RabbitMQ broker.
        Args:
            host: RabbitMQ host (default: from settings)
            port: RabbitMQ port (default: from settings)
            username: RabbitMQ username (default: from settings)
            password: RabbitMQ password (default: from settings)
            vhost: RabbitMQ virtual host (default: from settings)
            connection_name: Name for the connection (default: "asf-medical")
        """
        self.host = host or settings.RABBITMQ_HOST
        self.port = port or settings.RABBITMQ_PORT
        self.username = username or settings.RABBITMQ_USERNAME
        self.password = password or settings.RABBITMQ_PASSWORD
        self.vhost = vhost or settings.RABBITMQ_VHOST
        self.connection_name = connection_name or "asf-medical"
        self.connection_url = f"amqp://{self.username}:{self.password}@{self.host}:{self.port}/{self.vhost}"
        self.connection: Optional[AbstractRobustConnection] = None
        self.channel: Optional[AbstractRobustChannel] = None
        self.exchanges: Dict[str, aio_pika.RobustExchange] = {}
        self.queues: Dict[str, aio_pika.RobustQueue] = {}
        self.circuit_breaker = CircuitBreaker()
        self.serializer = MessageSerializer()
        self._closing = False
        self._connection_lock = asyncio.Lock()
        logger.info(f"Initialized RabbitMQ broker with host: {self.host}")
    async def connect(self) -> None:
        """
        Connect to RabbitMQ.
        Raises:
            ConnectionError: If connection fails
        """
        if self.connection and not self.connection.is_closed:
            return
        if not self.circuit_breaker.allow_request():
            raise ConnectionError(
                "rabbitmq",
                "Connection attempts are blocked by circuit breaker",
                details={"state": self.circuit_breaker.state}
            )
        async with self._connection_lock:
            # Check again in case another task acquired the lock first
            if self.connection and not self.connection.is_closed:
                return
            try:
                logger.info(f"Connecting to RabbitMQ: {self.host}:{self.port}")
                self.connection = await connect_robust(
                    self.connection_url,
                    client_properties={
                        "connection_name": self.connection_name,
                        "product": "ASF Medical Research Synthesizer",
                        "version": settings.VERSION
                    }
                )
                self.channel = await self.connection.channel()
                await self.channel.set_qos(prefetch_count=10)
                self.circuit_breaker.record_success()
                logger.info("Connected to RabbitMQ successfully")
            except Exception as e:
                self.circuit_breaker.record_failure()
                logger.error(f"Failed to connect to RabbitMQ: {str(e)}", exc_info=e)
                raise ConnectionError("rabbitmq", f"Failed to connect: {str(e)}")
    async def close(self) -> None:
        """
        Close the connection to RabbitMQ.
        Raises:
            ConnectionError: If disconnection fails
        """
        if not self.connection:
            return
        self._closing = True
        try:
            logger.info("Closing RabbitMQ connection")
            # Close channel first
            if self.channel:
                await self.channel.close()
                self.channel = None
            # Then close connection
            if self.connection:
                await self.connection.close()
                self.connection = None
            self.exchanges.clear()
            self.queues.clear()
            logger.info("RabbitMQ connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing RabbitMQ connection: {str(e)}", exc_info=e)
            raise ConnectionError("rabbitmq", f"Failed to disconnect: {str(e)}")
        finally:
            self._closing = False
    async def declare_exchange(
        self,
        name: str,
        type: Union[str, ExchangeType] = ExchangeType.TOPIC,
        durable: bool = True
    ) -> aio_pika.RobustExchange:
        """
        Declare an exchange.
        Args:
            name: Exchange name
            type: Exchange type (default: TOPIC)
            durable: Whether the exchange should survive broker restarts
        Returns:
            The declared exchange
        Raises:
            ConnectionError: If connection fails
            MessageBrokerError: If exchange declaration fails
        """
        await self.connect()
        if name in self.exchanges:
            return self.exchanges[name]
        try:
            exchange = await self.channel.declare_exchange(
                name=name,
                type=type,
                durable=durable
            )
            self.exchanges[name] = exchange
            logger.debug(f"Declared exchange: {name} (type={type}, durable={durable})")
            return exchange
        except Exception as e:
            logger.error(f"Failed to declare exchange {name}: {str(e)}", exc_info=e)
            raise MessageBrokerError(f"Failed to declare exchange {name}: {str(e)}")
    async def declare_queue(
        self,
        name: str,
        durable: bool = True,
        exclusive: bool = False,
        auto_delete: bool = False,
        arguments: Dict[str, Any] = None
    ) -> aio_pika.RobustQueue:
        """
        Declare a queue.
        Args:
            name: Queue name
            durable: Whether the queue should survive broker restarts
            exclusive: Whether the queue is exclusive to this connection
            auto_delete: Whether the queue should be deleted when no longer used
            arguments: Additional arguments for the queue
        Returns:
            The declared queue
        Raises:
            ConnectionError: If connection fails
            MessageBrokerError: If queue declaration fails
        """
        await self.connect()
        if name in self.queues:
            return self.queues[name]
        try:
            queue = await self.channel.declare_queue(
                name=name,
                durable=durable,
                exclusive=exclusive,
                auto_delete=auto_delete,
                arguments=arguments
            )
            self.queues[name] = queue
            logger.debug(f"Declared queue: {name} (durable={durable})")
            return queue
        except Exception as e:
            logger.error(f"Failed to declare queue {name}: {str(e)}", exc_info=e)
            raise MessageBrokerError(f"Failed to declare queue {name}: {str(e)}")
    async def bind_queue(
        self,
        queue_name: str,
        exchange_name: str,
        routing_key: str
    ) -> None:
        """
        Bind a queue to an exchange with a routing key.
        Args:
            queue_name: Queue name
            exchange_name: Exchange name
            routing_key: Routing key
        Raises:
            ConnectionError: If connection fails
            MessageBrokerError: If binding fails
        """
        await self.connect()
        try:
            # Get or declare the queue
            if queue_name not in self.queues:
                await self.declare_queue(queue_name)
            # Get or declare the exchange
            if exchange_name not in self.exchanges:
                await self.declare_exchange(exchange_name)
            queue = self.queues[queue_name]
            exchange = self.exchanges[exchange_name]
            await queue.bind(exchange, routing_key)
            logger.debug(f"Bound queue {queue_name} to exchange {exchange_name} with routing key {routing_key}")
        except Exception as e:
            logger.error(
                f"Failed to bind queue {queue_name} to exchange {exchange_name}: {str(e)}",
                exc_info=e
            )
            raise MessageBrokerError(
                f"Failed to bind queue {queue_name} to exchange {exchange_name}: {str(e)}"
            )
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
        delivery_mode: DeliveryMode = DeliveryMode.PERSISTENT,
        headers: Dict[str, Any] = None
    ) -> None:
        """
        Publish a message to an exchange.
        Args:
            exchange_name: Exchange name
            routing_key: Routing key
            message: Message to publish
            message_id: Message ID
            correlation_id: Correlation ID for request-reply pattern
            reply_to: Queue name for replies
            expiration: Message expiration in milliseconds
            priority: Message priority
            delivery_mode: Delivery mode (PERSISTENT or TRANSIENT)
            headers: Message headers
        Raises:
            ConnectionError: If connection fails
            MessageError: If serialization fails
            MessageBrokerError: If publishing fails
        """
        await self.connect()
        try:
            # Get or declare the exchange
            if exchange_name not in self.exchanges:
                await self.declare_exchange(exchange_name)
            exchange = self.exchanges[exchange_name]
            # Serialize the message
            body = self.serializer.serialize(message)
            # Create the message
            message_obj = Message(
                body=body,
                message_id=message_id,
                correlation_id=correlation_id,
                reply_to=reply_to,
                expiration=expiration,
                priority=int(priority),
                delivery_mode=delivery_mode,
                headers=headers or {}
            )
            # Publish the message
            await exchange.publish(message_obj, routing_key)
            logger.debug(
                f"Published message to exchange {exchange_name} with routing key {routing_key}",
                extra={"message_id": message_id, "correlation_id": correlation_id}
            )
        except MessageError:
            # Re-raise serialization errors
            raise
        except Exception as e:
            logger.error(
                f"Failed to publish message to exchange {exchange_name}: {str(e)}",
                exc_info=e
            )
            raise MessageBrokerError(
                f"Failed to publish message to exchange {exchange_name}: {str(e)}"
            )
    async def consume(
        self,
        queue_name: str,
        callback: Callable[[Dict[str, Any], AbstractIncomingMessage], Any],
        no_ack: bool = False
    ) -> str:
        """
        Consume messages from a queue.
        Args:
            queue_name: Queue name
            callback: Callback function to process messages
            no_ack: Whether to automatically acknowledge messages
        Returns:
            Consumer tag
        Raises:
            ConnectionError: If connection fails
            MessageBrokerError: If consumption fails
        """
        await self.connect()
        try:
            # Get or declare the queue
            if queue_name not in self.queues:
                await self.declare_queue(queue_name)
            queue = self.queues[queue_name]
            # Wrap the callback to handle deserialization
            async def wrapped_callback(message: AbstractIncomingMessage) -> None:
                async with message.process(requeue=True):
                    try:
                        # Deserialize the message
                        body = self.serializer.deserialize(message.body)
                        # Call the callback
                        await callback(body, message)
                        # Acknowledge the message if not using no_ack
                        if not no_ack:
                            await message.ack()
                    except MessageError as e:
                        logger.error(
                            f"Error processing message: {str(e)}",
                            extra={"queue": queue_name, "message_id": message.message_id},
                            exc_info=e
                        )
                        # Reject the message without requeuing for deserialization errors
                        await message.reject(requeue=False)
                    except Exception as e:
                        logger.error(
                            f"Error in message callback: {str(e)}",
                            extra={"queue": queue_name, "message_id": message.message_id},
                            exc_info=e
                        )
                        # Reject the message with requeuing for other errors
                        await message.reject(requeue=True)
            # Start consuming
            consumer_tag = await queue.consume(wrapped_callback, no_ack=no_ack)
            logger.info(f"Started consuming from queue {queue_name} with tag {consumer_tag}")
            return consumer_tag
        except Exception as e:
            logger.error(f"Failed to consume from queue {queue_name}: {str(e)}", exc_info=e)
            raise MessageBrokerError(f"Failed to consume from queue {queue_name}: {str(e)}")
    async def cancel_consumer(self, consumer_tag: str) -> None:
        """
        Cancel a consumer.
        Args:
            consumer_tag: Consumer tag
        Raises:
            ConnectionError: If connection fails
            MessageBrokerError: If cancellation fails
        """
        if not self.channel:
            return
        try:
            await self.channel.cancel(consumer_tag)
            logger.info(f"Cancelled consumer with tag {consumer_tag}")
        except Exception as e:
            logger.error(f"Failed to cancel consumer {consumer_tag}: {str(e)}", exc_info=e)
            raise MessageBrokerError(f"Failed to cancel consumer {consumer_tag}: {str(e)}")
    async def purge_queue(self, queue_name: str) -> int:
        """
        Purge a queue.
        Args:
            queue_name: Queue name
        Returns:
            Number of messages purged
        Raises:
            ConnectionError: If connection fails
            MessageBrokerError: If purging fails
        """
        await self.connect()
        try:
            # Get or declare the queue
            if queue_name not in self.queues:
                await self.declare_queue(queue_name)
            queue = self.queues[queue_name]
            # Purge the queue
            message_count = await queue.purge()
            logger.info(f"Purged {message_count} messages from queue {queue_name}")
            return message_count
        except Exception as e:
            logger.error(f"Failed to purge queue {queue_name}: {str(e)}", exc_info=e)
            raise MessageBrokerError(f"Failed to purge queue {queue_name}: {str(e)}")
    async def delete_queue(self, queue_name: str) -> None:
        """
        Delete a queue.
        Args:
            queue_name: Queue name
        Raises:
            ConnectionError: If connection fails
            MessageBrokerError: If deletion fails
        """
        await self.connect()
        try:
            # Get or declare the queue
            if queue_name not in self.queues:
                await self.declare_queue(queue_name)
            queue = self.queues[queue_name]
            # Delete the queue
            await queue.delete()
            self.queues.pop(queue_name, None)
            logger.info(f"Deleted queue {queue_name}")
        except Exception as e:
            logger.error(f"Failed to delete queue {queue_name}: {str(e)}", exc_info=e)
            raise MessageBrokerError(f"Failed to delete queue {queue_name}: {str(e)}")
    async def delete_exchange(self, exchange_name: str) -> None:
        """
        Delete an exchange.
        Args:
            exchange_name: Exchange name
        Raises:
            ConnectionError: If connection fails
            MessageBrokerError: If deletion fails
        """
        await self.connect()
        try:
            # Get or declare the exchange
            if exchange_name not in self.exchanges:
                await self.declare_exchange(exchange_name)
            exchange = self.exchanges[exchange_name]
            # Delete the exchange
            await exchange.delete()
            self.exchanges.pop(exchange_name, None)
            logger.info(f"Deleted exchange {exchange_name}")
        except Exception as e:
            logger.error(f"Failed to delete exchange {exchange_name}: {str(e)}", exc_info=e)
            raise MessageBrokerError(f"Failed to delete exchange {exchange_name}: {str(e)}")

# Create a global RabbitMQ broker instance
rabbitmq_broker = RabbitMQBroker()
# Function to get the global RabbitMQ broker instance
def get_rabbitmq_broker() -> RabbitMQBroker:
    """
    Get the global RabbitMQ broker instance.
    Returns:
        Global RabbitMQ broker instance
    """
    return rabbitmq_broker