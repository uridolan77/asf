"""
Redis Event Broker module for the Medical Research Synthesizer.

This module provides a Redis-based implementation of an event broker, allowing
for publishing and subscribing to events across processes or services.

Classes:
    RedisEventBroker: Redis-based event broker for publishing and subscribing to events.
    EventBridge: Bridge between Redis events and the local event bus.
"""

import json
import asyncio
from typing import Dict, Any, List, Callable, Set
import redis.asyncio as redis
from .events import Event, EventBroker, event_bus
from .config import settings
from .logging_config import get_logger

logger = get_logger(__name__)

class RedisEventBroker(EventBroker):
    """
    Redis-based event broker for publishing and subscribing to events.
    
    This class implements an event broker using Redis as the backend,
    allowing for cross-process or cross-service event communication.
    
    Attributes:
        redis_url (str): URL for the Redis server.
        redis_client (Optional[Redis]): Redis client instance.
        pubsub (Optional[PubSub]): Redis Pub/Sub instance.
        _handlers (Dict[str, List[Callable[[Event], Any]]]): Handlers for subscribed topics.
        _running (bool): Whether the broker is currently listening for events.
        _listen_task (Optional[asyncio.Task]): Task for listening to events.
    """
    def __init__(self, redis_url: str = None):
        """
        Initialize the Redis event broker.
        
        Args:
            redis_url (str): Redis connection URL (default: from settings).
        """
        self.redis_url = redis_url or settings.REDIS_URL
        if not self.redis_url:
            raise ValueError("Redis URL not configured")
        self.redis_client = None
        self.pubsub = None
        self._handlers: Dict[str, List[Callable[[Event], Any]]] = {}
        self._running = False
        self._listen_task = None
        logger.info(f"Initialized Redis event broker with URL: {self.redis_url}")

    async def connect(self) -> None:
        """
        Connect to Redis.
        
        Raises:
            Exception: If connection fails.
        """
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            self.pubsub = self.redis_client.pubsub()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}", exc_info=e)
            raise

    async def disconnect(self) -> None:
        """
        Disconnect from Redis.
        
        Raises:
            Exception: If disconnection fails.
        """
        try:
            if self._running:
                await self.stop_listening()
            if self.pubsub:
                await self.pubsub.close()
                self.pubsub = None
            if self.redis_client:
                await self.redis_client.close()
                self.redis_client = None
            logger.info("Disconnected from Redis")
        except Exception as e:
            logger.error(f"Failed to disconnect from Redis: {str(e)}", exc_info=e)
            raise

    async def publish(self, topic: str, event: Event) -> None:
        """
        Publish an event to a topic.
        
        Args:
            topic (str): Topic to publish to.
            event (Event): Event to publish.
        
        Raises:
            Exception: If publishing fails.
        """
        if not self.redis_client:
            await self.connect()
        try:
            message = json.dumps(event.to_dict())
            await self.redis_client.publish(topic, message)
            logger.debug(f"Published event to topic {topic}: {event}")
        except Exception as e:
            logger.error(f"Failed to publish event to topic {topic}: {str(e)}", exc_info=e)
            raise

    async def subscribe(self, topic: str, handler: Callable[[Event], Any]) -> None:
        """
        Subscribe to events on a topic.
        
        Args:
            topic (str): Topic to subscribe to.
            handler (Callable[[Event], Any]): Function to handle events.
        
        Raises:
            Exception: If subscription fails.
        """
        if not self.pubsub:
            await self.connect()
        try:
            if topic not in self._handlers:
                self._handlers[topic] = []
                await self.pubsub.subscribe(topic)
                logger.debug(f"Subscribed to topic: {topic}")
            self._handlers[topic].append(handler)
            logger.debug(f"Registered handler for topic: {topic}")
            if not self._running:
                await self.start_listening()
        except Exception as e:
            logger.error(f"Failed to subscribe to topic {topic}: {str(e)}", exc_info=e)
            raise

    async def unsubscribe(self, topic: str, handler: Callable[[Event], Any]) -> None:
        """
        Unsubscribe from events on a topic.
        
        Args:
            topic (str): Topic to unsubscribe from.
            handler (Callable[[Event], Any]): Function to unsubscribe.
        
        Raises:
            Exception: If unsubscription fails.
        """
        if not self.pubsub:
            return
        try:
            if topic in self._handlers:
                self._handlers[topic] = [h for h in self._handlers[topic] if h != handler]
                if not self._handlers[topic]:
                    await self.pubsub.unsubscribe(topic)
                    del self._handlers[topic]
                    logger.debug(f"Unsubscribed from topic: {topic}")
            logger.debug(f"Unregistered handler for topic: {topic}")
            if not self._handlers and self._running:
                await self.stop_listening()
        except Exception as e:
            logger.error(f"Failed to unsubscribe from topic {topic}: {str(e)}", exc_info=e)
            raise

    async def start_listening(self) -> None:
        """
        Start listening for events.
        
        Raises:
            Exception: If starting fails.
        """
        if self._running:
            return
        self._running = True
        self._listen_task = asyncio.create_task(self._listen_for_messages())
        logger.info("Started listening for Redis events")

    async def stop_listening(self) -> None:
        """
        Stop listening for events.
        
        Raises:
            Exception: If stopping fails.
        """
        if not self._running:
            return
        self._running = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None
        logger.info("Stopped listening for Redis events")

    async def _listen_for_messages(self) -> None:
        """
        Listen for messages from Redis Pub/Sub.
        """
        try:
            async for message in self.pubsub.listen():
                if not self._running:
                    break
                if message["type"] != "message":
                    continue
                topic = message["channel"]
                data = message["data"]
                try:
                    event_dict = json.loads(data)
                    event = Event.from_dict(event_dict)
                    if topic in self._handlers:
                        for handler in self._handlers[topic]:
                            try:
                                await handler(event)
                            except Exception as e:
                                logger.error(
                                    f"Error in event handler: {str(e)}",
                                    extra={"topic": topic, "event": str(event)},
                                    exc_info=e
                                )
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode event: {str(e)}", extra={"data": data}, exc_info=e)
                except Exception as e:
                    logger.error(f"Error processing event: {str(e)}", exc_info=e)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in Redis Pub/Sub listener: {str(e)}", exc_info=e)
            await asyncio.sleep(1)
            if self._running:
                self._listen_task = asyncio.create_task(self._listen_for_messages())

class EventBridge:
    """
    Bridge between Redis events and the local event bus.
    
    This class connects the Redis event broker to the local event bus,
    allowing events to be published and subscribed to across process boundaries.
    
    Attributes:
        redis_broker (RedisEventBroker): Redis event broker instance.
        _subscribed_topics (Set[str]): Set of topics the bridge is subscribed to.
    """
    def __init__(self, redis_broker: RedisEventBroker):
        """
        Initialize the event bridge.
        
        Args:
            redis_broker (RedisEventBroker): Redis event broker.
        """
        self.redis_broker = redis_broker
        self._subscribed_topics: Set[str] = set()
        logger.info("Initialized event bridge")

    async def connect(self) -> None:
        """
        Connect the bridge.
        
        Raises:
            Exception: If connection fails.
        """
        await self.redis_broker.connect()
        logger.info("Connected event bridge")

    async def disconnect(self) -> None:
        """
        Disconnect the bridge.
        
        Raises:
            Exception: If disconnection fails.
        """
        await self.redis_broker.disconnect()
        logger.info("Disconnected event bridge")

    async def subscribe_to_topic(self, topic: str) -> None:
        """
        Subscribe to a Redis topic and forward events to the local event bus.
        
        Args:
            topic (str): Topic to subscribe to.
        
        Raises:
            Exception: If subscription fails.
        """
        if topic in self._subscribed_topics:
            return
        await self.redis_broker.subscribe(topic, self._forward_to_event_bus)
        self._subscribed_topics.add(topic)
        logger.info(f"Subscribed to Redis topic: {topic}")

    async def unsubscribe_from_topic(self, topic: str) -> None:
        """
        Unsubscribe from a Redis topic.
        
        Args:
            topic (str): Topic to unsubscribe from.
        
        Raises:
            Exception: If unsubscription fails.
        """
        if topic not in self._subscribed_topics:
            return
        await self.redis_broker.unsubscribe(topic, self._forward_to_event_bus)
        self._subscribed_topics.remove(topic)
        logger.info(f"Unsubscribed from Redis topic: {topic}")

    async def publish_to_topic(self, topic: str, event: Event) -> None:
        """
        Publish an event to a Redis topic.
        
        Args:
            topic (str): Topic to publish to.
            event (Event): Event to publish.
        
        Raises:
            Exception: If publishing fails.
        """
        await self.redis_broker.publish(topic, event)

    async def _forward_to_event_bus(self, event: Event) -> None:
        """
        Forward an event from Redis to the local event bus.
        
        Args:
            event (Event): Event to forward.
        """
        await event_bus.publish(event)
        logger.debug(f"Forwarded event to local event bus: {event}")

redis_event_broker = RedisEventBroker()
event_bridge = EventBridge(redis_event_broker)

async def initialize_event_system() -> None:
    """
    Initialize the event system.
    
    This function connects the Redis event broker and subscribes to common topics.
    """
    try:
        await event_bridge.connect()
        common_topics = [
            "user.events",
            "search.events",
            "analysis.events",
            "task.events",
        ]
        for topic in common_topics:
            await event_bridge.subscribe_to_topic(topic)
        logger.info("Initialized event system")
    except Exception as e:
        logger.error(f"Failed to initialize event system: {str(e)}", exc_info=e)
        raise

async def shutdown_event_system() -> None:
    """
    Shutdown the event system.
    
    This function disconnects the Redis event broker and stops the event bus.
    """
    try:
        await event_bridge.disconnect()
        await event_bus.stop()
        logger.info("Shutdown event system")
    except Exception as e:
        logger.error(f"Error shutting down event system: {str(e)}", exc_info=e)
        raise