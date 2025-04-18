"""
Messaging system initialization for the Medical Research Synthesizer.

This module provides functions for initializing the messaging system,
including setting up exchanges, queues, and bindings.
"""
from ..logging_config import get_logger
from .rabbitmq_broker import get_rabbitmq_broker
from .producer import get_message_producer
from .consumer import get_message_consumer
from .schemas import EventType, TaskType, CommandType

logger = get_logger(__name__)

async def initialize_messaging_system():
    """
    Initialize the messaging system.
    
    This function sets up the RabbitMQ broker, declares exchanges and queues,
    and starts the message consumer.
    
    Raises:
        ConnectionError: If connection fails
        MessageBrokerError: If initialization fails
    """
    try:
        # Get the broker, producer, and consumer
        broker = get_rabbitmq_broker()
        producer = get_message_producer()
        consumer = get_message_consumer()
        
        # Connect to RabbitMQ
        await broker.connect()
        
        # Declare exchanges
        await broker.declare_exchange("events", "topic")
        await broker.declare_exchange("tasks", "topic")
        await broker.declare_exchange("commands", "topic")
        
        # Declare queues for each service
        service_queues = {
            "search_service": [
                # Events
                EventType.SEARCH_PERFORMED,
                EventType.SEARCH_COMPLETED,
                # Tasks
                TaskType.SEARCH_PUBMED,
                TaskType.SEARCH_CLINICAL_TRIALS,
                TaskType.SEARCH_KNOWLEDGE_BASE,
                # Commands
                f"search_service.{CommandType.PERFORM_SEARCH}",
                f"search_service.{CommandType.CANCEL_SEARCH}"
            ],
            "analysis_service": [
                # Events
                EventType.ANALYSIS_STARTED,
                EventType.ANALYSIS_COMPLETED,
                EventType.ANALYSIS_FAILED,
                # Tasks
                TaskType.ANALYZE_CONTRADICTIONS,
                TaskType.ANALYZE_BIAS,
                TaskType.ANALYZE_TRENDS,
                # Commands
                f"analysis_service.{CommandType.START_ANALYSIS}",
                f"analysis_service.{CommandType.CANCEL_ANALYSIS}"
            ],
            "export_service": [
                # Tasks
                TaskType.EXPORT_RESULTS,
                TaskType.EXPORT_ANALYSIS
            ],
            "user_service": [
                # Events
                EventType.USER_CREATED,
                EventType.USER_UPDATED,
                EventType.USER_DELETED,
                # Commands
                f"user_service.{CommandType.CREATE_USER}",
                f"user_service.{CommandType.UPDATE_USER}",
                f"user_service.{CommandType.DELETE_USER}"
            ],
            "system": [
                # Events
                EventType.SYSTEM_STARTUP,
                EventType.SYSTEM_SHUTDOWN,
                EventType.SYSTEM_ERROR,
                # Commands
                f"system.{CommandType.SHUTDOWN}",
                f"system.{CommandType.RESTART}"
            ]
        }
        
        # Declare and bind queues
        for service, routing_keys in service_queues.items():
            for routing_key in routing_keys:
                queue_name = f"asf-medical.{service}.{routing_key}"
                
                # Determine the exchange based on the routing key
                if "." in routing_key and not routing_key.startswith(("task.", "event.")):
                    # Commands have the format "target.command_type"
                    exchange_name = "commands"
                elif routing_key.startswith("task."):
                    exchange_name = "tasks"
                else:
                    exchange_name = "events"
                
                # Declare the queue
                await broker.declare_queue(queue_name)
                
                # Bind the queue to the exchange
                await broker.bind_queue(queue_name, exchange_name, routing_key)
        
        logger.info("Initialized messaging system")
    except Exception as e:
        logger.error(f"Failed to initialize messaging system: {str(e)}", exc_info=e)
        raise


async def shutdown_messaging_system():
    """
    Shutdown the messaging system.
    
    This function stops the message consumer and closes the RabbitMQ connection.
    
    Raises:
        ConnectionError: If disconnection fails
        MessageBrokerError: If shutdown fails
    """
    try:
        # Get the broker and consumer
        broker = get_rabbitmq_broker()
        consumer = get_message_consumer()
        
        # Stop the consumer
        await consumer.stop()
        
        # Close the connection
        await broker.close()
        
        logger.info("Shutdown messaging system")
    except Exception as e:
        logger.error(f"Error shutting down messaging system: {str(e)}", exc_info=e)
        raise
