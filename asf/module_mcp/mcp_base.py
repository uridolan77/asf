# === FILE: mcp_sdk/__init__.py ===

from mcp_sdk.client import MCPClient
from mcp_sdk.config import MCPConfig
from mcp_sdk.models import Message, Context, Prediction
from mcp_sdk.enums import (
    MessageType, 
    ContextLevel, 
    ConfidenceLevel, 
    InteractionStage,
    ModelState
)
from mcp_sdk.errors import MCPError, MCPConnectionError, MCPProtocolError

__version__ = "0.1.0"

__all__ = [
    'MCPClient',
    'MCPConfig',
    'Message',
    'Context',
    'Prediction',
    'MessageType',
    'ContextLevel',
    'ConfidenceLevel',
    'InteractionStage',
    'ModelState',
    'MCPError',
    'MCPConnectionError', 
    'MCPProtocolError'
]


# === FILE: mcp_sdk/enums.py ===

from enum import Enum, auto

class MessageType(Enum):
    """Types of MCP messages."""
    QUERY = auto()                # Request for information
    RESPONSE = auto()             # Response to a query
    PREDICTION = auto()           # Predicted interaction or state
    OBSERVATION = auto()          # Observed data from environment
    UPDATE = auto()               # Update to a model or state
    ACTION = auto()               # Action to be taken
    FEEDBACK = auto()             # Feedback on a prediction or action
    TEST = auto()                 # Active inference test
    COUNTERFACTUAL = auto()       # Counterfactual scenario
    METADATA = auto()             # Metadata about the model
    SYSTEM = auto()               # System-level message
    ERROR = auto()                # Error notification

class ContextLevel(Enum):
    """Context levels for MCP messages."""
    NONE = 0           # No context provided
    MINIMAL = 1        # Minimal context (e.g., ID only)
    BASIC = 2          # Basic context (e.g., ID, type, timestamp)
    STANDARD = 3       # Standard context (basic + history)
    ENHANCED = 4       # Enhanced context (standard + related entities)
    COMPLETE = 5       # Complete context (all available information)

class ConfidenceLevel(Enum):
    """Confidence levels for predictions and assertions."""
    SPECULATIVE = 0.2      # Highly uncertain, speculative
    LOW = 0.4              # Low confidence
    MODERATE = 0.6         # Moderate confidence
    HIGH = 0.8             # High confidence
    CERTAIN = 1.0          # Very high confidence

class InteractionStage(Enum):
    """Stages of interaction in the MCP lifecycle."""
    INITIALIZATION = auto()    # Initial connection
    PREDICTION = auto()        # Generating predictions
    OBSERVATION = auto()       # Receiving observations
    EVALUATION = auto()        # Comparing predictions to observations
    UPDATE = auto()            # Updating model based on evaluation
    ACTIVE_INFERENCE = auto()  # Actively testing predictions
    COUNTERFACTUAL = auto()    # Exploring counterfactual scenarios
    TERMINATION = auto()       # Ending the interaction

class ModelState(Enum):
    """States of the model in the MCP protocol."""
    INITIALIZING = auto()      # Model is initializing
    READY = auto()             # Model is ready for interaction
    PREDICTING = auto()        # Model is generating predictions
    UPDATING = auto()          # Model is updating its parameters
    LEARNING = auto()          # Model is learning from feedback
    TESTING = auto()           # Model is conducting tests
    ERROR = auto()             # Model is in an error state
    INACTIVE = auto()          # Model is inactive


# === FILE: mcp_sdk/models.py ===

import time
import uuid
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from mcp_sdk.enums import (
    MessageType,
    ContextLevel, 
    ConfidenceLevel, 
    InteractionStage,
    ModelState
)

@dataclass
class Context:
    """
    Context information for MCP messages.
    Provides rich contextual information for interpretation and processing.
    """
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    level: ContextLevel = ContextLevel.STANDARD
    timestamp: float = field(default_factory=time.time)
    entity_id: Optional[str] = None
    environmental_id: Optional[str] = None
    conversation_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    interaction_stage: Optional[InteractionStage] = None
    model_state: Optional[ModelState] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    related_entities: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        result = asdict(self)
        # Convert enums to serializable values
        if self.level:
            result['level'] = self.level.name
        if self.interaction_stage:
            result['interaction_stage'] = self.interaction_stage.name
        if self.model_state:
            result['model_state'] = self.model_state.name
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Context':
        """Create context from dictionary."""
        # Convert serialized enum values back to enums
        if 'level' in data and isinstance(data['level'], str):
            data['level'] = ContextLevel[data['level']]
        if 'interaction_stage' in data and isinstance(data['interaction_stage'], str):
            data['interaction_stage'] = InteractionStage[data['interaction_stage']]
        if 'model_state' in data and isinstance(data['model_state'], str):
            data['model_state'] = ModelState[data['model_state']]
        
        # Create instance with valid fields
        valid_fields = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**valid_fields)
    
    def add_history_entry(self, entry_type: str, entry_data: Dict[str, Any]) -> None:
        """Add an entry to the context history."""
        self.history.append({
            'timestamp': time.time(),
            'type': entry_type,
            'data': entry_data
        })
        
        # Limit history size
        if len(self.history) > 100:
            self.history = self.history[-100:]

@dataclass
class Prediction:
    """
    Prediction information for MCP messages.
    Encapsulates predictions with confidence and metadata.
    """
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    predicted_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    confidence_level: ConfidenceLevel = ConfidenceLevel.MODERATE
    precision: float = 1.0
    creation_time: float = field(default_factory=time.time)
    expiration_time: Optional[float] = None
    prediction_horizon: Optional[float] = None  # Time window for prediction
    alternative_predictions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to dictionary."""
        result = asdict(self)
        # Convert enums to serializable values
        if self.confidence_level:
            result['confidence_level'] = self.confidence_level.name
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Prediction':
        """Create prediction from dictionary."""
        # Convert serialized enum values back to enums
        if 'confidence_level' in data and isinstance(data['confidence_level'], str):
            data['confidence_level'] = ConfidenceLevel[data['confidence_level']]
        
        # Create instance with valid fields
        valid_fields = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**valid_fields)
    
    def is_expired(self) -> bool:
        """Check if the prediction has expired."""
        if self.expiration_time is None:
            return False
        return time.time() > self.expiration_time
    
    def add_alternative(self, prediction_data: Dict[str, Any], confidence: float) -> None:
        """Add an alternative prediction."""
        self.alternative_predictions.append({
            'predicted_data': prediction_data,
            'confidence': confidence,
            'timestamp': time.time()
        })

@dataclass
class Message:
    """
    Main message class for Model Context Protocol.
    Provides a standardized structure for all MCP communications.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.SYSTEM
    content: Dict[str, Any] = field(default_factory=dict)
    context: Context = field(default_factory=Context)
    prediction: Optional[Prediction] = None
    sender_id: Optional[str] = None
    recipient_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    reply_to: Optional[str] = None
    correlation_id: Optional[str] = None
    ttl: Optional[int] = None  # Time-to-live in seconds
    priority: float = 0.5  # 0.0 to 1.0, higher is higher priority
    is_encrypted: bool = False
    encryption_info: Optional[Dict[str, Any]] = None
    signature: Optional[str] = None
    trace_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        result = asdict(self)
        # Convert enums and complex types to serializable values
        if self.message_type:
            result['message_type'] = self.message_type.name
        if self.context:
            result['context'] = self.context.to_dict()
        if self.prediction:
            result['prediction'] = self.prediction.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        # Convert serialized enum values back to enums
        if 'message_type' in data and isinstance(data['message_type'], str):
            data['message_type'] = MessageType[data['message_type']]
        
        # Convert complex types
        if 'context' in data and isinstance(data['context'], dict):
            data['context'] = Context.from_dict(data['context'])
        if 'prediction' in data and isinstance(data['prediction'], dict):
            data['prediction'] = Prediction.from_dict(data['prediction'])
        
        # Create instance with valid fields
        valid_fields = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**valid_fields)
    
    def serialize(self) -> str:
        """Serialize message to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def deserialize(cls, data: str) -> 'Message':
        """Create message from JSON string."""
        return cls.from_dict(json.loads(data))
    
    def is_expired(self) -> bool:
        """Check if the message has expired based on TTL."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl
    
    def create_reply(self, content: Dict[str, Any], message_type: Optional[MessageType] = None) -> 'Message':
        """Create a reply to this message."""
        # Use the same message type for the reply if not specified
        if message_type is None:
            if self.message_type == MessageType.QUERY:
                message_type = MessageType.RESPONSE
            else:
                message_type = self.message_type
                
        # Create a new context based on the current one
        new_context = Context(
            entity_id=self.context.environmental_id,  # Swap IDs for reply
            environmental_id=self.context.entity_id,
            conversation_id=self.context.conversation_id,
            parent_message_id=self.id
        )
        
        # Add history entry
        if self.context.history:
            new_context.history = self.context.history.copy()
        new_context.add_history_entry('reply_to', {'message_id': self.id, 'message_type': self.message_type.name})
        
        return Message(
            message_type=message_type,
            content=content,
            context=new_context,
            sender_id=self.recipient_id,
            recipient_id=self.sender_id,
            reply_to=self.id,
            correlation_id=self.correlation_id or self.id,
            trace_id=self.trace_id
        )


# === FILE: mcp_sdk/errors.py ===

class MCPError(Exception):
    """Base exception for all MCP-related errors."""
    pass

class MCPConnectionError(MCPError):
    """Exception raised for connection errors."""
    pass

class MCPProtocolError(MCPError):
    """Exception raised for protocol errors."""
    pass

class MCPTimeoutError(MCPError):
    """Exception raised for timeout errors."""
    pass

class MCPValidationError(MCPError):
    """Exception raised for message validation errors."""
    pass

class MCPEncryptionError(MCPError):
    """Exception raised for encryption-related errors."""
    pass


# === FILE: mcp_sdk/config.py ===

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class MCPConfig:
    """Configuration for MCP client."""
    entity_id: str
    default_timeout: float = 10.0
    max_retries: int = 3
    retry_delay: float = 0.5
    enable_encryption: bool = False
    encryption_key: Optional[str] = None
    enable_compression: bool = False
    compression_level: int = 6
    max_batch_size: int = 100
    log_level: str = "INFO"
    additional_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "entity_id": self.entity_id,
            "default_timeout": self.default_timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "enable_encryption": self.enable_encryption,
            "encryption_key": self.encryption_key,
            "enable_compression": self.enable_compression,
            "compression_level": self.compression_level,
            "max_batch_size": self.max_batch_size,
            "log_level": self.log_level,
            "additional_settings": self.additional_settings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPConfig':
        """Create config from dictionary."""
        return cls(**data)


# === FILE: mcp_sdk/client.py ===

import asyncio
import logging
import time
import uuid
import json
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor

from mcp_sdk.models import Message, Context, Prediction
from mcp_sdk.enums import (
    MessageType, 
    ContextLevel, 
    ConfidenceLevel, 
    InteractionStage,
    ModelState
)
from mcp_sdk.config import MCPConfig
from mcp_sdk.errors import (
    MCPError, 
    MCPConnectionError, 
    MCPProtocolError,
    MCPTimeoutError,
    MCPValidationError
)

class MCPClient:
    """
    Client for interacting with the Model Context Protocol.
    Provides methods for sending and receiving MCP messages.
    """
    def __init__(self, config: MCPConfig):
        """
        Initialize the MCP client.
        
        Args:
            config: Configuration for the client
        """
        self.config = config
        self.entity_id = config.entity_id
        self.message_handlers: Dict[MessageType, List[Callable[[Message], Awaitable[Optional[Message]]]]] = {
            msg_type: [] for msg_type in MessageType
        }
        self.conversations: Dict[str, List[Message]] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.logger = logging.getLogger("mcp_sdk.client")
        
        # Set log level
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # Background task for message processing
        self._processing_task = None
        self._running = False
        self._message_queue = asyncio.Queue()
        
        # Thread pool for blocking operations
        self._thread_pool = ThreadPoolExecutor(max_workers=5)
        
        self.logger.info(f"MCP Client initialized for entity {self.entity_id}")
    
    async def start(self) -> None:
        """Start the client and background processing."""
        if self._running:
            return
            
        self._running = True
        self._processing_task = asyncio.create_task(self._process_messages())
        self.logger.info("MCP Client started")
    
    async def stop(self) -> None:
        """Stop the client and background processing."""
        if not self._running:
            return
            
        self._running = False
        
        # Cancel all pending responses
        for future in self.pending_responses.values():
            if not future.done():
                future.cancel()
                
        # Wait for processing task to complete
        if self._processing_task:
            try:
                self._processing_task.cancel()
                await self._processing_task
            except asyncio.CancelledError:
                pass
            
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)
        
        self.logger.info("MCP Client stopped")
    
    async def create_message(self, 
                         message_type: MessageType,
                         content: Dict[str, Any],
                         recipient_id: Optional[str] = None,
                         context_level: ContextLevel = ContextLevel.STANDARD,
                         reply_to: Optional[str] = None,
                         correlation_id: Optional[str] = None,
                         prediction_data: Optional[Dict[str, Any]] = None,
                         confidence: float = 0.5,
                         priority: float = 0.5,
                         ttl: Optional[int] = None,
                         **kwargs) -> Message:
        """
        Create a new MCP message.
        
        Args:
            message_type: Type of message to create
            content: Message content
            recipient_id: Optional recipient ID
            context_level: Level of context to include
            reply_to: Optional ID of message this is replying to
            correlation_id: Optional correlation ID for tracking conversations
            prediction_data: Optional prediction data
            confidence: Confidence for prediction (0.0 to 1.0)
            priority: Message priority (0.0 to 1.0)
            ttl: Time-to-live in seconds
            **kwargs: Additional message parameters
            
        Returns:
            The created MCP message
        """
        # Create context
        context = Context(
            entity_id=self.entity_id,
            environmental_id=recipient_id,
            level=context_level
        )
        
        # If replying to a message, include conversation info
        if reply_to:
            # Find the original message
            original_msg = None
            for conv_msgs in self.conversations.values():
                for msg in conv_msgs:
                    if msg.id == reply_to:
                        original_msg = msg
                        break
                if original_msg:
                    break
                    
            if original_msg:
                context.conversation_id = original_msg.context.conversation_id
                context.parent_message_id = original_msg.id
                
                # Copy history
                if original_msg.context.history:
                    context.history = original_msg.context.history.copy()
                    
                # Add reply entry
                context.add_history_entry('reply_to', {
                    'message_id': original_msg.id,
                    'message_type': original_msg.message_type.name
                })
        
        # Create prediction if data provided
        prediction = None
        if prediction_data:
            confidence_level = self._confidence_to_level(confidence)
            prediction = Prediction(
                predicted_data=prediction_data,
                confidence=confidence,
                confidence_level=confidence_level
            )
        
        # Create the message
        message = Message(
            message_type=message_type,
            content=content,
            context=context,
            prediction=prediction,
            sender_id=self.entity_id,
            recipient_id=recipient_id,
            reply_to=reply_to,
            correlation_id=correlation_id,
            ttl=ttl,
            priority=priority,
            **kwargs
        )
        
        # Add to conversation history
        self._add_to_conversation(message)
        
        return message
    
    async def send_message(self, message: Message, wait_for_response: bool = False, 
                        timeout: Optional[float] = None) -> Optional[Message]:
        """
        Send an MCP message.
        
        Args:
            message: The message to send
            wait_for_response: Whether to wait for a response
            timeout: Timeout in seconds (None for default)
            
        Returns:
            Response message if wait_for_response is True, else None
        """
        # Add to conversation history
        self._add_to_conversation(message)
        
        # Process through registered handlers
        await self._message_queue.put(message)
        
        # If not waiting for response, return immediately
        if not wait_for_response:
            return None
            
        # Wait for response with timeout
        timeout = timeout or self.config.default_timeout
        
        # Create a future for the response
        response_future = asyncio.Future()
        self.pending_responses[message.id] = response_future
        
        try:
            # Wait for response with timeout
            return await asyncio.wait_for(response_future, timeout)
        except asyncio.TimeoutError:
            # Remove the pending response
            if message.id in self.pending_responses:
                del self.pending_responses[message.id]
                
            raise MCPTimeoutError(f"Timeout waiting for response to message {message.id}")
        finally:
            # Clean up in case of other errors
            if message.id in self.pending_responses:
                del self.pending_responses[message.id]
    
    async def register_handler(self, message_type: MessageType, 
                           handler: Callable[[Message], Awaitable[Optional[Message]]]) -> None:
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: The message type to handle
            handler: The handler function (async)
        """
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
            
        self.message_handlers[message_type].append(handler)
        self.logger.debug(f"Registered handler for message type {message_type.name}")
    
    async def process_message(self, message: Union[str, Dict[str, Any], Message]) -> Optional[Message]:
        """
        Process an incoming message through the registered handlers.
        
        Args:
            message: The message to process (string, dict, or Message)
            
        Returns:
            Response message if any handler returns one
        """
        # Convert to Message object if needed
        if isinstance(message, str):
            try:
                message = Message.deserialize(message)
            except Exception as e:
                raise MCPProtocolError(f"Error deserializing message: {str(e)}")
        elif isinstance(message, dict):
            try:
                message = Message.from_dict(message)
            except Exception as e:
                raise MCPProtocolError(f"Error converting dictionary to message: {str(e)}")
        elif not isinstance(message, Message):
            raise MCPValidationError(f"Expected Message object, got {type(message)}")
            
        # Check if expired
        if message.is_expired():
            self.logger.warning(f"Received expired message: {message.id}")
            return None
            
        # Add to conversation history
        self._add_to_conversation(message)
        
        # Check if this is a response to a pending message
        if message.reply_to in self.pending_responses:
            response_future = self.pending_responses[message.reply_to]
            if not response_future.done():
                response_future.set_result(message)
                return None  # No need to process further
        
        # Process through registered handlers
        await self._message_queue.put(message)
        
        # This doesn't wait for the result, handlers will process asynchronously
        return None
    
    async def query(self, recipient_id: str, query_content: Dict[str, Any], 
                 context_level: ContextLevel = ContextLevel.STANDARD,
                 timeout: Optional[float] = None) -> Message:
        """
        Send a query and wait for response.
        
        Args:
            recipient_id: ID of the recipient
            query_content: Query content
            context_level: Level of context to include
            timeout: Timeout in seconds (None for default)
            
        Returns:
            Response message
        """
        message = await self.create_message(
            message_type=MessageType.QUERY,
            content=query_content,
            recipient_id=recipient_id,
            context_level=context_level
        )
        
        response = await self.send_message(message, wait_for_response=True, timeout=timeout)
        return response
    
    async def send_prediction(self, recipient_id: str, prediction_data: Dict[str, Any],
                           confidence: float = 0.7,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send a prediction.
        
        Args:
            recipient_id: ID of the recipient
            prediction_data: The prediction data
            confidence: Confidence in the prediction (0.0 to 1.0)
            metadata: Optional metadata
            
        Returns:
            Result of sending the prediction
        """
        message = await self.create_message(
            message_type=MessageType.PREDICTION,
            content=metadata or {},
            recipient_id=recipient_id,
            prediction_data=prediction_data,
            confidence=confidence
        )
        
        await self.send_message(message)
        return {
            'success': True,
            'message_id': message.id,
            'prediction_id': message.prediction.prediction_id if message.prediction else None
        }
    
    async def send_observation(self, recipient_id: str, observation_data: Dict[str, Any],
                            metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send an observation.
        
        Args:
            recipient_id: ID of the recipient
            observation_data: The observation data
            metadata: Optional metadata
            
        Returns:
            Result of sending the observation
        """
        message = await self.create_message(
            message_type=MessageType.OBSERVATION,
            content=observation_data,
            recipient_id=recipient_id,
            context_level=ContextLevel.ENHANCED,
            metadata=metadata or {}
        )
        
        await self.send_message(message)
        return {
            'success': True,
            'message_id': message.id
        }
    
    async def send_action(self, recipient_id: str, action_data: Dict[str, Any],
                       wait_for_result: bool = True,
                       timeout: Optional[float] = None) -> Optional[Message]:
        """
        Send an action and optionally wait for result.
        
        Args:
            recipient_id: ID of the recipient
            action_data: The action data
            wait_for_result: Whether to wait for result
            timeout: Timeout in seconds (None for default)
            
        Returns:
            Result message if wait_for_result is True, else None
        """
        message = await self.create_message(
            message_type=MessageType.ACTION,
            content=action_data,
            recipient_id=recipient_id,
            context_level=ContextLevel.STANDARD
        )
        
        if wait_for_result:
            return await self.send_message(message, wait_for_response=True, timeout=timeout)
        else:
            await self.send_message(message)
            return None
    
    async def send_test(self, recipient_id: str, test_parameters: Dict[str, Any],
                     expected_results: Dict[str, Any],
                     test_type: str = 'active_inference',
                     confidence: float = 0.7,
                     wait_for_result: bool = True,
                     timeout: Optional[float] = None) -> Optional[Message]:
        """
        Send a test (e.g., active inference).
        
        Args:
            recipient_id: ID of the recipient
            test_parameters: The test parameters
            expected_results: Expected results of the test
            test_type: Type of test
            confidence: Confidence in expected results (0.0 to 1.0)
            wait_for_result: Whether to wait for result
            timeout: Timeout in seconds (None for default)
            
        Returns:
            Result message if wait_for_result is True, else None
        """
        # Create context with test metadata
        context = Context(
            entity_id=self.entity_id,
            environmental_id=recipient_id,
            level=ContextLevel.ENHANCED,
            metadata={
                'test_type': test_type,
                'test_id': str(uuid.uuid4())
            }
        )
        
        # Set interaction stage for active inference
        if test_type == 'active_inference':
            context.interaction_stage = InteractionStage.ACTIVE_INFERENCE
        
        # Create prediction with expected results
        prediction = Prediction(
            predicted_data=expected_results,
            confidence=confidence,
            confidence_level=self._confidence_to_level(confidence)
        )
        
        # Create message
        message = Message(
            message_type=MessageType.TEST,
            content=test_parameters,
            context=context,
            prediction=prediction,
            sender_id=self.entity_id,
            recipient_id=recipient_id
        )
        
        # Add to conversation history
        self._add_to_conversation(message)
        
        if wait_for_result:
            return await self.send_message(message, wait_for_response=True, timeout=timeout)
        else:
            await self.send_message(message)
            return None
    
    async def send_counterfactual(self, recipient_id: str, 
                               variation_parameters: Dict[str, Any],
                               expected_outcomes: Dict[str, Any],
                               confidence: float = 0.6,
                               wait_for_result: bool = True,
                               timeout: Optional[float] = None) -> Optional[Message]:
        """
        Send a counterfactual simulation.
        
        Args:
            recipient_id: ID of the recipient
            variation_parameters: The variation parameters
            expected_outcomes: Expected outcomes of the simulation
            confidence: Confidence in expected outcomes (0.0 to 1.0)
            wait_for_result: Whether to wait for result
            timeout: Timeout in seconds (None for default)
            
        Returns:
            Result message if wait_for_result is True, else None
        """
        # Create context with counterfactual metadata
        context = Context(
            entity_id=self.entity_id,
            environmental_id=recipient_id,
            level=ContextLevel.ENHANCED,
            interaction_stage=InteractionStage.COUNTERFACTUAL,
            metadata={
                'simulation_id': str(uuid.uuid4()),
                'variation_type': variation_parameters.get('variation_type', 'unknown')
            }
        )
        
        # Create prediction with expected outcomes
        prediction = Prediction(
            predicted_data=expected_outcomes,
            confidence=confidence,
            confidence_level=self._confidence_to_level(confidence)
        )
        
        # Create message
        message = Message(
            message_type=MessageType.COUNTERFACTUAL,
            content=variation_parameters,
            context=context,
            prediction=prediction,
            sender_id=self.entity_id,
            recipient_id=recipient_id
        )
        
        # Add to conversation history
        self._add_to_conversation(message)
        
        if wait_for_result:
            return await self.send_message(message, wait_for_response=True, timeout=timeout)
        else:
            await self.send_message(message)
            return None
    
    async def send_feedback(self, recipient_id: str, prediction_id: str, 
                         feedback_value: float,
                         feedback_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send feedback on a prediction.
        
        Args:
            recipient_id: ID of the recipient
            prediction_id: ID of the prediction to provide feedback on
            feedback_value: Feedback value (-1.0 to 1.0, where positive is good)
            feedback_data: Optional additional feedback data
            
        Returns:
            Result of sending the feedback
        """
        feedback_content = {
            'prediction_id': prediction_id,
            'feedback_value': feedback_value,
            **(feedback_data or {})
        }
        
        message = await self.create_message(
            message_type=MessageType.FEEDBACK,
            content=feedback_content,
            recipient_id=recipient_id
        )
        
        await self.send_message(message)
        return {
            'success': True,
            'message_id': message.id,
            'prediction_id': prediction_id
        }
    
    async def get_conversation_history(self, conversation_id: str) -> List[Message]:
        """
        Get the history of a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            List of messages in the conversation
        """
        return self.conversations.get(conversation_id, [])
    
    async def _process_messages(self) -> None:
        """Background task for processing messages."""
        self.logger.info("Message processing task started")
        
        while self._running:
            try:
                # Get the next message from the queue
                message = await self._message_queue.get()
                
                # Process through registered handlers
                handlers = self.message_handlers.get(message.message_type, [])
                
                if not handlers:
                    self.logger.debug(f"No handlers registered for message type {message.message_type.name}")
                    self._message_queue.task_done()
                    continue
                
                # Process through each handler
                for handler in handlers:
                    try:
                        response = await handler(message)
                        
                        # If handler returns a response, send it
                        if response:
                            # Add to conversation history
                            self._add_to_conversation(response)
                            
                            # If response is to a pending message, resolve the future
                            if response.reply_to in self.pending_responses:
                                future = self.pending_responses[response.reply_to]
                                if not future.done():
                                    future.set_result(response)
                            
                            # Put back in queue for processing by other components
                            await self._message_queue.put(response)
                            break  # Stop processing after first handler returns a response
                    except Exception as e:
                        self.logger.error(f"Error in message handler for {message.message_type.name}: {str(e)}")
                
                # Mark task as done
                self._message_queue.task_done()
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {str(e)}")
        
        self.logger.info("Message processing task stopped")
    
    def _add_to_conversation(self, message: Message) -> None:
        """Add a message to its conversation history."""
        # Determine conversation ID
        conversation_id = message.context.conversation_id
        
        # If no conversation ID, use message ID as conversation ID
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            message.context.conversation_id = conversation_id
            
        # Add to conversation
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
            
        self.conversations[conversation_id].append(message)
        
        # Limit conversation history
        if len(self.conversations[conversation_id]) > 100:
            self.conversations[conversation_id] = self.conversations[conversation_id][-100:]
    
    def _confidence_to_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence value to confidence level."""
        if confidence < 0.3:
            return ConfidenceLevel.SPECULATIVE
        elif confidence < 0.5:
            return ConfidenceLevel.LOW
        elif confidence < 0.7:
            return ConfidenceLevel.MODERATE
        elif confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.CERTAIN


# === FILE: mcp_sdk/utils.py ===

import json
import time
import uuid
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor

from mcp_sdk.models import Message, Context, Prediction
from mcp_sdk.enums import (
    MessageType, 
    ContextLevel, 
    ConfidenceLevel, 
    InteractionStage,
    ModelState
)
from mcp_sdk.errors import MCPError, MCPValidationError

logger = logging.getLogger("mcp_sdk.utils")

def validate_message(message: Message) -> bool:
    """
    Validate an MCP message.
    
    Args:
        message: The message to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Check if expired
    if message.is_expired():
        logger.warning(f"Message {message.id} has expired")
        return False
        
    # Check required fields
    if not message.message_type:
        logger.warning(f"Message {message.id} is missing message_type")
        return False
        
    if not message.context:
        logger.warning(f"Message {message.id} is missing context")
        return False
        
    # Check message type-specific requirements
    if message.message_type == MessageType.QUERY and not message.content:
        logger.warning(f"Query message {message.id} is missing content")
        return False
        
    if message.message_type == MessageType.PREDICTION and not message.prediction:
        logger.warning(f"Prediction message {message.id} is missing prediction")
        return False
        
    return True

def confidence_to_level(confidence: float) -> ConfidenceLevel:
    """
    Convert a confidence value to a confidence level.
    
    Args:
        confidence: Confidence value (0.0 to 1.0)
        
    Returns:
        Corresponding confidence level
    """
    if confidence < 0.3:
        return ConfidenceLevel.SPECULATIVE
    elif confidence < 0.5:
        return ConfidenceLevel.LOW
    elif confidence < 0.7:
        return ConfidenceLevel.MODERATE
    elif confidence < 0.9:
        return ConfidenceLevel.HIGH
    else:
        return ConfidenceLevel.CERTAIN

def generate_correlation_id() -> str:
    """
    Generate a unique correlation ID.
    
    Returns:
        Unique correlation ID
    """
    return f"corr_{uuid.uuid4()}"

async def wait_for_responses(client, message_ids: List[str], timeout: float = 10.0) -> Dict[str, Optional[Message]]:
    """
    Wait for responses to multiple messages.
    
    Args:
        client: The MCP client
        message_ids: List of message IDs to wait for
        timeout: Timeout in seconds
        
    Returns:
        Dictionary mapping message IDs to responses (or None if no response)
    """
    # Create futures for each message
    futures = {}
    for msg_id in message_ids:
        if msg_id not in client.pending_responses:
            client.pending_responses[msg_id] = asyncio.Future()
        futures[msg_id] = client.pending_responses[msg_id]
        
    # Wait for all futures with timeout
    results = {}
    try:
        done, pending = await asyncio.wait(
            [futures[msg_id] for msg_id in message_ids],
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED
        )
        
        # Process results
        for msg_id in message_ids:
            future = futures[msg_id]
            if future in done and not future.cancelled():
                try:
                    results[msg_id] = future.result()
                except Exception:
                    results[msg_id] = None
            else:
                results[msg_id] = None
                
        # Cancel any pending futures
        for future in pending:
            future.cancel()
            
    except asyncio.TimeoutError:
        # Timeout, mark all remaining as None
        for msg_id in message_ids:
            if msg_id not in results:
                results[msg_id] = None
                
    # Clean up
    for msg_id in message_ids:
        if msg_id in client.pending_responses:
            del client.pending_responses[msg_id]
            
    return results

def create_ping_message(entity_id: str, recipient_id: Optional[str] = None) -> Message:
    """
    Create a ping message.
    
    Args:
        entity_id: ID of the sending entity
        recipient_id: Optional recipient ID
        
    Returns:
        Ping message
    """
    context = Context(
        entity_id=entity_id,
        environmental_id=recipient_id,
        level=ContextLevel.MINIMAL
    )
    
    return Message(
        message_type=MessageType.SYSTEM,
        content={
            'command': 'ping',
            'timestamp': time.time()
        },
        context=context,
        sender_id=entity_id,
        recipient_id=recipient_id
    )

def create_status_message(entity_id: str, recipient_id: Optional[str] = None) -> Message:
    """
    Create a status request message.
    
    Args:
        entity_id: ID of the sending entity
        recipient_id: Optional recipient ID
        
    Returns:
        Status request message
    """
    context = Context(
        entity_id=entity_id,
        environmental_id=recipient_id,
        level=ContextLevel.MINIMAL
    )
    
    return Message(
        message_type=MessageType.SYSTEM,
        content={
            'command': 'status',
            'timestamp': time.time()
        },
        context=context,
        sender_id=entity_id,
        recipient_id=recipient_id
    )