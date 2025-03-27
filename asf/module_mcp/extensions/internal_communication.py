# === FILE: mcp_sdk/extensions/internal/__init__.py ===

from mcp_sdk.extensions.internal.client import InternalMCPClient
from mcp_sdk.extensions.internal.models import (
    LayerMessage, 
    LayerContext, 
    LayerRoutingInfo
)
from mcp_sdk.extensions.internal.enums import (
    LayerType, 
    LayerPriority,
    LayerProcessingMode,
    LayerCommunicationType
)

__all__ = [
    'InternalMCPClient',
    'LayerMessage',
    'LayerContext',
    'LayerRoutingInfo',
    'LayerType',
    'LayerPriority',
    'LayerProcessingMode',
    'LayerCommunicationType'
]


# === FILE: mcp_sdk/extensions/internal/enums.py ===

from enum import Enum, auto
from typing import Dict, Any

class LayerType(Enum):
    """Types of ASF layers."""
    UNKNOWN = 0
    LAYER1_SENSORY = 1
    LAYER2_PATTERN = 2
    LAYER3_CONCEPTUAL = 3
    LAYER4_ENVIRONMENTAL = 4
    LAYER5_AUTOPOIETIC = 5
    LAYER6_DISTRIBUTION = 6
    LAYER7_MEMORY = 7
    LAYER8_INTROSPECTION = 8
    LAYER9_ORCHESTRATION = 9
    
    @classmethod
    def from_name(cls, name: str) -> 'LayerType':
        """Get layer type from name."""
        name = name.upper()
        if name.startswith("LAYER"):
            # Try to parse from number
            try:
                num = int(name[5:].split("_")[0])
                return next((l for l in cls if l.value == num), cls.UNKNOWN)
            except:
                pass
                
        # Try to match by name
        for layer in cls:
            if layer.name.endswith(name):
                return layer
                
        return cls.UNKNOWN

class LayerPriority(Enum):
    """Priority levels for inter-layer communication."""
    BACKGROUND = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    SYSTEM = 5

class LayerProcessingMode(Enum):
    """Processing modes for inter-layer messages."""
    ASYNC = auto()        # Process asynchronously
    SYNC = auto()         # Process synchronously (blocking)
    BATCH = auto()        # Batch with similar messages
    THROTTLED = auto()    # Rate-limited processing
    GUARANTEED = auto()   # Guaranteed delivery (with persistence)

class LayerCommunicationType(Enum):
    """Types of communication between layers."""
    REQUEST = auto()          # Request data/action from another layer
    RESPONSE = auto()         # Response to a request
    NOTIFICATION = auto()     # One-way notification
    BROADCAST = auto()        # Broadcast to multiple/all layers
    STATE_UPDATE = auto()     # Update about layer state
    COORDINATION = auto()     # Coordination message
    FEEDBACK = auto()         # Feedback about processing
    ERROR = auto()            # Error notification
    HEALTH_CHECK = auto()     # Health/status check


# === FILE: mcp_sdk/extensions/internal/models.py ===

import time
import uuid
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field, asdict

from mcp_sdk.models import Message, Context, Prediction
from mcp_sdk.enums import MessageType, ContextLevel

from mcp_sdk.extensions.internal.enums import (
    LayerType,
    LayerPriority,
    LayerProcessingMode,
    LayerCommunicationType
)

@dataclass
class LayerRoutingInfo:
    """Routing information for inter-layer communication."""
    source_layer: LayerType
    target_layer: Optional[LayerType] = None
    targets: Set[LayerType] = field(default_factory=set)  # For multi-targeting
    hop_count: int = 0  # Number of layer hops
    max_hops: int = 3  # Max allowed layer traversals
    broadcast: bool = False  # Whether to broadcast to all layers
    exclude_layers: Set[LayerType] = field(default_factory=set)  # Exclude from broadcast
    routing_path: List[LayerType] = field(default_factory=list)  # Path of traversed layers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'source_layer': self.source_layer.name,
            'hop_count': self.hop_count,
            'max_hops': self.max_hops,
            'broadcast': self.broadcast
        }
        
        if self.target_layer:
            result['target_layer'] = self.target_layer.name
            
        if self.targets:
            result['targets'] = [layer.name for layer in self.targets]
            
        if self.exclude_layers:
            result['exclude_layers'] = [layer.name for layer in self.exclude_layers]
            
        if self.routing_path:
            result['routing_path'] = [layer.name for layer in self.routing_path]
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerRoutingInfo':
        """Create from dictionary."""
        routing_info = cls(source_layer=LayerType.UNKNOWN)
        
        if 'source_layer' in data:
            if isinstance(data['source_layer'], str):
                routing_info.source_layer = LayerType[data['source_layer']]
            else:
                routing_info.source_layer = data['source_layer']
        
        if 'target_layer' in data:
            if isinstance(data['target_layer'], str):
                routing_info.target_layer = LayerType[data['target_layer']]
            else:
                routing_info.target_layer = data['target_layer']
        
        if 'targets' in data:
            if isinstance(data['targets'], list):
                routing_info.targets = {
                    LayerType[target] if isinstance(target, str) else target
                    for target in data['targets']
                }
        
        if 'exclude_layers' in data:
            if isinstance(data['exclude_layers'], list):
                routing_info.exclude_layers = {
                    LayerType[layer] if isinstance(layer, str) else layer
                    for layer in data['exclude_layers']
                }
        
        if 'routing_path' in data:
            if isinstance(data['routing_path'], list):
                routing_info.routing_path = [
                    LayerType[layer] if isinstance(layer, str) else layer
                    for layer in data['routing_path']
                ]
        
        if 'hop_count' in data:
            routing_info.hop_count = data['hop_count']
            
        if 'max_hops' in data:
            routing_info.max_hops = data['max_hops']
            
        if 'broadcast' in data:
            routing_info.broadcast = data['broadcast']
            
        return routing_info
    
    def add_hop(self, layer: LayerType) -> bool:
        """
        Add a layer hop to the routing path.
        
        Returns:
            True if hop was added, False if max hops exceeded
        """
        if self.hop_count >= self.max_hops:
            return False
            
        self.routing_path.append(layer)
        self.hop_count += 1
        return True
    
    def should_process(self, current_layer: LayerType) -> bool:
        """
        Check if the current layer should process this message.
        
        Args:
            current_layer: The layer checking if it should process
            
        Returns:
            True if the layer should process the message
        """
        # Don't process if we've already processed it
        if current_layer in self.routing_path:
            return False
            
        # If broadcast, process unless excluded
        if self.broadcast:
            return current_layer not in self.exclude_layers
            
        # If targeted to specific layers
        if self.targets:
            return current_layer in self.targets
            
        # If targeted to a single layer
        if self.target_layer:
            return current_layer == self.target_layer
            
        # Default: don't process
        return False

@dataclass
class LayerContext(Context):
    """Extended context for inter-layer communication."""
    layer_routing: LayerRoutingInfo = field(default_factory=lambda: LayerRoutingInfo(source_layer=LayerType.UNKNOWN))
    priority: LayerPriority = LayerPriority.NORMAL
    processing_mode: LayerProcessingMode = LayerProcessingMode.ASYNC
    communication_type: LayerCommunicationType = LayerCommunicationType.NOTIFICATION
    require_ack: bool = False
    idempotency_key: Optional[str] = None
    time_budget_ms: Optional[int] = None
    processing_deadline: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)  # IDs of messages this depends on
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        # Start with base context
        result = super().to_dict()
        
        # Add layer-specific fields
        result['layer_routing'] = self.layer_routing.to_dict() 
        result['priority'] = self.priority.name
        result['processing_mode'] = self.processing_mode.name
        result['communication_type'] = self.communication_type.name
        result['require_ack'] = self.require_ack
        
        if self.idempotency_key:
            result['idempotency_key'] = self.idempotency_key
            
        if self.time_budget_ms is not None:
            result['time_budget_ms'] = self.time_budget_ms
            
        if self.processing_deadline is not None:
            result['processing_deadline'] = self.processing_deadline
            
        if self.dependencies:
            result['dependencies'] = self.dependencies
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerContext':
        """Create context from dictionary."""
        # Convert base context fields
        context = super().from_dict(data)
        
        # Convert layer-specific fields
        if 'layer_routing' in data:
            context.layer_routing = LayerRoutingInfo.from_dict(data['layer_routing'])
            
        if 'priority' in data and isinstance(data['priority'], str):
            context.priority = LayerPriority[data['priority']]
            
        if 'processing_mode' in data and isinstance(data['processing_mode'], str):
            context.processing_mode = LayerProcessingMode[data['processing_mode']]
            
        if 'communication_type' in data and isinstance(data['communication_type'], str):
            context.communication_type = LayerCommunicationType[data['communication_type']]
            
        if 'require_ack' in data:
            context.require_ack = data['require_ack']
            
        if 'idempotency_key' in data:
            context.idempotency_key = data['idempotency_key']
            
        if 'time_budget_ms' in data:
            context.time_budget_ms = data['time_budget_ms']
            
        if 'processing_deadline' in data:
            context.processing_deadline = data['processing_deadline']
            
        if 'dependencies' in data:
            context.dependencies = data['dependencies']
            
        return context
    
    def has_expired(self) -> bool:
        """Check if the context has expired based on deadline."""
        if self.processing_deadline is None:
            return False
            
        return time.time() > self.processing_deadline
    
    def is_high_priority(self) -> bool:
        """Check if this is high priority."""
        return self.priority in [LayerPriority.HIGH, LayerPriority.CRITICAL, LayerPriority.SYSTEM]

@dataclass
class LayerMessage(Message):
    """Extended message for inter-layer communication."""
    context: LayerContext = field(default_factory=LayerContext)
    layer_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        # Start with base message
        result = super().to_dict()
        
        # Add layer-specific fields
        result['layer_data'] = self.layer_data
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerMessage':
        """Create message from dictionary."""
        # Extract and convert content
        context_data = data.get('context', {})
        
        # Create layer context
        if context_data:
            context = LayerContext.from_dict(context_data)
        else:
            context = LayerContext()
            
        # Copy all normal message fields
        message = super().from_dict(data)
        
        # Replace with layer context
        message.context = context
        
        # Add layer data
        if 'layer_data' in data:
            message.layer_data = data['layer_data']
            
        return message
    
    def create_reply(self, content: Dict[str, Any], message_type: Optional[MessageType] = None) -> 'LayerMessage':
        """Create a reply to this message."""
        # Get base reply
        reply = super().create_reply(content, message_type)
        
        # Convert to LayerMessage
        layer_reply = LayerMessage(
            id=reply.id,
            message_type=reply.message_type,
            content=reply.content,
            context=reply.context,
            prediction=reply.prediction,
            sender_id=reply.sender_id,
            recipient_id=reply.recipient_id,
            created_at=reply.created_at,
            reply_to=reply.reply_to,
            correlation_id=reply.correlation_id,
            ttl=reply.ttl,
            priority=reply.priority,
            is_encrypted=reply.is_encrypted,
            encryption_info=reply.encryption_info,
            signature=reply.signature,
            trace_id=reply.trace_id
        )
        
        # Convert context to LayerContext if it's not already
        if not isinstance(layer_reply.context, LayerContext):
            context_dict = layer_reply.context.to_dict()
            layer_reply.context = LayerContext.from_dict(context_dict)
        
        # Swap source and target layers
        if layer_reply.context.layer_routing:
            source_layer = layer_reply.context.layer_routing.source_layer
            target_layer = layer_reply.context.layer_routing.target_layer
            
            # Set up routing for reply
            layer_reply.context.layer_routing.source_layer = target_layer or LayerType.UNKNOWN
            layer_reply.context.layer_routing.target_layer = source_layer
            layer_reply.context.layer_routing.hop_count = 0
            layer_reply.context.layer_routing.routing_path = []
            
            # Set appropriate communication type for reply
            if layer_reply.context.communication_type == LayerCommunicationType.REQUEST:
                layer_reply.context.communication_type = LayerCommunicationType.RESPONSE
        
        return layer_reply
    
    def duplicate_to_target(self, target_layer: LayerType) -> 'LayerMessage':
        """
        Duplicate this message to a new target layer.
        Useful for broadcasts or forwarding.
        """
        new_msg = LayerMessage(
            id=str(uuid.uuid4()),  # New ID for the duplicated message
            message_type=self.message_type,
            content=self.content.copy() if self.content else {},
            prediction=self.prediction,  # Reuse same prediction
            sender_id=self.sender_id,
            created_at=time.time(),  # New timestamp
            correlation_id=self.correlation_id,  # Keep same correlation
            ttl=self.ttl,
            priority=self.priority,
            trace_id=self.trace_id
        )
        
        # Copy context but update routing
        if isinstance(self.context, LayerContext):
            new_context = LayerContext(
                entity_id=self.context.entity_id,
                environmental_id=self.context.environmental_id,
                level=self.context.level,
                conversation_id=self.context.conversation_id,
                priority=self.context.priority,
                processing_mode=self.context.processing_mode,
                communication_type=self.context.communication_type,
                require_ack=self.context.require_ack
            )
            
            # Set up new routing
            new_routing = LayerRoutingInfo(
                source_layer=self.context.layer_routing.source_layer,
                target_layer=target_layer,
                hop_count=0,  # Reset hop count
                max_hops=self.context.layer_routing.max_hops
            )
            
            new_context.layer_routing = new_routing
            new_msg.context = new_context
        
        # Copy layer-specific data
        if self.layer_data:
            new_msg.layer_data = self.layer_data.copy()
            
        return new_msg


# === FILE: mcp_sdk/extensions/internal/client.py ===

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, Set

from mcp_sdk.client import MCPClient
from mcp_sdk.config import MCPConfig
from mcp_sdk.models import Message, Context, Prediction
from mcp_sdk.enums import MessageType, ContextLevel

from mcp_sdk.extensions.internal.models import (
    LayerMessage, 
    LayerContext, 
    LayerRoutingInfo
)
from mcp_sdk.extensions.internal.enums import (
    LayerType,
    LayerPriority,
    LayerProcessingMode,
    LayerCommunicationType
)

class InternalMCPClient(MCPClient):
    """
    Enhanced MCP client for inter-layer ASF communication.
    Extends the base MCPClient with layer-specific functionality.
    """
    def __init__(self, config: MCPConfig, layer_type: LayerType):
        """
        Initialize the internal MCP client.
        
        Args:
            config: Configuration for the client
            layer_type: The type of layer this client represents
        """
        super().__init__(config)
        self.layer_type = layer_type
        self.connected_layers: Set[LayerType] = set()
        self.logger = logging.getLogger(f"mcp_sdk.internal.{layer_type.name}")
        
        # Layer-specific message handlers
        self.layer_handlers: Dict[LayerCommunicationType, 
                               List[Callable[[LayerMessage], Awaitable[Optional[LayerMessage]]]]] = {
            comm_type: [] for comm_type in LayerCommunicationType
        }
    
    async def start(self) -> None:
        """Start the client and background processing."""
        await super().start()
        self.logger.info(f"Internal MCP Client started for layer {self.layer_type.name}")
    
    async def connect_to_layer(self, layer_type: LayerType) -> bool:
        """
        Connect to another ASF layer.
        
        Args:
            layer_type: The layer to connect to
            
        Returns:
            True if connection successful
        """
        if layer_type == self.layer_type:
            return False  # Can't connect to self
            
        # In a real implementation, this would establish a connection
        # For this example, we'll just track connected layers
        self.connected_layers.add(layer_type)
        self.logger.info(f"Connected to layer {layer_type.name}")
        
        return True
    
    async def disconnect_from_layer(self, layer_type: LayerType) -> bool:
        """
        Disconnect from another ASF layer.
        
        Args:
            layer_type: The layer to disconnect from
            
        Returns:
            True if disconnection successful
        """
        if layer_type in self.connected_layers:
            self.connected_layers.remove(layer_type)
            self.logger.info(f"Disconnected from layer {layer_type.name}")
            return True
            
        return False
    
    async def register_layer_handler(self, 
                                 communication_type: LayerCommunicationType,
                                 handler: Callable[[LayerMessage], Awaitable[Optional[LayerMessage]]]) -> None:
        """
        Register a handler for a specific layer communication type.
        
        Args:
            communication_type: The type of communication to handle
            handler: The handler function (async)
        """
        if communication_type not in self.layer_handlers:
            self.layer_handlers[communication_type] = []
            
        self.layer_handlers[communication_type].append(handler)
        self.logger.debug(f"Registered handler for communication type {communication_type.name}")
    
    async def create_layer_message(self,
                              message_type: MessageType,
                              content: Dict[str, Any],
                              target_layer: LayerType,
                              communication_type: LayerCommunicationType = LayerCommunicationType.NOTIFICATION,
                              processing_mode: LayerProcessingMode = LayerProcessingMode.ASYNC,
                              priority: LayerPriority = LayerPriority.NORMAL,
                              layer_data: Optional[Dict[str, Any]] = None,
                              **kwargs) -> LayerMessage:
        """
        Create a new inter-layer message.
        
        Args:
            message_type: Type of message
            content: Message content
            target_layer: Target layer
            communication_type: Type of layer communication
            processing_mode: How the message should be processed
            priority: Priority level
            layer_data: Layer-specific data
            **kwargs: Additional message parameters
            
        Returns:
            The created LayerMessage
        """
        # Create layer routing info
        routing = LayerRoutingInfo(
            source_layer=self.layer_type,
            target_layer=target_layer,
            max_hops=3
        )
        
        # Create layer context
        context = LayerContext(
            entity_id=self.entity_id,
            level=ContextLevel.ENHANCED,
            layer_routing=routing,
            communication_type=communication_type,
            processing_mode=processing_mode,
            priority=priority
        )
        
        # Handle deadline if time_budget provided
        if 'time_budget_ms' in kwargs:
            context.time_budget_ms = kwargs.pop('time_budget_ms')
            context.processing_deadline = time.time() + (context.time_budget_ms / 1000.0)
            
        # Create the message
        message = LayerMessage(
            message_type=message_type,
            content=content,
            context=context,
            sender_id=self.entity_id,
            layer_data=layer_data or {},
            **kwargs
        )
        
        # Add to conversation history
        self._add_to_conversation(message)
        
        return message
    
    async def send_layer_message(self, 
                            message: LayerMessage,
                            wait_for_response: bool = False,
                            timeout: Optional[float] = None) -> Optional[LayerMessage]:
        """
        Send an inter-layer message.
        
        Args:
            message: The message to send
            wait_for_response: Whether to wait for a response
            timeout: Timeout in seconds (None for default)
            
        Returns:
            Response message if wait_for_response is True, else None
        """
        # Check if target layer is connected
        target_layer = message.context.layer_routing.target_layer
        
        if target_layer and target_layer != self.layer_type and target_layer not in self.connected_layers:
            self.logger.warning(f"Attempting to send to disconnected layer {target_layer.name}")
            
        # Add to conversation history
        self._add_to_conversation(message)
        
        # Process through message queue
        await self._message_queue.put(message)
        
        # Set up for handling response if needed
        if wait_for_response:
            timeout = timeout or self.config.default_timeout
            
            # Create a future for the response
            response_future = asyncio.Future()
            self.pending_responses[message.id] = response_future
            
            try:
                # Wait for response with timeout
                response = await asyncio.wait_for(response_future, timeout)
                
                # Check if response is a LayerMessage
                if response and not isinstance(response, LayerMessage):
                    # Convert to LayerMessage if needed
                    if isinstance(response, Message):
                        # Convert Message to LayerMessage
                        response_dict = response.to_dict()
                        response = LayerMessage.from_dict(response_dict)
                
                return response
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout waiting for response to message {message.id}")
                
                # Clean up
                if message.id in self.pending_responses:
                    del self.pending_responses[message.id]
                    
                return None
                
            finally:
                # Clean up
                if message.id in self.pending_responses:
                    del self.pending_responses[message.id]
        
        return None
    
    async def send_request_to_layer(self,
                               target_layer: LayerType,
                               request_data: Dict[str, Any],
                               priority: LayerPriority = LayerPriority.NORMAL,
                               timeout: Optional[float] = None) -> Optional[LayerMessage]:
        """
        Send a request to another layer and wait for response.
        
        Args:
            target_layer: Target layer
            request_data: Request data
            priority: Priority level
            timeout: Timeout in seconds (None for default)
            
        Returns:
            Response message or None if timeout
        """
        # Create the message
        message = await self.create_layer_message(
            message_type=MessageType.QUERY,
            content=request_data,
            target_layer=target_layer,
            communication_type=LayerCommunicationType.REQUEST,
            priority=priority
        )
        
        # Send and wait for response
        return await self.send_layer_message(message, wait_for_response=True, timeout=timeout)
    
    async def broadcast_to_layers(self,
                             content: Dict[str, Any],
                             message_type: MessageType = MessageType.SYSTEM,
                             excluded_layers: Optional[List[LayerType]] = None,
                             priority: LayerPriority = LayerPriority.NORMAL) -> Dict[str, Any]:
        """
        Broadcast a message to all connected layers.
        
        Args:
            content: Message content
            message_type: Type of message
            excluded_layers: Layers to exclude from broadcast
            priority: Priority level
            
        Returns:
            Results of broadcast
        """
        excluded = set(excluded_layers or [])
        excluded.add(self.layer_type)  # Don't broadcast to self
        
        # Create routing info for broadcast
        routing = LayerRoutingInfo(
            source_layer=self.layer_type,
            broadcast=True,
            exclude_layers=excluded,
            max_hops=1  # Direct broadcast only
        )
        
        # Create layer context
        context = LayerContext(
            entity_id=self.entity_id,
            level=ContextLevel.BASIC,
            layer_routing=routing,
            communication_type=LayerCommunicationType.BROADCAST,
            processing_mode=LayerProcessingMode.ASYNC,
            priority=priority
        )
        
        # Create the message
        message = LayerMessage(
            message_type=message_type,
            content=content,
            context=context,
            sender_id=self.entity_id
        )
        
        # In a real implementation, this would send to all layers
        # For this example, we'll just log and process locally
        self.logger.info(f"Broadcasting message {message.id} to all layers except {[l.name for l in excluded]}")
        
        # Add to conversation history
        self._add_to_conversation(message)
        
        # Process through message queue
        await self._message_queue.put(message)
        
        return {
            'message_id': message.id,
            'broadcast_targets': len(self.connected_layers - excluded),
            'success': True
        }
    
    async def send_notification_to_layer(self,
                                     target_layer: LayerType,
                                     notification_data: Dict[str, Any],
                                     message_type: MessageType = MessageType.SYSTEM,
                                     priority: LayerPriority = LayerPriority.NORMAL) -> Dict[str, Any]:
        """
        Send a one-way notification to another layer.
        
        Args:
            target_layer: Target layer
            notification_data: Notification data
            message_type: Type of message
            priority: Priority level
            
        Returns:
            Result of sending notification
        """
        # Create the message
        message = await self.create_layer_message(
            message_type=message_type,
            content=notification_data,
            target_layer=target_layer,
            communication_type=LayerCommunicationType.NOTIFICATION,
            priority=priority
        )
        
        # Send without waiting for response
        await self.send_layer_message(message)
        
        return {
            'message_id': message.id,
            'target_layer': target_layer.name,
            'success': True
        }
    
    async def process_layer_message(self, message: LayerMessage) -> Optional[LayerMessage]:
        """
        Process an inter-layer message.
        
        Args:
            message: The message to process
            
        Returns:
            Response message if any
        """
        # Check if message is for this layer
        if not self._should_process_message(message):
            self.logger.debug(f"Skipping message {message.id} not targeted at this layer")
            return None
            
        self.logger.debug(f"Processing layer message {message.id} of type {message.context.communication_type.name}")
        
        # Check for deadline expiration
        if isinstance(message.context, LayerContext) and message.context.has_expired():
            self.logger.warning(f"Message {message.id} has expired (deadline passed)")
            return self._create_error_response(message, "Processing deadline exceeded")
            
        # Update routing path
        if isinstance(message.context, LayerContext) and message.context.layer_routing:
            message.context.layer_routing.add_hop(self.layer_type)
            
        # Check for layer-specific handlers
        if isinstance(message.context, LayerContext):
            comm_type = message.context.communication_type
            handlers = self.layer_handlers.get(comm_type, [])
            
            for handler in handlers:
                try:
                    response = await handler(message)
                    if response:
                        return response
                except Exception as e:
                    self.logger.error(f"Error in layer handler for {comm_type.name}: {str(e)}")
        
        # Fall back to base message processing
        return await super().process_message(message)
    
    def _should_process_message(self, message: LayerMessage) -> bool:
        """
        Check if this layer should process the message.
        
        Args:
            message: The message to check
            
        Returns:
            True if this layer should process the message
        """
        if not isinstance(message.context, LayerContext) or not message.context.layer_routing:
            # Not a layer message or missing routing info
            return True
            
        # Check routing
        routing = message.context.layer_routing
        return routing.should_process(self.layer_type)
    
    async def _process_messages(self) -> None:
        """Override of background task for processing messages."""
        self.logger.info("Layer message processing task started")
        
        while self._running:
            try:
                # Get the next message
                message = await self._message_queue.get()
                
                # Special handling for layer messages
                if isinstance(message, LayerMessage):
                    response = await self.process_layer_message(message)
                    
                    # If handler returns a response, send it
                    if response:
                        # Add to conversation history
                        self._add_to_conversation(response)
                        
                        # If response is to a pending message, resolve the future
                        if response.reply_to in self.pending_responses:
                            future = self.pending_responses[response.reply_to]
                            if not future.done():
                                future.set_result(response)
                else:
                    # Normal message processing
                    await super()._process_messages()
                    
                # Mark as done
                self._message_queue.task_done()
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                self.logger.error(f"Error in layer message processing: {str(e)}")
                
        self.logger.info("Layer message processing task stopped")


# === FILE: mcp_sdk/extensions/internal/utils.py ===

from typing import List, Dict, Any, Optional, Union, Set
import asyncio
import time

from mcp_sdk.extensions.internal.models import (
    LayerMessage, 
    LayerContext, 
    LayerRoutingInfo
)
from mcp_sdk.extensions.internal.enums import (
    LayerType,
    LayerPriority,
    LayerProcessingMode,
    LayerCommunicationType
)

async def collect_layer_responses(client, message_ids: List[str], 
                               timeout: float = 10.0,
                               min_responses: int = 1) -> Dict[str, Optional[LayerMessage]]:
    """
    Wait for responses from multiple layers.
    
    Args:
        client: The InternalMCPClient
        message_ids: List of message IDs to wait for
        timeout: Maximum time to wait in seconds
        min_responses: Minimum number of responses to collect before returning
        
    Returns:
        Dictionary mapping message IDs to responses
    """
    # Create futures for each message
    futures = {}
    for msg_id in message_ids:
        if msg_id not in client.pending_responses:
            client.pending_responses[msg_id] = asyncio.Future()
        futures[msg_id] = client.pending_responses[msg_id]
        
    # Wait for minimum responses or timeout
    start_time = time.time()
    results = {}
    
    while len(results) < min_responses and (time.time() - start_time) < timeout:
        # Check which futures are done
        for msg_id, future in list(futures.items()):
            if future.done() and msg_id not in results:
                try:
                    results[msg_id] = future.result()
                except Exception:
                    results[msg_id] = None
                    
        # If we have enough responses, break
        if len(results) >= min_responses:
            break
            
        # Wait a bit before checking again
        await asyncio.sleep(0.1)
        
    # Add any remaining futures that completed during final sleep
    for msg_id, future in futures.items():
        if future.done() and msg_id not in results:
            try:
                results[msg_id] = future.result()
            except Exception:
                results[msg_id] = None
    
    # Clean up
    for msg_id in message_ids:
        if msg_id in client.pending_responses:
            del client.pending_responses[msg_id]
            
    return results

def find_layer_connections(all_layers: Set[LayerType], 
                        source_layer: LayerType, 
                        target_layer: LayerType,
                        max_hops: int = 3) -> List[List[LayerType]]:
    """
    Find all possible connection paths between layers.
    
    Args:
        all_layers: Set of all layers
        source_layer: Source layer
        target_layer: Target layer
        max_hops: Maximum path length
        
    Returns:
        List of possible paths
    """
    # Simple BFS to find paths
    queue = [[source_layer]]
    paths = []
    
    while queue:
        path = queue.pop(0)
        current = path[-1]
        
        # Found target
        if current == target_layer:
            paths.append(path)
            continue
            
        # Max hops reached
        if len(path) > max_hops:
            continue
            
        # Try all connections
        for layer in all_layers:
            if layer not in path:  # Avoid cycles
                new_path = list(path)
                new_path.append(layer)
                queue.append(new_path)
    
    return paths

def get_optimal_routing_path(source_layer: LayerType, 
                          target_layer: LayerType,
                          available_layers: Set[LayerType]) -> List[LayerType]:
    """
    Get optimal routing path between layers.
    
    Args:
        source_layer: Source layer
        target_layer: Target layer
        available_layers: Set of available layers
        
    Returns:
        Optimal routing path
    """
    # Direct connection if target is available
    if target_layer in available_layers:
        return [source_layer, target_layer]
        
    # Find all possible paths
    paths = find_layer_connections(available_layers, source_layer, target_layer)
    
    # No path found
    if not paths:
        return [source_layer]
        
    # Find shortest path
    shortest = min(paths, key=len)
    return shortest

def create_dependency_chain(messages: List[LayerMessage]) -> List[LayerMessage]:
    """
    Create a dependency chain between messages.
    First message depends on nothing, second depends on first, etc.
    
    Args:
        messages: List of messages to chain
        
    Returns:
        Updated list of messages with dependencies
    """
    if not messages or len(messages) < 2:
        return messages
        
    # Create dependency chain
    for i in range(1, len(messages)):
        prev_id = messages[i-1].id
        
        # Ensure context is LayerContext
        if not isinstance(messages[i].context, LayerContext):
            context_dict = messages[i].context.to_dict()
            messages[i].context = LayerContext.from_dict(context_dict)
            
        # Add dependency
        if prev_id not in messages[i].context.dependencies:
            messages[i].context.dependencies.append(prev_id)
            
    return messages

def merge_layer_data(message1: LayerMessage, message2: LayerMessage) -> Dict[str, Any]:
    """
    Merge layer data from two messages.
    
    Args:
        message1: First message
        message2: Second message
        
    Returns:
        Merged layer data
    """
    result = {}
    
    # Add data from first message
    if message1.layer_data:
        result.update(message1.layer_data)
        
    # Add data from second message, overriding duplicates
    if message2.layer_data:
        result.update(message2.layer_data)
        
    return result