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
            try:
                num = int(name[5:].split("_")[0])
                return next((l for l in cls if l.value == num), cls.UNKNOWN)
            except:
                pass
                
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
        if current_layer in self.routing_path:
            return False
            
        if self.broadcast:
            return current_layer not in self.exclude_layers
            
        if self.targets:
            return current_layer in self.targets
            
        if self.target_layer:
            return current_layer == self.target_layer
            
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
        result = super().to_dict()
        
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
        context_data = data.get('context', {})
        
        if context_data:
            context = LayerContext.from_dict(context_data)
        else:
            context = LayerContext()
            
        message = super().from_dict(data)
        
        message.context = context
        
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
    Enhanced MCP client for inter-layer ASF communication.
    Extends the base MCPClient with layer-specific functionality.
        Initialize the internal MCP client.
        
        Args:
            config: Configuration for the client
            layer_type: The type of layer this client represents
        await super().start()
        self.logger.info(f"Internal MCP Client started for layer {self.layer_type.name}")
    
    async def connect_to_layer(self, layer_type: LayerType) -> bool:
        if layer_type == self.layer_type:
            return False  # Can't connect to self
            
        self.connected_layers.add(layer_type)
        self.logger.info(f"Connected to layer {layer_type.name}")
        
        return True
    
    async def disconnect_from_layer(self, layer_type: LayerType) -> bool:
        if layer_type in self.connected_layers:
            self.connected_layers.remove(layer_type)
            self.logger.info(f"Disconnected from layer {layer_type.name}")
            return True
            
        return False
    
    async def register_layer_handler(self, 
                                 communication_type: LayerCommunicationType,
                                 handler: Callable[[LayerMessage], Awaitable[Optional[LayerMessage]]]) -> None:
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
        routing = LayerRoutingInfo(
            source_layer=self.layer_type,
            target_layer=target_layer,
            max_hops=3
        )
        
        context = LayerContext(
            entity_id=self.entity_id,
            level=ContextLevel.ENHANCED,
            layer_routing=routing,
            communication_type=communication_type,
            processing_mode=processing_mode,
            priority=priority
        )
        
        if 'time_budget_ms' in kwargs:
            context.time_budget_ms = kwargs.pop('time_budget_ms')
            context.processing_deadline = time.time() + (context.time_budget_ms / 1000.0)
            
        message = LayerMessage(
            message_type=message_type,
            content=content,
            context=context,
            sender_id=self.entity_id,
            layer_data=layer_data or {},
            **kwargs
        )
        
        self._add_to_conversation(message)
        
        return message
    
    async def send_layer_message(self, 
                            message: LayerMessage,
                            wait_for_response: bool = False,
                            timeout: Optional[float] = None) -> Optional[LayerMessage]:
        target_layer = message.context.layer_routing.target_layer
        
        if target_layer and target_layer != self.layer_type and target_layer not in self.connected_layers:
            self.logger.warning(f"Attempting to send to disconnected layer {target_layer.name}")
            
        self._add_to_conversation(message)
        
        await self._message_queue.put(message)
        
        if wait_for_response:
            timeout = timeout or self.config.default_timeout
            
            response_future = asyncio.Future()
            self.pending_responses[message.id] = response_future
            
            try:
                response = await asyncio.wait_for(response_future, timeout)
                
                if response and not isinstance(response, LayerMessage):
                    if isinstance(response, Message):
                        response_dict = response.to_dict()
                        response = LayerMessage.from_dict(response_dict)
                
                return response
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout waiting for response to message {message.id}")
                
                if message.id in self.pending_responses:
                    del self.pending_responses[message.id]
                    
                return None
                
            finally:
                if message.id in self.pending_responses:
                    del self.pending_responses[message.id]
        
        return None
    
    async def send_request_to_layer(self,
                               target_layer: LayerType,
                               request_data: Dict[str, Any],
                               priority: LayerPriority = LayerPriority.NORMAL,
                               timeout: Optional[float] = None) -> Optional[LayerMessage]:
        message = await self.create_layer_message(
            message_type=MessageType.QUERY,
            content=request_data,
            target_layer=target_layer,
            communication_type=LayerCommunicationType.REQUEST,
            priority=priority
        )
        
        return await self.send_layer_message(message, wait_for_response=True, timeout=timeout)
    
    async def broadcast_to_layers(self,
                             content: Dict[str, Any],
                             message_type: MessageType = MessageType.SYSTEM,
                             excluded_layers: Optional[List[LayerType]] = None,
                             priority: LayerPriority = LayerPriority.NORMAL) -> Dict[str, Any]:
        excluded = set(excluded_layers or [])
        excluded.add(self.layer_type)  # Don't broadcast to self
        
        routing = LayerRoutingInfo(
            source_layer=self.layer_type,
            broadcast=True,
            exclude_layers=excluded,
            max_hops=1  # Direct broadcast only
        )
        
        context = LayerContext(
            entity_id=self.entity_id,
            level=ContextLevel.BASIC,
            layer_routing=routing,
            communication_type=LayerCommunicationType.BROADCAST,
            processing_mode=LayerProcessingMode.ASYNC,
            priority=priority
        )
        
        message = LayerMessage(
            message_type=message_type,
            content=content,
            context=context,
            sender_id=self.entity_id
        )
        
        self.logger.info(f"Broadcasting message {message.id} to all layers except {[l.name for l in excluded]}")
        
        self._add_to_conversation(message)
        
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
        message = await self.create_layer_message(
            message_type=message_type,
            content=notification_data,
            target_layer=target_layer,
            communication_type=LayerCommunicationType.NOTIFICATION,
            priority=priority
        )
        
        await self.send_layer_message(message)
        
        return {
            'message_id': message.id,
            'target_layer': target_layer.name,
            'success': True
        }
    
    async def process_layer_message(self, message: LayerMessage) -> Optional[LayerMessage]:
        if not self._should_process_message(message):
            self.logger.debug(f"Skipping message {message.id} not targeted at this layer")
            return None
            
        self.logger.debug(f"Processing layer message {message.id} of type {message.context.communication_type.name}")
        
        if isinstance(message.context, LayerContext) and message.context.has_expired():
            self.logger.warning(f"Message {message.id} has expired (deadline passed)")
            return self._create_error_response(message, "Processing deadline exceeded")
            
        if isinstance(message.context, LayerContext) and message.context.layer_routing:
            message.context.layer_routing.add_hop(self.layer_type)
            
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
            return True
            
        routing = message.context.layer_routing
        return routing.should_process(self.layer_type)
    
    async def _process_messages(self) -> None:
    Wait for responses from multiple layers.
    
    Args:
        client: The InternalMCPClient
        message_ids: List of message IDs to wait for
        timeout: Maximum time to wait in seconds
        min_responses: Minimum number of responses to collect before returning
        
    Returns:
        Dictionary mapping message IDs to responses
    Find all possible connection paths between layers.
    
    Args:
        all_layers: Set of all layers
        source_layer: Source layer
        target_layer: Target layer
        max_hops: Maximum path length
        
    Returns:
        List of possible paths
    Get optimal routing path between layers.
    
    Args:
        source_layer: Source layer
        target_layer: Target layer
        available_layers: Set of available layers
        
    Returns:
        Optimal routing path
    Create a dependency chain between messages.
    First message depends on nothing, second depends on first, etc.
    
    Args:
        messages: List of messages to chain
        
    Returns:
        Updated list of messages with dependencies
    Merge layer data from two messages.
    
    Args:
        message1: First message
        message2: Second message
        
    Returns:
        Merged layer data