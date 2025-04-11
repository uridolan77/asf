from mcp_sdk.extensions.external.client import ExternalMCPClient
from mcp_sdk.extensions.external.models import (
    EnrichedMessage, 
    EnrichedContext, 
    SourceMetadata,
    AugmentationInfo,
    TransformationRule
)
from mcp_sdk.extensions.external.enums import (
    SourceType, 
    AugmentationType,
    ProtocolType,
    TransformationType,
    SecurityLevel
)

__all__ = [
    'ExternalMCPClient',
    'EnrichedMessage',
    'EnrichedContext',
    'SourceMetadata',
    'AugmentationInfo',
    'TransformationRule',
    'SourceType',
    'AugmentationType',
    'ProtocolType',
    'TransformationType',
    'SecurityLevel'
]



from enum import Enum, auto

class SourceType(Enum):
    """Types of external sources."""
    UNKNOWN = auto()
    USER = auto()              # Human user
    API = auto()               # External API
    MODEL = auto()             # External model
    SYSTEM = auto()            # External system
    DATABASE = auto()          # Database
    FILE = auto()              # File system
    SENSOR = auto()            # Sensor/IoT device
    STREAM = auto()            # Data stream
    SERVICE = auto()           # External service
    AGENT = auto()             # Autonomous agent

class AugmentationType(Enum):
    """Types of data augmentation."""
    NONE = auto()
    ENRICHMENT = auto()        # Add additional information
    TRANSLATION = auto()       # Convert between formats/languages
    SUMMARIZATION = auto()     # Summarize content
    EXPANSION = auto()         # Expand content with details
    CORRECTION = auto()        # Fix errors or inconsistencies
    ANNOTATION = auto()        # Add annotations or metadata
    FILTERING = auto()         # Filter or redact information
    REORGANIZATION = auto()    # Reorganize content structure
    FUSION = auto()            # Combine multiple sources

class ProtocolType(Enum):
    """Types of external protocols."""
    UNKNOWN = auto()
    HTTP = auto()              # HTTP/HTTPS
    WEBSOCKET = auto()         # WebSocket
    MQTT = auto()              # MQTT
    GRPC = auto()              # gRPC
    KAFKA = auto()             # Kafka
    RABBITMQ = auto()          # RabbitMQ
    REDIS = auto()             # Redis pub/sub
    SQL = auto()               # SQL databases
    FILE_SYSTEM = auto()       # File system
    CUSTOM = auto()            # Custom protocol

class TransformationType(Enum):
    """Types of transformations for protocol adaptation."""
    NONE = auto()
    FORMAT_CONVERSION = auto() # Convert between data formats
    SCHEMA_MAPPING = auto()    # Map between schemas
    PROTOCOL_ADAPTATION = auto() # Adapt between protocols
    SEMANTIC_MAPPING = auto()  # Map between semantic models
    FIELD_MAPPING = auto()     # Map specific fields
    FIELD_TRANSFORMATION = auto() # Transform field values
    AGGREGATION = auto()       # Aggregate multiple fields
    SPLITTING = auto()         # Split fields into multiple
    TIME_CONVERSION = auto()   # Convert between time formats/zones

class SecurityLevel(Enum):
    """Security levels for external communication."""
    PUBLIC = 0                 # Public, no security
    BASIC = 1                  # Basic security (e.g., simple authentication)
    STANDARD = 2               # Standard security (e.g., TLS)
    ENHANCED = 3               # Enhanced security (e.g., strong encryption)
    HIGH = 4                   # High security (e.g., multi-factor)
    MAXIMUM = 5                # Maximum security (e.g., zero-trust)


# === FILE: mcp_sdk/extensions/external/models.py ===

import time
import uuid
import json
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, field, asdict

from mcp_sdk.models import Message, Context, Prediction
from mcp_sdk.enums import MessageType, ContextLevel

from mcp_sdk.extensions.external.enums import (
    SourceType,
    AugmentationType,
    ProtocolType,
    TransformationType,
    SecurityLevel
)

@dataclass
class SourceMetadata:
    """Metadata about an external source."""
    source_id: str
    source_type: SourceType = SourceType.UNKNOWN
    source_name: Optional[str] = None
    source_version: Optional[str] = None
    protocol: ProtocolType = ProtocolType.UNKNOWN
    schema_version: Optional[str] = None
    location: Optional[str] = None  # URL, path, endpoint, etc.
    capabilities: Set[str] = field(default_factory=set)
    security_level: SecurityLevel = SecurityLevel.STANDARD
    connection_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'source_id': self.source_id,
            'source_type': self.source_type.name,
            'protocol': self.protocol.name,
            'security_level': self.security_level.name
        }
        
        if self.source_name:
            result['source_name'] = self.source_name
            
        if self.source_version:
            result['source_version'] = self.source_version
            
        if self.schema_version:
            result['schema_version'] = self.schema_version
            
        if self.location:
            result['location'] = self.location
            
        if self.capabilities:
            result['capabilities'] = list(self.capabilities)
            
        if self.connection_info:
            result['connection_info'] = self.connection_info
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SourceMetadata':
        """Create from dictionary."""
        source_metadata = cls(source_id=data.get('source_id', ''))
        
        if 'source_type' in data and isinstance(data['source_type'], str):
            source_metadata.source_type = SourceType[data['source_type']]
            
        if 'protocol' in data and isinstance(data['protocol'], str):
            source_metadata.protocol = ProtocolType[data['protocol']]
            
        if 'security_level' in data and isinstance(data['security_level'], str):
            source_metadata.security_level = SecurityLevel[data['security_level']]
            
        if 'source_name' in data:
            source_metadata.source_name = data['source_name']
            
        if 'source_version' in data:
            source_metadata.source_version = data['source_version']
            
        if 'schema_version' in data:
            source_metadata.schema_version = data['schema_version']
            
        if 'location' in data:
            source_metadata.location = data['location']
            
        if 'capabilities' in data and isinstance(data['capabilities'], list):
            source_metadata.capabilities = set(data['capabilities'])
            
        if 'connection_info' in data:
            source_metadata.connection_info = data['connection_info']
            
        return source_metadata

@dataclass
class TransformationRule:
    """Rule for transforming data between protocols/formats."""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    transformation_type: TransformationType = TransformationType.NONE
    source_format: Optional[str] = None
    target_format: Optional[str] = None
    source_schema: Optional[Dict[str, Any]] = None
    target_schema: Optional[Dict[str, Any]] = None
    field_mappings: Dict[str, str] = field(default_factory=dict)  # source->target
    transformers: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # field->transform_info
    default_values: Dict[str, Any] = field(default_factory=dict)  # field->default
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'rule_id': self.rule_id,
            'transformation_type': self.transformation_type.name
        }
        
        if self.source_format:
            result['source_format'] = self.source_format
            
        if self.target_format:
            result['target_format'] = self.target_format
            
        if self.source_schema:
            result['source_schema'] = self.source_schema
            
        if self.target_schema:
            result['target_schema'] = self.target_schema
            
        if self.field_mappings:
            result['field_mappings'] = self.field_mappings
            
        if self.transformers:
            result['transformers'] = self.transformers
            
        if self.default_values:
            result['default_values'] = self.default_values
            
        if self.validation_rules:
            result['validation_rules'] = self.validation_rules
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransformationRule':
        """Create from dictionary."""
        rule = cls()
        
        if 'rule_id' in data:
            rule.rule_id = data['rule_id']
            
        if 'transformation_type' in data and isinstance(data['transformation_type'], str):
            rule.transformation_type = TransformationType[data['transformation_type']]
            
        if 'source_format' in data:
            rule.source_format = data['source_format']
            
        if 'target_format' in data:
            rule.target_format = data['target_format']
            
        if 'source_schema' in data:
            rule.source_schema = data['source_schema']
            
        if 'target_schema' in data:
            rule.target_schema = data['target_schema']
            
        if 'field_mappings' in data:
            rule.field_mappings = data['field_mappings']
            
        if 'transformers' in data:
            rule.transformers = data['transformers']
            
        if 'default_values' in data:
            rule.default_values = data['default_values']
            
        if 'validation_rules' in data:
            rule.validation_rules = data['validation_rules']
            
        return rule
        
    def apply_transformation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply this transformation rule to data.
        
        Args:
            data: Source data to transform
            
        Returns:
            Transformed data
        Transform a field value based on transform info.
        
        Args:
            value: The value to transform
            transform_info: Transformation instructions
            
        Returns:
            Transformed value
    augmentation_type: AugmentationType = AugmentationType.NONE
    augmented_fields: Set[str] = field(default_factory=set)
    augmentation_source: Optional[str] = None
    augmentation_time: float = field(default_factory=time.time)
    augmentation_model: Optional[str] = None
    confidence: float = 1.0
    original_data: Optional[Dict[str, Any]] = None
    transformations_applied: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'augmentation_type': self.augmentation_type.name,
            'augmentation_time': self.augmentation_time,
            'confidence': self.confidence
        }
        
        if self.augmented_fields:
            result['augmented_fields'] = list(self.augmented_fields)
            
        if self.augmentation_source:
            result['augmentation_source'] = self.augmentation_source
            
        if self.augmentation_model:
            result['augmentation_model'] = self.augmentation_model
            
        if self.original_data:
            result['original_data'] = self.original_data
            
        if self.transformations_applied:
            result['transformations_applied'] = self.transformations_applied
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AugmentationInfo':
        """Create from dictionary."""
        augmentation = cls()
        
        if 'augmentation_type' in data and isinstance(data['augmentation_type'], str):
            augmentation.augmentation_type = AugmentationType[data['augmentation_type']]
            
        if 'augmented_fields' in data and isinstance(data['augmented_fields'], list):
            augmentation.augmented_fields = set(data['augmented_fields'])
            
        if 'augmentation_source' in data:
            augmentation.augmentation_source = data['augmentation_source']
            
        if 'augmentation_time' in data:
            augmentation.augmentation_time = data['augmentation_time']
            
        if 'augmentation_model' in data:
            augmentation.augmentation_model = data['augmentation_model']
            
        if 'confidence' in data:
            augmentation.confidence = data['confidence']
            
        if 'original_data' in data:
            augmentation.original_data = data['original_data']
            
        if 'transformations_applied' in data:
            augmentation.transformations_applied = data['transformations_applied']
            
        return augmentation

@dataclass
class EnrichedContext(Context):
    """Enhanced context for external communication."""
    source_metadata: Optional[SourceMetadata] = None
    target_metadata: Optional[SourceMetadata] = None
    augmentation_info: Optional[AugmentationInfo] = None
    transformation_rules: List[TransformationRule] = field(default_factory=list)
    extra_context: Dict[str, Any] = field(default_factory=dict)
    security_info: Dict[str, Any] = field(default_factory=dict)
    processing_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        result = super().to_dict()
        
        if self.source_metadata:
            result['source_metadata'] = self.source_metadata.to_dict()
            
        if self.target_metadata:
            result['target_metadata'] = self.target_metadata.to_dict()
            
        if self.augmentation_info:
            result['augmentation_info'] = self.augmentation_info.to_dict()
            
        if self.transformation_rules:
            result['transformation_rules'] = [rule.to_dict() for rule in self.transformation_rules]
            
        if self.extra_context:
            result['extra_context'] = self.extra_context
            
        if self.security_info:
            result['security_info'] = self.security_info
            
        if self.processing_info:
            result['processing_info'] = self.processing_info
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnrichedContext':
        """Create context from dictionary."""
        # Convert base context
        context = super().from_dict(data)
        
        # Convert enriched context fields
        if 'source_metadata' in data:
            context.source_metadata = SourceMetadata.from_dict(data['source_metadata'])
            
        if 'target_metadata' in data:
            context.target_metadata = SourceMetadata.from_dict(data['target_metadata'])
            
        if 'augmentation_info' in data:
            context.augmentation_info = AugmentationInfo.from_dict(data['augmentation_info'])
            
        if 'transformation_rules' in data and isinstance(data['transformation_rules'], list):
            context.transformation_rules = [
                TransformationRule.from_dict(rule) for rule in data['transformation_rules']
            ]
            
        if 'extra_context' in data:
            context.extra_context = data['extra_context']
            
        if 'security_info' in data:
            context.security_info = data['security_info']
            
        if 'processing_info' in data:
            context.processing_info = data['processing_info']
            
        return context
    
    def add_transformation_rule(self, rule: TransformationRule) -> None:
        """Add a transformation rule."""
        self.transformation_rules.append(rule)
    
    def get_source_capability(self, capability: str) -> bool:
        """Check if source has a capability."""
        if self.source_metadata and self.source_metadata.capabilities:
            return capability in self.source_metadata.capabilities
        return False
    
    def record_augmentation(self, 
                         augmentation_type: AugmentationType,
                         augmented_fields: Set[str],
                         original_data: Optional[Dict[str, Any]] = None,
                         confidence: float = 1.0) -> None:
        """Record an augmentation."""
        if not self.augmentation_info:
            self.augmentation_info = AugmentationInfo()
            
        self.augmentation_info.augmentation_type = augmentation_type
        self.augmentation_info.augmented_fields = augmented_fields
        self.augmentation_info.augmentation_time = time.time()
        self.augmentation_info.confidence = confidence
        
        if original_data:
            self.augmentation_info.original_data = original_data

@dataclass
class EnrichedMessage(Message):
    """Enhanced message for external communication."""
    context: EnrichedContext = field(default_factory=EnrichedContext)
    original_format: Optional[str] = None
    transformed_data: Dict[str, Any] = field(default_factory=dict)
    augmented_content: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        result = super().to_dict()
        
        if self.original_format:
            result['original_format'] = self.original_format
            
        if self.transformed_data:
            result['transformed_data'] = self.transformed_data
            
        if self.augmented_content:
            result['augmented_content'] = self.augmented_content
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnrichedMessage':
        """Create message from dictionary."""
        # Extract and convert content
        context_data = data.get('context', {})
        
        # Create enriched context
        if context_data:
            context = EnrichedContext.from_dict(context_data)
        else:
            context = EnrichedContext()
            
        # Copy all normal message fields
        message = super().from_dict(data)
        
        # Replace with enriched context
        message.context = context
        
        # Add enriched fields
        if 'original_format' in data:
            message.original_format = data['original_format']
            
        if 'transformed_data' in data:
            message.transformed_data = data['transformed_data']
            
        if 'augmented_content' in data:
            message.augmented_content = data['augmented_content']
            
        return message
    
    def create_reply(self, content: Dict[str, Any], message_type: Optional[MessageType] = None) -> 'EnrichedMessage':
        """Create a reply to this message."""
        reply = super().create_reply(content, message_type)
        
        enriched_reply = EnrichedMessage(
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
        
        if not isinstance(enriched_reply.context, EnrichedContext):
            context_dict = enriched_reply.context.to_dict()
            enriched_reply.context = EnrichedContext.from_dict(context_dict)
        
        source_metadata = None
        target_metadata = None
        
        if hasattr(self.context, 'source_metadata') and self.context.source_metadata:
            target_metadata = self.context.source_metadata
            
        if hasattr(self.context, 'target_metadata') and self.context.target_metadata:
            source_metadata = self.context.target_metadata
            
        enriched_reply.context.source_metadata = source_metadata
        enriched_reply.context.target_metadata = target_metadata
        
        return enriched_reply
    
    def add_augmentation(self, 
                      augmentation_type: AugmentationType,
                      augmented_data: Dict[str, Any],
                      augmented_fields: Set[str],
                      save_original: bool = True) -> None:
        if not isinstance(self.context, EnrichedContext):
            context_dict = self.context.to_dict()
            self.context = EnrichedContext.from_dict(context_dict)
            
        original_data = None
        if save_original:
            original_data = {}
            for field in augmented_fields:
                if field in self.content:
                    original_data[field] = self.content[field]
            
        self.context.record_augmentation(
            augmentation_type=augmentation_type,
            augmented_fields=augmented_fields,
            original_data=original_data,
            confidence=1.0
        )
        
        self.augmented_content.update(augmented_data)
        
        for field, value in augmented_data.items():
            self.content[field] = value
    
    def apply_transformation_rule(self, rule: TransformationRule, apply_to_content: bool = True) -> Dict[str, Any]:
        """
        Apply a transformation rule to the message.
        
        Args:
            rule: The transformation rule to apply
            apply_to_content: Whether to update message content
            
        Returns:
            Transformed data
        """
        transformed = rule.apply_transformation(self.content)
        
        self.transformed_data.update(transformed)
        
        if apply_to_content:
            self.content.update(transformed)
            
        if isinstance(self.context, EnrichedContext):
            self.context.add_transformation_rule(rule)
            
            if self.context.augmentation_info:
                if rule.rule_id not in self.context.augmentation_info.transformations_applied:
                    self.context.augmentation_info.transformations_applied.append(rule.rule_id)
        
        return transformed



import asyncio
import logging
import time
import uuid
import json
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, Set

from mcp_sdk.client import MCPClient
from mcp_sdk.config import MCPConfig
from mcp_sdk.models import Message, Context, Prediction
from mcp_sdk.enums import MessageType, ContextLevel

from mcp_sdk.extensions.external.models import (
    EnrichedMessage, 
    EnrichedContext, 
    SourceMetadata,
    AugmentationInfo,
    TransformationRule
)
from mcp_sdk.extensions.external.enums import (
    SourceType,
    AugmentationType,
    ProtocolType,
    TransformationType,
    SecurityLevel
)

class ExternalMCPClient(MCPClient):
    """
    Enhanced MCP client for external communication.
    Extends the base MCPClient with enrichment and transformation capabilities.
    """
    def __init__(self, config: MCPConfig):
        """
        Initialize the external MCP client.
        
        Args:
            config: Configuration for the client
        """
        super().__init__(config)
        self.logger = logging.getLogger("mcp_sdk.external")
        
        self.sources: Dict[str, SourceMetadata] = {}
        
        self.transformation_rules: Dict[str, TransformationRule] = {}
        
        self.augmentation_handlers: Dict[AugmentationType, 
                                    List[Callable[[EnrichedMessage], Awaitable[EnrichedMessage]]]] = {
            aug_type: [] for aug_type in AugmentationType
        }
        
        self.source_handlers: Dict[str, List[Callable[[EnrichedMessage], Awaitable[Optional[EnrichedMessage]]]]] = {}
    
    async def register_source(self, source_metadata: SourceMetadata) -> str:
        source_id = source_metadata.source_id
        self.sources[source_id] = source_metadata
        self.logger.info(f"Registered source: {source_id} ({source_metadata.source_type.name})")
        return source_id
    
    async def register_transformation_rule(self, rule: TransformationRule) -> str:
        rule_id = rule.rule_id
        self.transformation_rules[rule_id] = rule
        self.logger.info(f"Registered transformation rule: {rule_id} ({rule.transformation_type.name})")
        return rule_id
    
    async def register_augmentation_handler(self, 
                                       augmentation_type: AugmentationType,
                                       handler: Callable[[EnrichedMessage], Awaitable[EnrichedMessage]]) -> None:
        if augmentation_type not in self.augmentation_handlers:
            self.augmentation_handlers[augmentation_type] = []
            
        self.augmentation_handlers[augmentation_type].append(handler)
        self.logger.debug(f"Registered handler for augmentation type {augmentation_type.name}")
    
    async def register_source_handler(self, 
                                  source_id: str,
                                  handler: Callable[[EnrichedMessage], Awaitable[Optional[EnrichedMessage]]]) -> None:
        if source_id not in self.source_handlers:
            self.source_handlers[source_id] = []
            
        self.source_handlers[source_id].append(handler)
        self.logger.debug(f"Registered handler for source {source_id}")
    
    async def create_enriched_message(self, 
                                  message_type: MessageType,
                                  content: Dict[str, Any],
                                  source_id: Optional[str] = None,
                                  target_id: Optional[str] = None,
                                  context_level: ContextLevel = ContextLevel.ENHANCED,
                                  security_level: SecurityLevel = SecurityLevel.STANDARD,
                                  extra_context: Optional[Dict[str, Any]] = None,
                                  **kwargs) -> EnrichedMessage:
        source_metadata = None
        target_metadata = None
        
        if source_id and source_id in self.sources:
            source_metadata = self.sources[source_id]
            
        if target_id and target_id in self.sources:
            target_metadata = self.sources[target_id]
            
        context = EnrichedContext(
            entity_id=self.entity_id,
            environmental_id=target_id,
            level=context_level,
            source_metadata=source_metadata,
            target_metadata=target_metadata,
            extra_context=extra_context or {},
            security_info={
                'security_level': security_level.name,
                'timestamp': time.time()
            }
        )
        
        message = EnrichedMessage(
            message_type=message_type,
            content=content,
            context=context,
            sender_id=self.entity_id,
            recipient_id=target_id,
            **kwargs
        )
        
        self._add_to_conversation(message)
        
        return message
    
    async def send_enriched_message(self, 
                               message: EnrichedMessage,
                               apply_transformations: bool = True,
                               apply_augmentations: bool = True,
                               wait_for_response: bool = False,
                               timeout: Optional[float] = None) -> Optional[EnrichedMessage]:
        if apply_transformations:
            message = await self._apply_transformations(message)
            
        if apply_augmentations:
            message = await self._apply_augmentations(message)
            
        self._add_to_conversation(message)
        
        await self._message_queue.put(message)
        
        if not wait_for_response:
            return None
            
        timeout = timeout or self.config.default_timeout
        
        response_future = asyncio.Future()
        self.pending_responses[message.id] = response_future
        
        try:
            response = await asyncio.wait_for(response_future, timeout)
            
            if response and not isinstance(response, EnrichedMessage):
                if isinstance(response, Message):
                    response_dict = response.to_dict()
                    response = EnrichedMessage.from_dict(response_dict)
                    
            return response
            
        except asyncio.TimeoutError:
            if message.id in self.pending_responses:
                del self.pending_responses[message.id]
                
            return None
            
        finally:
            if message.id in self.pending_responses:
                del self.pending_responses[message.id]
    
    async def process_external_message(self, 
                                   message: Union[Dict[str, Any], Message, EnrichedMessage],
                                   source_id: Optional[str] = None) -> Optional[EnrichedMessage]:
        if isinstance(message, dict):
            try:
                message = EnrichedMessage.from_dict(message)
            except Exception as e:
                self.logger.error(f"Error converting dictionary to message: {str(e)}")
                return None
        elif isinstance(message, Message) and not isinstance(message, EnrichedMessage):
            message_dict = message.to_dict()
            message = EnrichedMessage.from_dict(message_dict)
            
        if source_id and source_id in self.sources:
            source_metadata = self.sources[source_id]
            
            if not isinstance(message.context, EnrichedContext):
                context_dict = message.context.to_dict()
                message.context = EnrichedContext.from_dict(context_dict)
                
            message.context.source_metadata = source_metadata
            
        if hasattr(message.context, 'source_metadata') and message.context.source_metadata:
            source_id = message.context.source_metadata.source_id
            
            if source_id in self.source_handlers:
                for handler in self.source_handlers[source_id]:
                    try:
                        result = await handler(message)
                        if result:
                            return result
                    except Exception as e:
                        self.logger.error(f"Error in source handler for {source_id}: {str(e)}")
        
        return await super().process_message(message)
    
    async def query_external_source(self,
                               source_id: str,
                               query_data: Dict[str, Any],
                               apply_transformations: bool = True,
                               apply_augmentations: bool = True,
                               timeout: Optional[float] = None) -> Optional[EnrichedMessage]:
        if source_id not in self.sources:
            self.logger.warning(f"Unknown source: {source_id}")
            return None
            
        source_metadata = self.sources[source_id]
        
        context = EnrichedContext(
            entity_id=self.entity_id,
            environmental_id=source_id,
            level=ContextLevel.ENHANCED,
            source_metadata=None,
            target_metadata=source_metadata
        )
        
        message = EnrichedMessage(
            message_type=MessageType.QUERY,
            content=query_data,
            context=context,
            sender_id=self.entity_id,
            recipient_id=source_id
        )
        
        return await self.send_enriched_message(
            message,
            apply_transformations=apply_transformations,
            apply_augmentations=apply_augmentations,
            wait_for_response=True,
            timeout=timeout
        )
    
    async def _apply_transformations(self, message: EnrichedMessage) -> EnrichedMessage:
        if not self.transformation_rules:
            return message
            
        source_format = None
        target_format = None
        
        if hasattr(message.context, 'source_metadata') and message.context.source_metadata:
            source_metadata = message.context.source_metadata
            if hasattr(source_metadata, 'connection_info'):
                source_format = source_metadata.connection_info.get('format')
                
        if hasattr(message.context, 'target_metadata') and message.context.target_metadata:
            target_metadata = message.context.target_metadata
            if hasattr(target_metadata, 'connection_info'):
                target_format = target_metadata.connection_info.get('format')
                
        for rule in self.transformation_rules.values():
            if ((rule.source_format is None or rule.source_format == source_format) and
                (rule.target_format is None or rule.target_format == target_format)):
                
                message.apply_transformation_rule(rule)
                
        return message
    
    async def _apply_augmentations(self, message: EnrichedMessage) -> EnrichedMessage:
        augmentation_types = list(self.augmentation_handlers.keys())
        if not augmentation_types:
            return message
            
        for aug_type in augmentation_types:
            if aug_type == AugmentationType.NONE:
                continue
                
            handlers = self.augmentation_handlers.get(aug_type, [])
            for handler in handlers:
                try:
                    message = await handler(message)
                except Exception as e:
                    self.logger.error(f"Error in augmentation handler for {aug_type.name}: {str(e)}")
                    
        return message
    
    async def get_source_stats(self) -> Dict[str, Any]:
    def __init__(self):
        """Initialize the augmentor."""
        self.augmentation_type = AugmentationType.NONE
    
    async def augment(self, message: EnrichedMessage) -> EnrichedMessage:
        """
        Augment a message.
        
        Args:
            message: The message to augment
            
        Returns:
            Augmented message
        Record an augmentation in the message.
        
        Args:
            message: The message being augmented
            augmented_fields: Set of field names that were augmented
            original_data: Original data before augmentation
            confidence: Confidence in the augmentation
    def __init__(self, enrichment_function: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]):
        """
        Initialize the enrichment augmentor.
        
        Args:
            enrichment_function: Async function that takes content and returns enriched content
        """
        super().__init__()
        self.augmentation_type = AugmentationType.ENRICHMENT
        self.enrichment_function = enrichment_function
    
    async def augment(self, message: EnrichedMessage) -> EnrichedMessage:
        original_content = message.content.copy()
        
        enriched_content = await self.enrichment_function(original_content)
        
        augmented_fields = set()
        for key in enriched_content:
            if key not in original_content or enriched_content[key] != original_content[key]:
                augmented_fields.add(key)
                
        if not augmented_fields:
            return message
            
        message.add_augmentation(
            augmentation_type=self.augmentation_type,
            augmented_data=enriched_content,
            augmented_fields=augmented_fields,
            save_original=True
        )
        
        return message

class TranslationAugmentor(Augmentor):
    """Augmentor that translates message content between formats or languages."""
    def __init__(self, 
              translation_function: Callable[[Dict[str, Any], str, str], Awaitable[Dict[str, Any]]],
              source_format: str,
              target_format: str):
        """
        Initialize the translation augmentor.
        
        Args:
            translation_function: Async function that translates content
            source_format: Source format/language
            target_format: Target format/language
        Translate a message.
        
        Args:
            message: The message to translate
            
        Returns:
            Translated message
    def __init__(self, 
              summarization_function: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
              fields_to_summarize: List[str]):
        super().__init__()
        self.augmentation_type = AugmentationType.SUMMARIZATION
        self.summarization_function = summarization_function
        self.fields_to_summarize = fields_to_summarize
    
    async def augment(self, message: EnrichedMessage) -> EnrichedMessage:
        original_content = message.content.copy()
        
        to_summarize = {}
        for field in self.fields_to_summarize:
            if field in original_content:
                to_summarize[field] = original_content[field]
                
        if not to_summarize:
            return message
            
        summarized_content = await self.summarization_function(to_summarize)
        
        augmented_data = {}
        for field, value in summarized_content.items():
            summary_field = f"{field}_summary"
            augmented_data[summary_field] = value
            
        augmented_fields = set(augmented_data.keys())
        message.add_augmentation(
            augmentation_type=self.augmentation_type,
            augmented_data=augmented_data,
            augmented_fields=augmented_fields,
            save_original=False  # Original fields are preserved
        )
        
        return message

class ExpansionAugmentor(Augmentor):
    """Augmentor that expands message content with additional details."""
    def __init__(self, 
              expansion_function: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
              fields_to_expand: List[str]):
        """
        Initialize the expansion augmentor.
        
        Args:
            expansion_function: Async function that expands content
            fields_to_expand: List of field names to expand
        Expand a message with additional details.
        
        Args:
            message: The message to expand
            
        Returns:
            Expanded message
    def __init__(self, correction_function: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]):
        """
        Initialize the correction augmentor.
        
        Args:
            correction_function: Async function that corrects content
        """
        super().__init__()
        self.augmentation_type = AugmentationType.CORRECTION
        self.correction_function = correction_function
    
    async def augment(self, message: EnrichedMessage) -> EnrichedMessage:
        original_content = message.content.copy()
        
        corrected_content = await self.correction_function(original_content)
        
        augmented_fields = set()
        for key in corrected_content:
            if key not in original_content or corrected_content[key] != original_content[key]:
                augmented_fields.add(key)
                
        if not augmented_fields:
            return message
            
        message.add_augmentation(
            augmentation_type=self.augmentation_type,
            augmented_data=corrected_content,
            augmented_fields=augmented_fields,
            save_original=True
        )
        
        return message

class AnnotationAugmentor(Augmentor):
    """Augmentor that adds annotations or metadata to message content."""
    def __init__(self, annotation_function: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]):
        """
        Initialize the annotation augmentor.
        
        Args:
            annotation_function: Async function that adds annotations
        Annotate a message.
        
        Args:
            message: The message to annotate
            
        Returns:
            Annotated message
    def __init__(self, 
              filtering_function: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
              reason: str):
        super().__init__()
        self.augmentation_type = AugmentationType.FILTERING
        self.filtering_function = filtering_function
        self.reason = reason
    
    async def augment(self, message: EnrichedMessage) -> EnrichedMessage:
        original_content = message.content.copy()
        
        filtered_content = await self.filtering_function(original_content)
        
        augmented_fields = set()
        for key in original_content:
            if key not in filtered_content or filtered_content[key] != original_content[key]:
                augmented_fields.add(key)
                
        if not augmented_fields:
            return message
            
        message.add_augmentation(
            augmentation_type=self.augmentation_type,
            augmented_data=filtered_content,
            augmented_fields=augmented_fields,
            save_original=True
        )
        
        if isinstance(message.context, EnrichedContext):
            if 'filtering' not in message.context.extra_context:
                message.context.extra_context['filtering'] = {}
                
            message.context.extra_context['filtering']['reason'] = self.reason
            message.context.extra_context['filtering']['fields'] = list(augmented_fields)
            
        return message

class ReorganizationAugmentor(Augmentor):
    """Augmentor that reorganizes message content structure."""
    def __init__(self, reorganization_function: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]):
        """
        Initialize the reorganization augmentor.
        
        Args:
            reorganization_function: Async function that reorganizes content
        Reorganize a message.
        
        Args:
            message: The message to reorganize
            
        Returns:
            Reorganized message
    def __init__(self, 
              fusion_function: Callable[[Dict[str, Any], List[Dict[str, Any]]], Awaitable[Dict[str, Any]]],
              additional_sources: List[Dict[str, Any]]):
        super().__init__()
        self.augmentation_type = AugmentationType.FUSION
        self.fusion_function = fusion_function
        self.additional_sources = additional_sources
    
    async def augment(self, message: EnrichedMessage) -> EnrichedMessage:
        original_content = message.content.copy()
        
        fused_content = await self.fusion_function(original_content, self.additional_sources)
        
        augmented_fields = set()
        for key in fused_content:
            if key not in original_content or fused_content[key] != original_content[key]:
                augmented_fields.add(key)
                
        if not augmented_fields:
            return message
            
        message.add_augmentation(
            augmentation_type=self.augmentation_type,
            augmented_data=fused_content,
            augmented_fields=augmented_fields,
            save_original=True
        )
        
        if isinstance(message.context, EnrichedContext):
            if 'fusion' not in message.context.extra_context:
                message.context.extra_context['fusion'] = {}
                
            message.context.extra_context['fusion']['source_count'] = len(self.additional_sources)
            
        return message



import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, Set

from mcp_sdk.extensions.external.models import (
    EnrichedMessage, 
    EnrichedContext, 
    SourceMetadata,
    TransformationRule
)
from mcp_sdk.extensions.external.enums import (
    SourceType,
    ProtocolType,
    TransformationType,
    SecurityLevel
)

class ProtocolAdapter:
    """Base class for protocol adapters."""
    def __init__(self, 
              source_protocol: ProtocolType,
              target_protocol: ProtocolType = ProtocolType.UNKNOWN):
        """
        Initialize the protocol adapter.
        
        Args:
            source_protocol: Source protocol
            target_protocol: Target protocol
        Adapt external data to MCP message.
        
        Args:
            external_data: Data from external protocol
            
        Returns:
            MCP message or None if adaptation failed
        Adapt MCP message to external protocol.
        
        Args:
            message: MCP message
            
        Returns:
            Data in external protocol format
    def __init__(self):
        """Initialize the HTTP adapter."""
        super().__init__(ProtocolType.HTTP)
    
    async def adapt_to_mcp(self, http_data: Dict[str, Any]) -> Optional[EnrichedMessage]:
        """
        Adapt HTTP request data to MCP message.
        
        Args:
            http_data: HTTP request data
            
        Returns:
            MCP message
        Adapt MCP message to HTTP response.
        
        Args:
            message: MCP message
            
        Returns:
            HTTP response data
    def __init__(self):
        """Initialize the WebSocket adapter."""
        super().__init__(ProtocolType.WEBSOCKET)
    
    async def adapt_to_mcp(self, ws_data: Dict[str, Any]) -> Optional[EnrichedMessage]:
        """
        Adapt WebSocket message to MCP message.
        
        Args:
            ws_data: WebSocket message data
            
        Returns:
            MCP message
        Adapt MCP message to WebSocket message.
        
        Args:
            message: MCP message
            
        Returns:
            WebSocket message data
    def __init__(self):
        """Initialize the MQTT adapter."""
        super().__init__(ProtocolType.MQTT)
    
    async def adapt_to_mcp(self, mqtt_data: Dict[str, Any]) -> Optional[EnrichedMessage]:
        """
        Adapt MQTT message to MCP message.
        
        Args:
            mqtt_data: MQTT message data
            
        Returns:
            MCP message
        Adapt MCP message to MQTT message.
        
        Args:
            message: MCP message
            
        Returns:
            MQTT message data