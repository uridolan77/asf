# === FILE: mcp_sdk/extensions/external/__init__.py ===

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


# === FILE: mcp_sdk/extensions/external/enums.py ===

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
        """
        result = {}
        
        # Apply field mappings
        for src_field, tgt_field in self.field_mappings.items():
            if src_field in data:
                result[tgt_field] = data[src_field]
                
        # Apply field transformations
        for field, transform_info in self.transformers.items():
            if field in result:
                result[field] = self._transform_field(result[field], transform_info)
                
        # Apply default values for missing fields
        for field, default in self.default_values.items():
            if field not in result:
                result[field] = default
                
        return result
        
    def _transform_field(self, value: Any, transform_info: Dict[str, Any]) -> Any:
        """
        Transform a field value based on transform info.
        
        Args:
            value: The value to transform
            transform_info: Transformation instructions
            
        Returns:
            Transformed value
        """
        # This is a simplified implementation
        transform_type = transform_info.get('type', 'identity')
        
        if transform_type == 'identity':
            return value
            
        elif transform_type == 'string_format':
            format_str = transform_info.get('format', '{0}')
            return format_str.format(value)
            
        elif transform_type == 'number_scale':
            scale = transform_info.get('scale', 1.0)
            return value * scale
            
        elif transform_type == 'enum_map':
            mapping = transform_info.get('mapping', {})
            return mapping.get(value, value)
            
        elif transform_type == 'datetime_format':
            from datetime import datetime
            src_format = transform_info.get('source_format')
            tgt_format = transform_info.get('target_format')
            
            if isinstance(value, str) and src_format:
                dt = datetime.strptime(value, src_format)
            elif isinstance(value, (int, float)):
                dt = datetime.fromtimestamp(value)
            else:
                return value
                
            if tgt_format:
                return dt.strftime(tgt_format)
            else:
                return dt.isoformat()
                
        # Default: return unchanged
        return value

@dataclass
class AugmentationInfo:
    """Information about data augmentation."""
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
        # Start with base context
        result = super().to_dict()
        
        # Add enriched context fields
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
        # Start with base message
        result = super().to_dict()
        
        # Add enriched message fields
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
        # Get base reply
        reply = super().create_reply(content, message_type)
        
        # Convert to EnrichedMessage
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
        
        # Convert context to EnrichedContext if it's not already
        if not isinstance(enriched_reply.context, EnrichedContext):
            context_dict = enriched_reply.context.to_dict()
            enriched_reply.context = EnrichedContext.from_dict(context_dict)
        
        # Swap source and target metadata
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
        """
        Add augmentation to the message.
        
        Args:
            augmentation_type: Type of augmentation
            augmented_data: The augmented data
            augmented_fields: Set of field names that were augmented
            save_original: Whether to save original content
        """
        # Record the augmentation in context
        if not isinstance(self.context, EnrichedContext):
            # Convert to EnrichedContext
            context_dict = self.context.to_dict()
            self.context = EnrichedContext.from_dict(context_dict)
            
        # Save original if needed
        original_data = None
        if save_original:
            original_data = {}
            for field in augmented_fields:
                if field in self.content:
                    original_data[field] = self.content[field]
            
        # Record augmentation
        self.context.record_augmentation(
            augmentation_type=augmentation_type,
            augmented_fields=augmented_fields,
            original_data=original_data,
            confidence=1.0
        )
        
        # Store augmented data
        self.augmented_content.update(augmented_data)
        
        # Update message content with augmented data
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
        # Apply transformation to content
        transformed = rule.apply_transformation(self.content)
        
        # Store transformed data
        self.transformed_data.update(transformed)
        
        # Update content if requested
        if apply_to_content:
            self.content.update(transformed)
            
        # Add rule to context if it's enriched
        if isinstance(self.context, EnrichedContext):
            self.context.add_transformation_rule(rule)
            
            # Update augmentation info if it exists
            if self.context.augmentation_info:
                if rule.rule_id not in self.context.augmentation_info.transformations_applied:
                    self.context.augmentation_info.transformations_applied.append(rule.rule_id)
        
        return transformed


# === FILE: mcp_sdk/extensions/external/client.py ===

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
        
        # Registered sources
        self.sources: Dict[str, SourceMetadata] = {}
        
        # Transformation rules
        self.transformation_rules: Dict[str, TransformationRule] = {}
        
        # Augmentation handlers
        self.augmentation_handlers: Dict[AugmentationType, 
                                    List[Callable[[EnrichedMessage], Awaitable[EnrichedMessage]]]] = {
            aug_type: [] for aug_type in AugmentationType
        }
        
        # Source-specific handlers
        self.source_handlers: Dict[str, List[Callable[[EnrichedMessage], Awaitable[Optional[EnrichedMessage]]]]] = {}
    
    async def register_source(self, source_metadata: SourceMetadata) -> str:
        """
        Register an external source.
        
        Args:
            source_metadata: Metadata about the source
            
        Returns:
            Source ID
        """
        source_id = source_metadata.source_id
        self.sources[source_id] = source_metadata
        self.logger.info(f"Registered source: {source_id} ({source_metadata.source_type.name})")
        return source_id
    
    async def register_transformation_rule(self, rule: TransformationRule) -> str:
        """
        Register a transformation rule.
        
        Args:
            rule: The transformation rule
            
        Returns:
            Rule ID
        """
        rule_id = rule.rule_id
        self.transformation_rules[rule_id] = rule
        self.logger.info(f"Registered transformation rule: {rule_id} ({rule.transformation_type.name})")
        return rule_id
    
    async def register_augmentation_handler(self, 
                                       augmentation_type: AugmentationType,
                                       handler: Callable[[EnrichedMessage], Awaitable[EnrichedMessage]]) -> None:
        """
        Register a handler for a specific augmentation type.
        
        Args:
            augmentation_type: Type of augmentation
            handler: The handler function (async)
        """
        if augmentation_type not in self.augmentation_handlers:
            self.augmentation_handlers[augmentation_type] = []
            
        self.augmentation_handlers[augmentation_type].append(handler)
        self.logger.debug(f"Registered handler for augmentation type {augmentation_type.name}")
    
    async def register_source_handler(self, 
                                  source_id: str,
                                  handler: Callable[[EnrichedMessage], Awaitable[Optional[EnrichedMessage]]]) -> None:
        """
        Register a handler for a specific source.
        
        Args:
            source_id: Source ID
            handler: The handler function (async)
        """
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
        """
        Create a new enriched message.
        
        Args:
            message_type: Type of message
            content: Message content
            source_id: Source ID (must be registered)
            target_id: Target ID (optional)
            context_level: Level of context to include
            security_level: Security level
            extra_context: Additional context information
            **kwargs: Additional message parameters
            
        Returns:
            The created EnrichedMessage
        """
        # Get source metadata if registered
        source_metadata = None
        target_metadata = None
        
        if source_id and source_id in self.sources:
            source_metadata = self.sources[source_id]
            
        if target_id and target_id in self.sources:
            target_metadata = self.sources[target_id]
            
        # Create enriched context
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
        
        # Create the message
        message = EnrichedMessage(
            message_type=message_type,
            content=content,
            context=context,
            sender_id=self.entity_id,
            recipient_id=target_id,
            **kwargs
        )
        
        # Add to conversation history
        self._add_to_conversation(message)
        
        return message
    
    async def send_enriched_message(self, 
                               message: EnrichedMessage,
                               apply_transformations: bool = True,
                               apply_augmentations: bool = True,
                               wait_for_response: bool = False,
                               timeout: Optional[float] = None) -> Optional[EnrichedMessage]:
        """
        Send an enriched message with transformations and augmentations.
        
        Args:
            message: The message to send
            apply_transformations: Whether to apply transformation rules
            apply_augmentations: Whether to apply augmentations
            wait_for_response: Whether to wait for a response
            timeout: Timeout in seconds (None for default)
            
        Returns:
            Response message if wait_for_response is True, else None
        """
        # Apply transformations if requested
        if apply_transformations:
            message = await self._apply_transformations(message)
            
        # Apply augmentations if requested
        if apply_augmentations:
            message = await self._apply_augmentations(message)
            
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
            response = await asyncio.wait_for(response_future, timeout)
            
            # Ensure response is an EnrichedMessage
            if response and not isinstance(response, EnrichedMessage):
                # Convert to EnrichedMessage if needed
                if isinstance(response, Message):
                    # Convert Message to EnrichedMessage
                    response_dict = response.to_dict()
                    response = EnrichedMessage.from_dict(response_dict)
                    
            return response
            
        except asyncio.TimeoutError:
            # Remove the pending response
            if message.id in self.pending_responses:
                del self.pending_responses[message.id]
                
            return None
            
        finally:
            # Clean up
            if message.id in self.pending_responses:
                del self.pending_responses[message.id]
    
    async def process_external_message(self, 
                                   message: Union[Dict[str, Any], Message, EnrichedMessage],
                                   source_id: Optional[str] = None) -> Optional[EnrichedMessage]:
        """
        Process a message from an external source with enrichment and transformations.
        
        Args:
            message: The message to process
            source_id: Optional source ID for context
            
        Returns:
            Response message if any
        """
        # Convert to EnrichedMessage if needed
        if isinstance(message, dict):
            try:
                message = EnrichedMessage.from_dict(message)
            except Exception as e:
                self.logger.error(f"Error converting dictionary to message: {str(e)}")
                return None
        elif isinstance(message, Message) and not isinstance(message, EnrichedMessage):
            # Convert Message to EnrichedMessage
            message_dict = message.to_dict()
            message = EnrichedMessage.from_dict(message_dict)
            
        # Add source metadata if provided
        if source_id and source_id in self.sources:
            source_metadata = self.sources[source_id]
            
            # Ensure we have an EnrichedContext
            if not isinstance(message.context, EnrichedContext):
                context_dict = message.context.to_dict()
                message.context = EnrichedContext.from_dict(context_dict)
                
            message.context.source_metadata = source_metadata
            
        # Process through source-specific handlers if available
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
        
        # Process through normal message handlers
        return await super().process_message(message)
    
    async def query_external_source(self,
                               source_id: str,
                               query_data: Dict[str, Any],
                               apply_transformations: bool = True,
                               apply_augmentations: bool = True,
                               timeout: Optional[float] = None) -> Optional[EnrichedMessage]:
        """
        Send a query to an external source.
        
        Args:
            source_id: Source ID (must be registered)
            query_data: Query data
            apply_transformations: Whether to apply transformation rules
            apply_augmentations: Whether to apply augmentations
            timeout: Timeout in seconds (None for default)
            
        Returns:
            Response message
        """
        if source_id not in self.sources:
            self.logger.warning(f"Unknown source: {source_id}")
            return None
            
        # Get source metadata
        source_metadata = self.sources[source_id]
        
        # Create enriched context
        context = EnrichedContext(
            entity_id=self.entity_id,
            environmental_id=source_id,
            level=ContextLevel.ENHANCED,
            source_metadata=None,
            target_metadata=source_metadata
        )
        
        # Create the message
        message = EnrichedMessage(
            message_type=MessageType.QUERY,
            content=query_data,
            context=context,
            sender_id=self.entity_id,
            recipient_id=source_id
        )
        
        # Send the message
        return await self.send_enriched_message(
            message,
            apply_transformations=apply_transformations,
            apply_augmentations=apply_augmentations,
            wait_for_response=True,
            timeout=timeout
        )
    
    async def _apply_transformations(self, message: EnrichedMessage) -> EnrichedMessage:
        """
        Apply relevant transformation rules to a message.
        
        Args:
            message: The message to transform
            
        Returns:
            Transformed message
        """
        # Skip if no transformation rules
        if not self.transformation_rules:
            return message
            
        # Check source and target to determine applicable rules
        source_format = None
        target_format = None
        
        # Get source format
        if hasattr(message.context, 'source_metadata') and message.context.source_metadata:
            source_metadata = message.context.source_metadata
            if hasattr(source_metadata, 'connection_info'):
                source_format = source_metadata.connection_info.get('format')
                
        # Get target format
        if hasattr(message.context, 'target_metadata') and message.context.target_metadata:
            target_metadata = message.context.target_metadata
            if hasattr(target_metadata, 'connection_info'):
                target_format = target_metadata.connection_info.get('format')
                
        # Apply relevant rules
        for rule in self.transformation_rules.values():
            # Check if rule is applicable
            if ((rule.source_format is None or rule.source_format == source_format) and
                (rule.target_format is None or rule.target_format == target_format)):
                
                # Apply the rule
                message.apply_transformation_rule(rule)
                
        return message
    
    async def _apply_augmentations(self, message: EnrichedMessage) -> EnrichedMessage:
        """
        Apply registered augmentations to a message.
        
        Args:
            message: The message to augment
            
        Returns:
            Augmented message
        """
        # Skip if no augmentation handlers
        augmentation_types = list(self.augmentation_handlers.keys())
        if not augmentation_types:
            return message
            
        # Apply each augmentation type in sequence
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
        """Get statistics about registered sources."""
        source_types = {}
        for source_id, metadata in self.sources.items():
            source_type = metadata.source_type.name
            if source_type not in source_types:
                source_types[source_type] = 0
            source_types[source_type] += 1
            
        return {
            'total_sources': len(self.sources),
            'source_types': source_types,
            'transformation_rules': len(self.transformation_rules)
        }


# === FILE: mcp_sdk/extensions/external/augmentors.py ===

import asyncio
from typing import Dict, Any, List, Optional, Set, Callable, Awaitable, Union
import json
import time

from mcp_sdk.extensions.external.models import (
    EnrichedMessage, 
    EnrichedContext, 
    AugmentationInfo
)
from mcp_sdk.extensions.external.enums import (
    AugmentationType
)

class Augmentor:
    """Base class for message augmentors."""
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
        """
        # This should be implemented by subclasses
        return message
    
    def _record_augmentation(self, 
                           message: EnrichedMessage,
                           augmented_fields: Set[str],
                           original_data: Optional[Dict[str, Any]] = None,
                           confidence: float = 1.0) -> None:
        """
        Record an augmentation in the message.
        
        Args:
            message: The message being augmented
            augmented_fields: Set of field names that were augmented
            original_data: Original data before augmentation
            confidence: Confidence in the augmentation
        """
        message.add_augmentation(
            augmentation_type=self.augmentation_type,
            augmented_data={},  # This will be updated by the specific augmentor
            augmented_fields=augmented_fields,
            save_original=original_data is None
        )

class EnrichmentAugmentor(Augmentor):
    """Augmentor that enriches message content with additional information."""
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
        """
        Enrich a message with additional information.
        
        Args:
            message: The message to enrich
            
        Returns:
            Enriched message
        """
        # Get original content
        original_content = message.content.copy()
        
        # Apply enrichment function
        enriched_content = await self.enrichment_function(original_content)
        
        # Identify augmented fields
        augmented_fields = set()
        for key in enriched_content:
            if key not in original_content or enriched_content[key] != original_content[key]:
                augmented_fields.add(key)
                
        # Skip if no fields were actually enriched
        if not augmented_fields:
            return message
            
        # Add enrichment
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
        """
        super().__init__()
        self.augmentation_type = AugmentationType.TRANSLATION
        self.translation_function = translation_function
        self.source_format = source_format
        self.target_format = target_format
    
    async def augment(self, message: EnrichedMessage) -> EnrichedMessage:
        """
        Translate a message.
        
        Args:
            message: The message to translate
            
        Returns:
            Translated message
        """
        # Get original content
        original_content = message.content.copy()
        
        # Apply translation function
        translated_content = await self.translation_function(
            original_content, 
            self.source_format, 
            self.target_format
        )
        
        # Identify augmented fields
        augmented_fields = set()
        for key in translated_content:
            if key not in original_content or translated_content[key] != original_content[key]:
                augmented_fields.add(key)
                
        # Skip if no fields were actually translated
        if not augmented_fields:
            return message
            
        # Add translation
        message.add_augmentation(
            augmentation_type=self.augmentation_type,
            augmented_data=translated_content,
            augmented_fields=augmented_fields,
            save_original=True
        )
        
        # Store original format
        message.original_format = self.source_format
        
        return message

class SummarizationAugmentor(Augmentor):
    """Augmentor that summarizes message content."""
    def __init__(self, 
              summarization_function: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
              fields_to_summarize: List[str]):
        """
        Initialize the summarization augmentor.
        
        Args:
            summarization_function: Async function that summarizes content
            fields_to_summarize: List of field names to summarize
        """
        super().__init__()
        self.augmentation_type = AugmentationType.SUMMARIZATION
        self.summarization_function = summarization_function
        self.fields_to_summarize = fields_to_summarize
    
    async def augment(self, message: EnrichedMessage) -> EnrichedMessage:
        """
        Summarize a message.
        
        Args:
            message: The message to summarize
            
        Returns:
            Summarized message
        """
        # Get original content
        original_content = message.content.copy()
        
        # Extract fields to summarize
        to_summarize = {}
        for field in self.fields_to_summarize:
            if field in original_content:
                to_summarize[field] = original_content[field]
                
        # Skip if no fields to summarize
        if not to_summarize:
            return message
            
        # Apply summarization function
        summarized_content = await self.summarization_function(to_summarize)
        
        # Add summaries to the content with "_summary" suffix
        augmented_data = {}
        for field, value in summarized_content.items():
            summary_field = f"{field}_summary"
            augmented_data[summary_field] = value
            
        # Add summarization
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
        """
        super().__init__()
        self.augmentation_type = AugmentationType.EXPANSION
        self.expansion_function = expansion_function
        self.fields_to_expand = fields_to_expand
    
    async def augment(self, message: EnrichedMessage) -> EnrichedMessage:
        """
        Expand a message with additional details.
        
        Args:
            message: The message to expand
            
        Returns:
            Expanded message
        """
        # Get original content
        original_content = message.content.copy()
        
        # Extract fields to expand
        to_expand = {}
        for field in self.fields_to_expand:
            if field in original_content:
                to_expand[field] = original_content[field]
                
        # Skip if no fields to expand
        if not to_expand:
            return message
            
        # Apply expansion function
        expanded_content = await self.expansion_function(to_expand)
        
        # Identify augmented fields - just the new fields that aren't in original
        augmented_fields = set()
        augmented_data = {}
        for key, value in expanded_content.items():
            if key not in original_content:
                augmented_fields.add(key)
                augmented_data[key] = value
                
        # Skip if no fields were actually expanded
        if not augmented_fields:
            return message
            
        # Add expansion
        message.add_augmentation(
            augmentation_type=self.augmentation_type,
            augmented_data=augmented_data,
            augmented_fields=augmented_fields,
            save_original=False  # Original fields are preserved
        )
        
        return message

class CorrectionAugmentor(Augmentor):
    """Augmentor that corrects errors or inconsistencies in message content."""
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
        """
        Correct a message.
        
        Args:
            message: The message to correct
            
        Returns:
            Corrected message
        """
        # Get original content
        original_content = message.content.copy()
        
        # Apply correction function
        corrected_content = await self.correction_function(original_content)
        
        # Identify augmented fields
        augmented_fields = set()
        for key in corrected_content:
            if key not in original_content or corrected_content[key] != original_content[key]:
                augmented_fields.add(key)
                
        # Skip if no fields were actually corrected
        if not augmented_fields:
            return message
            
        # Add correction
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
        """
        super().__init__()
        self.augmentation_type = AugmentationType.ANNOTATION
        self.annotation_function = annotation_function
    
    async def augment(self, message: EnrichedMessage) -> EnrichedMessage:
        """
        Annotate a message.
        
        Args:
            message: The message to annotate
            
        Returns:
            Annotated message
        """
        # Get original content
        original_content = message.content.copy()
        
        # Apply annotation function
        annotations = await self.annotation_function(original_content)
        
        # Add annotations with "_annotation" suffix
        augmented_data = {}
        for field, annotation in annotations.items():
            annotation_field = f"{field}_annotation"
            augmented_data[annotation_field] = annotation
            
        # Skip if no annotations were generated
        if not augmented_data:
            return message
            
        # Add annotations
        augmented_fields = set(augmented_data.keys())
        message.add_augmentation(
            augmentation_type=self.augmentation_type,
            augmented_data=augmented_data,
            augmented_fields=augmented_fields,
            save_original=False  # Original fields are preserved
        )
        
        return message

class FilteringAugmentor(Augmentor):
    """Augmentor that filters or redacts information from message content."""
    def __init__(self, 
              filtering_function: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
              reason: str):
        """
        Initialize the filtering augmentor.
        
        Args:
            filtering_function: Async function that filters content
            reason: Reason for filtering
        """
        super().__init__()
        self.augmentation_type = AugmentationType.FILTERING
        self.filtering_function = filtering_function
        self.reason = reason
    
    async def augment(self, message: EnrichedMessage) -> EnrichedMessage:
        """
        Filter or redact information from a message.
        
        Args:
            message: The message to filter
            
        Returns:
            Filtered message
        """
        # Get original content
        original_content = message.content.copy()
        
        # Apply filtering function
        filtered_content = await self.filtering_function(original_content)
        
        # Identify filtered fields
        augmented_fields = set()
        for key in original_content:
            if key not in filtered_content or filtered_content[key] != original_content[key]:
                augmented_fields.add(key)
                
        # Skip if no fields were actually filtered
        if not augmented_fields:
            return message
            
        # Add filtering with reason
        message.add_augmentation(
            augmentation_type=self.augmentation_type,
            augmented_data=filtered_content,
            augmented_fields=augmented_fields,
            save_original=True
        )
        
        # Add reason to extra context
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
        """
        super().__init__()
        self.augmentation_type = AugmentationType.REORGANIZATION
        self.reorganization_function = reorganization_function
    
    async def augment(self, message: EnrichedMessage) -> EnrichedMessage:
        """
        Reorganize a message.
        
        Args:
            message: The message to reorganize
            
        Returns:
            Reorganized message
        """
        # Get original content
        original_content = message.content.copy()
        
        # Apply reorganization function
        reorganized_content = await self.reorganization_function(original_content)
        
        # Skip if content didn't change
        if reorganized_content == original_content:
            return message
            
        # Add reorganization
        augmented_fields = set(reorganized_content.keys())
        message.add_augmentation(
            augmentation_type=self.augmentation_type,
            augmented_data=reorganized_content,
            augmented_fields=augmented_fields,
            save_original=True
        )
        
        return message

class FusionAugmentor(Augmentor):
    """Augmentor that combines information from multiple sources."""
    def __init__(self, 
              fusion_function: Callable[[Dict[str, Any], List[Dict[str, Any]]], Awaitable[Dict[str, Any]]],
              additional_sources: List[Dict[str, Any]]):
        """
        Initialize the fusion augmentor.
        
        Args:
            fusion_function: Async function that fuses content from multiple sources
            additional_sources: List of additional data sources to fuse
        """
        super().__init__()
        self.augmentation_type = AugmentationType.FUSION
        self.fusion_function = fusion_function
        self.additional_sources = additional_sources
    
    async def augment(self, message: EnrichedMessage) -> EnrichedMessage:
        """
        Fuse information from multiple sources into a message.
        
        Args:
            message: The message to augment
            
        Returns:
            Fused message
        """
        # Get original content
        original_content = message.content.copy()
        
        # Apply fusion function
        fused_content = await self.fusion_function(original_content, self.additional_sources)
        
        # Identify augmented fields
        augmented_fields = set()
        for key in fused_content:
            if key not in original_content or fused_content[key] != original_content[key]:
                augmented_fields.add(key)
                
        # Skip if no fields were actually fused
        if not augmented_fields:
            return message
            
        # Add fusion
        message.add_augmentation(
            augmentation_type=self.augmentation_type,
            augmented_data=fused_content,
            augmented_fields=augmented_fields,
            save_original=True
        )
        
        # Add source information to extra context
        if isinstance(message.context, EnrichedContext):
            if 'fusion' not in message.context.extra_context:
                message.context.extra_context['fusion'] = {}
                
            message.context.extra_context['fusion']['source_count'] = len(self.additional_sources)
            
        return message


# === FILE: mcp_sdk/extensions/external/adapters.py ===

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
        """
        self.source_protocol = source_protocol
        self.target_protocol = target_protocol
    
    async def adapt_to_mcp(self, external_data: Any) -> Optional[EnrichedMessage]:
        """
        Adapt external data to MCP message.
        
        Args:
            external_data: Data from external protocol
            
        Returns:
            MCP message or None if adaptation failed
        """
        # This should be implemented by subclasses
        return None
    
    async def adapt_from_mcp(self, message: EnrichedMessage) -> Optional[Any]:
        """
        Adapt MCP message to external protocol.
        
        Args:
            message: MCP message
            
        Returns:
            Data in external protocol format
        """
        # This should be implemented by subclasses
        return None

class HTTPAdapter(ProtocolAdapter):
    """Adapter for HTTP protocol."""
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
        """
        from mcp_sdk.enums import MessageType, ContextLevel
        from mcp_sdk.extensions.external.models import EnrichedMessage, EnrichedContext, SourceMetadata
        from mcp_sdk.extensions.external.enums import SourceType, ProtocolType, SecurityLevel
        
        # Extract HTTP components
        method = http_data.get('method', 'GET')
        url = http_data.get('url', '')
        headers = http_data.get('headers', {})
        body = http_data.get('body', {})
        
        # Create source metadata
        source_metadata = SourceMetadata(
            source_id=http_data.get('source_id', f"http_{int(time.time())}"),
            source_type=SourceType.API,
            protocol=ProtocolType.HTTP,
            location=url,
            security_level=SecurityLevel.STANDARD,
            connection_info={
                'method': method,
                'headers': headers
            }
        )
        
        # Determine message type based on HTTP method
        message_type = MessageType.QUERY
        if method == 'POST':
            message_type = MessageType.ACTION
        elif method == 'PUT':
            message_type = MessageType.UPDATE
        elif method == 'DELETE':
            message_type = MessageType.SYSTEM
            
        # Create enriched context
        context = EnrichedContext(
            level=ContextLevel.ENHANCED,
            source_metadata=source_metadata,
            extra_context={
                'http': {
                    'method': method,
                    'url': url,
                    'headers': headers
                }
            }
        )
        
        # Create enriched message
        message = EnrichedMessage(
            message_type=message_type,
            content=body,
            context=context,
            original_format='http'
        )
        
        return message
    
    async def adapt_from_mcp(self, message: EnrichedMessage) -> Optional[Dict[str, Any]]:
        """
        Adapt MCP message to HTTP response.
        
        Args:
            message: MCP message
            
        Returns:
            HTTP response data
        """
        # Create HTTP response
        http_response = {
            'status': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': message.content
        }
        
        # Adjust status based on message type
        if message.message_type.name == 'ERROR':
            http_response['status'] = 400
            
        # Add headers based on context
        if isinstance(message.context, EnrichedContext) and message.context.extra_context:
            http_headers = message.context.extra_context.get('http_response_headers', {})
            http_response['headers'].update(http_headers)
            
        return http_response

class WebSocketAdapter(ProtocolAdapter):
    """Adapter for WebSocket protocol."""
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
        """
        from mcp_sdk.enums import MessageType, ContextLevel
        from mcp_sdk.extensions.external.models import EnrichedMessage, EnrichedContext, SourceMetadata
        from mcp_sdk.extensions.external.enums import SourceType, ProtocolType, SecurityLevel
        
        # Extract WebSocket components
        client_id = ws_data.get('client_id', f"ws_{int(time.time())}")
        message_content = ws_data.get('message', {})
        event_type = ws_data.get('event_type', 'message')
        
        # Create source metadata
        source_metadata = SourceMetadata(
            source_id=client_id,
            source_type=SourceType.SERVICE,
            protocol=ProtocolType.WEBSOCKET,
            security_level=SecurityLevel.STANDARD,
            connection_info={
                'event_type': event_type
            }
        )
        
        # Determine message type based on event type
        message_type = MessageType.QUERY
        if event_type == 'update':
            message_type = MessageType.UPDATE
        elif event_type == 'command':
            message_type = MessageType.ACTION
            
        # Create enriched context
        context = EnrichedContext(
            level=ContextLevel.ENHANCED,
            source_metadata=source_metadata,
            extra_context={
                'websocket': {
                    'client_id': client_id,
                    'event_type': event_type
                }
            }
        )
        
        # Create enriched message
        message = EnrichedMessage(
            message_type=message_type,
            content=message_content,
            context=context,
            original_format='websocket'
        )
        
        return message
    
    async def adapt_from_mcp(self, message: EnrichedMessage) -> Optional[Dict[str, Any]]:
        """
        Adapt MCP message to WebSocket message.
        
        Args:
            message: MCP message
            
        Returns:
            WebSocket message data
        """
        # Create WebSocket message
        ws_message = {
            'event': 'message',
            'data': message.content
        }
        
        # Adjust event based on message type
        if message.message_type.name == 'RESPONSE':
            ws_message['event'] = 'response'
        elif message.message_type.name == 'ERROR':
            ws_message['event'] = 'error'
        elif message.message_type.name == 'UPDATE':
            ws_message['event'] = 'update'
            
        return ws_message

class MQTTAdapter(ProtocolAdapter):
    """Adapter for MQTT protocol."""
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
        """
        from mcp_sdk.enums import MessageType, ContextLevel
        from mcp_sdk.extensions.external.models import EnrichedMessage, EnrichedContext, SourceMetadata
        from mcp_sdk.extensions.external.enums import SourceType, ProtocolType, SecurityLevel
        
        # Extract MQTT components
        topic = mqtt_data.get('topic', '')
        payload = mqtt_data.get('payload', {})
        qos = mqtt_data.get('qos', 0)
        
        # Create source metadata
        source_metadata = SourceMetadata(
            source_id=f"mqtt_{topic.replace('/', '_')}",
            source_type=SourceType.STREAM,
            protocol=ProtocolType.MQTT,
            location=topic,
            security_level=SecurityLevel.STANDARD,
            connection_info={
                'topic': topic,
                'qos': qos
            }
        )
        
        # Determine message type based on topic pattern
        message_type = MessageType.OBSERVATION
        if topic.endswith('/query'):
            message_type = MessageType.QUERY
        elif topic.endswith('/action'):
            message_type = MessageType.ACTION
        elif topic.endswith('/update'):
            message_type = MessageType.UPDATE
            
        # Create enriched context
        context = EnrichedContext(
            level=ContextLevel.ENHANCED,
            source_metadata=source_metadata,
            extra_context={
                'mqtt': {
                    'topic': topic,
                    'qos': qos
                }
            }
        )
        
        # Create enriched message
        message = EnrichedMessage(
            message_type=message_type,
            content=payload,
            context=context,
            original_format='mqtt'
        )
        
        return message
    
    async def adapt_from_mcp(self, message: EnrichedMessage) -> Optional[Dict[str, Any]]:
        """
        Adapt MCP message to MQTT message.
        
        Args:
            message: MCP message
            
        Returns:
            MQTT message data
        """
        # Determine topic from context
        topic = 'mcp/default'
        qos = 1
        
        if isinstance(message.context, EnrichedContext) and message.context.extra_context:
            mqtt_info = message.context.extra_context.get('mqtt', {})
            reply_topic = mqtt_info.get('reply_topic')
            
            if reply_topic:
                topic = reply_topic
            elif 'topic' in mqtt_info:
                base_topic = mqtt_info['topic'].split('/')
                if len(base_topic) > 1:
                    topic = '/'.join(base_topic[:-1]) + '/response'
                else:
                    topic = base_topic[0] + '/response'
                    
            qos = mqtt_info.get('qos', 1)
            
        # Create MQTT message
        mqtt_message = {
            'topic': topic,
            'payload': message.content,
            'qos': qos
        }
            
        return mqtt_message