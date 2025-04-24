"""
Core data models for the Conexus LLM Gateway.

This module defines the fundamental data structures used throughout 
the LLM Gateway system, including requests, responses, and configurations.
"""

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Type, AsyncGenerator, cast, Literal

from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict

# --- Base Model with Common Config ---
logger = logging.getLogger(__name__)

class BaseGatewayModel(BaseModel):
    """Base model for common configuration like assignment validation."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='allow',  # Allow extra fields for forward compatibility
    )

# --- Enums ---

class FinishReason(str, Enum):
    """Reasons why a generation completed."""
    STOP = "stop"  # Natural stop
    LENGTH = "length"  # Hit max tokens
    TOOL_CALLS = "tool_calls"  # Stopped to make tool calls
    CONTENT_FILTERED = "content_filtered"  # Content was filtered
    ERROR = "error"  # Error occurred
    TIMEOUT = "timeout"  # Request timed out
    UNKNOWN = "unknown"  # Unknown reason

class ComplianceStatus(str, Enum):
    """Status of compliance verification."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_CHECKED = "not_checked"
    ERROR = "error"

class ViolationSeverity(str, Enum):
    """Severity levels for compliance violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorLevel(str, Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

# --- Role and Content Type Enums ---

class MCPRole(str, Enum):
    """Roles in an MCP conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"  # Represents the model's response
    TOOL = "tool"  # Represents the result of a tool execution

class MCPContentType(str, Enum):
    """Content types for MCP messages."""
    TEXT = "text"
    CODE = "code" 
    JSON = "json"
    MARKDOWN = "markdown"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    TOOL_USE = "tool_use"  # Model requests to use a tool
    TOOL_RESULT = "tool_result"  # Result provided back to the model

class MCPStopReason(str, Enum):
    """Stop reasons potentially alignable with MCP standard."""
    END_TURN = "endTurn"  # Natural end of model response
    TOOL_USE = "tool_use"  # Model stopped to use a tool
    MAX_TOKENS = "maxTokens"  # Output length limit reached
    STOP_SEQUENCE = "stopSequence"  # Stop sequence detected
    CONTENT_FILTERED = "content_filtered"  # Provider filtered content
    ERROR = "error"  # Provider error

# --- Content & Message Models ---

class ContentItem(BaseGatewayModel):
    """A single piece of potentially multimodal content."""
    model_config = ConfigDict(frozen=True)

    type: MCPContentType = Field(..., description="Type of content.")
    data: Dict[str, Any] = Field(..., description="Structured data representing the content based on its type.")
    mime_type: Optional[str] = Field(None, description="MIME type for the content, e.g., 'image/jpeg'")
    text_content: Optional[str] = Field(None, description="Convenience field for accessing text content")

    @model_validator(mode='after')
    def set_text_content(self) -> 'ContentItem':
        """Set text_content for TEXT type to simplify access."""
        if self.type == MCPContentType.TEXT and not self.text_content:
            object.__setattr__(self, 'text_content', self.data.get("text"))
        return self

    # Convenience methods
    @classmethod
    def from_text(cls, text: str) -> 'ContentItem':
        """Create a text content item."""
        return cls(type=MCPContentType.TEXT, data={"text": text}, text_content=text)

    @classmethod
    def from_image_base64(cls, base64_data: str, mime_type: str = "image/jpeg") -> 'ContentItem':
        """Create an image content item from base64 data."""
        return cls(
            type=MCPContentType.IMAGE,
            data={"image": {"source": {"type": "base64", "data": base64_data}}},
            mime_type=mime_type
        )

    @classmethod
    def from_image_url(cls, url: str, mime_type: Optional[str] = None) -> 'ContentItem':
        """Create an image content item from a URL."""
        return cls(
            type=MCPContentType.IMAGE,
            data={"image": {"source": {"type": "url", "url": url}}},
            mime_type=mime_type
        )

class Message(BaseGatewayModel):
    """Represents a message in the conversation."""
    model_config = ConfigDict(frozen=True)

    role: MCPRole = Field(..., description="The role of the message author.")
    content: List[ContentItem] = Field(..., description="A list of content items comprising the message.")
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this message instance.")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata.")

class ConversationTurn(BaseGatewayModel):
    """A single turn in a conversation history."""
    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    content: Union[str, List[ContentItem], Dict[str, Any]]
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Convenience methods
    @classmethod
    def from_text(cls, role: str, text: str) -> 'ConversationTurn':
        """Create a turn from plain text."""
        return cls(role=role, content=text)
    
    @classmethod
    def from_message(cls, message: Message) -> 'ConversationTurn':
        """Create a turn from a Message object."""
        return cls(
            role=message.role.value,
            content=message.content,
            timestamp=message.timestamp,
            metadata=message.metadata
        )

# --- Tool Definition and Usage Models ---

class ToolParameterSchema(BaseGatewayModel):
    """Parameter schema for a tool."""
    model_config = ConfigDict(frozen=True)

    type: str = Field("object", description="Typically 'object' for tool parameters.")
    properties: Dict[str, Dict[str, Any]] = Field(..., description="Schema for each parameter.")
    required: Optional[List[str]] = Field(None, description="List of required parameter names.")

class ToolFunction(BaseGatewayModel):
    """Definition of a tool function."""
    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Name of the tool.")
    description: Optional[str] = Field(None, description="Description of what the tool does.")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Schema for input parameters.")

class ToolDefinition(BaseGatewayModel):
    """Definition of a tool."""
    model_config = ConfigDict(frozen=True)

    function: ToolFunction = Field(..., description="Function definition for the tool.")

    @classmethod
    def from_function(cls, name: str, description: Optional[str] = None, 
                     parameters: Optional[Dict[str, Any]] = None) -> 'ToolDefinition':
        """Create a tool definition from function details."""
        function = ToolFunction(name=name, description=description, parameters=parameters)
        return cls(function=function)

class ToolUseRequest(BaseGatewayModel):
    """Represents a request from the model to use a tool."""
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this tool use request.")
    type: str = Field("function", description="Type of tool use (typically 'function').")
    function: ToolFunction = Field(..., description="Function details including name and parameters.")

class ToolResult(BaseGatewayModel):
    """Represents the result of executing a tool."""
    model_config = ConfigDict(frozen=True)

    tool_call_id: str = Field(..., description="ID matching the corresponding ToolUseRequest.")
    output: Union[str, Dict[str, Any]] = Field(..., description="Result of the tool execution.")
    is_error: bool = Field(False, description="Indicates if the execution resulted in an error.")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Details about any error.")

# --- Performance and Usage Models ---

class UsageStats(BaseGatewayModel):
    """Token usage statistics."""
    model_config = ConfigDict(frozen=True)
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @model_validator(mode='after')
    def calculate_total(self) -> 'UsageStats':
        """Calculate total tokens if not provided."""
        if self.total_tokens == 0 and (self.prompt_tokens > 0 or self.completion_tokens > 0):
            object.__setattr__(self, 'total_tokens', self.prompt_tokens + self.completion_tokens)
        return self

class PerformanceMetrics(BaseGatewayModel):
    """Performance metrics."""
    model_config = ConfigDict(frozen=True)
    
    total_duration_ms: Optional[float] = None
    llm_latency_ms: Optional[float] = None
    pre_processing_duration_ms: Optional[float] = None
    post_processing_duration_ms: Optional[float] = None
    compliance_check_duration_ms: Optional[float] = None
    gateway_overhead_ms: Optional[float] = None

    @model_validator(mode='after')
    def calculate_overhead(self) -> 'PerformanceMetrics':
        """Calculate gateway overhead if possible."""
        if (self.total_duration_ms is not None and self.llm_latency_ms is not None and 
            self.gateway_overhead_ms is None):
            object.__setattr__(self, 'gateway_overhead_ms', 
                             max(0, self.total_duration_ms - self.llm_latency_ms))
        return self

# --- Error Handling Models ---

class ErrorDetails(BaseGatewayModel):
    """Structured error information."""
    model_config = ConfigDict(frozen=True)
    
    code: str
    message: str
    level: ErrorLevel = ErrorLevel.ERROR
    provider_error_details: Optional[Dict[str, Any]] = None
    intervention_error_details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    retryable: bool = False
    retry_after_seconds: Optional[int] = None
    stage: Optional[Literal["pre_intervention", "provider_call", "post_intervention", 
                           "stream_intervention", "manager", "factory", "client"]] = None
    source: Optional[str] = None  # Component that generated the error

# --- Configuration Models ---

class ExtensionPoints(BaseGatewayModel):
    """Container for forward compatibility."""
    provider_extensions: Dict[str, Any] = Field(default_factory=dict)
    experimental_features: Dict[str, Any] = Field(default_factory=dict)

class RequestMetadata(BaseGatewayModel):
    """Observability metadata."""
    client_id: Optional[str] = None
    application_name: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_path: Optional[str] = None
    request_method: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    correlation_id: Optional[str] = None  # Trace ID

class CacheMetadata(BaseGatewayModel):
    """Information about cache hits."""
    model_config = ConfigDict(frozen=True)
    
    cache_hit: bool = False
    cache_key: Optional[str] = None
    ttl_seconds_remaining: Optional[int] = None
    stored_at: Optional[datetime] = None

class CacheConfig(BaseGatewayModel):
    """Configuration for LLM response caching."""
    model_config = ConfigDict(frozen=True)
    
    enabled: bool = True
    cache_ttl_seconds: int = Field(3600, ge=0)  # Default 1 hour
    use_redis: bool = False
    redis_url: Optional[str] = None
    exclude_models: List[str] = Field(default_factory=list)
    exclude_providers: List[str] = Field(default_factory=list)
    cache_embeddings: bool = True

class QuotaInfo(BaseGatewayModel):
    """Rate limit quota info."""
    model_config = ConfigDict(frozen=True)
    
    requests_remaining: Optional[int] = None
    tokens_remaining: Optional[int] = None
    reset_at: Optional[datetime] = None

# --- Resource Management Models ---

class ResourceType(str, Enum):
    """Types of resources that can be managed."""
    HTTP_CONNECTION = "http_connection"
    GRPC_CONNECTION = "grpc_connection"
    MCP_SESSION = "mcp_session"
    MODEL_INSTANCE = "model_instance"
    REDIS_CONNECTION = "redis_connection"
    DATABASE_CONNECTION = "database_connection"
    CUSTOM = "custom"

class ResourceStatus(str, Enum):
    """Status of a managed resource."""
    AVAILABLE = "available"
    IN_USE = "in_use"
    UNAVAILABLE = "unavailable"
    CREATING = "creating"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"

class ResourceMetrics(BaseGatewayModel):
    """Metrics collected for a resource."""
    model_config = ConfigDict(frozen=True)
    
    creation_time: datetime = Field(default_factory=datetime.utcnow)
    last_used_time: Optional[datetime] = None
    total_uses: int = 0
    total_errors: int = 0
    total_timeouts: int = 0
    last_error_time: Optional[datetime] = None
    avg_response_time_ms: Optional[float] = None
    health_score: float = 1.0  # 0.0 (unhealthy) to 1.0 (healthy)

class ResourceLimits(BaseGatewayModel):
    """Limits for resource pools."""
    max_pool_size: int = 10
    min_pool_size: int = 1
    max_idle_time_seconds: int = 300  # 5 minutes
    max_lifetime_seconds: Optional[int] = None  # None means no limit
    circuit_breaker_threshold: int = 5  # Number of errors before circuit breaks
    circuit_breaker_timeout_seconds: int = 30  # Time before circuit half-opens
    acquisition_timeout_seconds: float = 30.0  # Max time to wait for resource

class ResourcePoolConfig(BaseGatewayModel):
    """Configuration for a resource pool."""
    
    resource_type: ResourceType
    pool_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    provider_id: Optional[str] = None  # Which provider this pool belongs to, if any
    limits: ResourceLimits = Field(default_factory=ResourceLimits)
    creation_kwargs: Dict[str, Any] = Field(default_factory=dict)  # Args for resource creation
    validation_fn: Optional[str] = None  # Name of function to validate resources
    initialization_fn: Optional[str] = None  # Name of function to initialize resources
    cleanup_fn: Optional[str] = None  # Name of function to cleanup resources
    enable_metrics: bool = True
    enable_health_checks: bool = True
    health_check_interval_seconds: int = 60
    adaptive_sizing: bool = False  # Whether pool should resize based on load

class ResourcePoolStats(BaseGatewayModel):
    """Statistics for a resource pool."""
    model_config = ConfigDict(frozen=True)
    
    pool_id: str
    resource_type: ResourceType
    provider_id: Optional[str] = None
    current_size: int = 0
    available_count: int = 0
    in_use_count: int = 0
    waiting_requests: int = 0
    creation_errors: int = 0
    circuit_state: str = "closed"  # closed, open, half-open
    error_rate: float = 0.0  # 0.0 to 1.0
    avg_acquisition_time_ms: Optional[float] = None
    avg_wait_time_ms: Optional[float] = None
    last_resize_time: Optional[datetime] = None
    total_acquisitions: int = 0
    total_timeouts: int = 0

class ResourceInfo(BaseGatewayModel):
    """Information about a specific resource in a pool."""
    model_config = ConfigDict(frozen=True)
    
    resource_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pool_id: str
    resource_type: ResourceType
    provider_id: Optional[str] = None
    status: ResourceStatus = ResourceStatus.CREATING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    last_validated_at: Optional[datetime] = None
    metrics: ResourceMetrics = Field(default_factory=ResourceMetrics)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_count: int = 0
    current_owner: Optional[str] = None  # ID of current request holding this resource

# --- Compliance and Guardrails Models ---

class Violation(BaseGatewayModel):
    """Compliance violation details."""
    model_config = ConfigDict(frozen=True)
    
    rule_id: Optional[str] = None
    framework_id: Optional[str] = None
    severity: ViolationSeverity = ViolationSeverity.MEDIUM
    type: str = "Unknown"
    description: str
    affected_text: Optional[str] = None
    affected_elements: Optional[List[str]] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    handler_action: Optional[str] = None

class ComplianceResult(BaseGatewayModel):
    """Aggregated compliance check results."""
    model_config = ConfigDict(frozen=True)
    
    status: ComplianceStatus = ComplianceStatus.NOT_CHECKED
    violations: List[Violation] = Field(default_factory=list)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    proof_id: Optional[str] = None
    checked_at: Optional[datetime] = None

class GuardrailConfig(BaseGatewayModel):
    """Guardrail settings."""
    
    content_filter_categories: Dict[str, bool] = Field(
        default_factory=lambda: {
            "hate": True, 
            "sexual": True, 
            "violence": True, 
            "self_harm": True
        }
    )
    content_filter_severity_threshold: ViolationSeverity = ViolationSeverity.MEDIUM
    detect_prompt_injection: bool = True
    detect_jailbreak: bool = True
    pii_detection_level: Optional[str] = "medium"
    custom_topic_blocklist: Optional[List[str]] = None
    custom_term_blocklist: Optional[List[str]] = None

# --- Intervention Pipeline Models ---

class InterventionData(BaseGatewayModel):
    """Flexible container for intervention data sharing."""
    
    data: Dict[str, Any] = Field(default_factory=dict)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the data dictionary."""
        self.data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the data dictionary."""
        return self.data.get(key, default)

class InterventionConfig(BaseGatewayModel):
    """Configuration for intervention pipeline execution."""
    
    enabled_pre_interventions: List[str] = Field(default_factory=list)
    enabled_post_interventions: List[str] = Field(default_factory=list)
    total_intervention_timeout_ms: int = Field(10000, gt=0)
    fail_open: bool = Field(False)

class InterventionContext(BaseGatewayModel):
    """Mutable context object passed through the intervention pipeline."""
    # Allow mutation during pipeline execution
    
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    user_info: Dict[str, Any] = Field(default_factory=dict)
    timestamp_start: datetime = Field(default_factory=datetime.utcnow)
    target_domain: Optional[str] = None
    required_compliance_frameworks: List[str] = Field(default_factory=list)
    compliance_mode: str = Field("strict")
    conversation_history: List[ConversationTurn] = Field(default_factory=list)
    intervention_data: InterventionData = Field(default_factory=InterventionData)
    trace_id: Optional[str] = None
    guardrail_config: Optional[GuardrailConfig] = None
    request_metadata: Optional[RequestMetadata] = None
    quota_info: Optional[QuotaInfo] = None
    intervention_config: InterventionConfig = Field(default_factory=InterventionConfig)

    # Convenience methods
    def add_conversation_turn(self, role: str, content: Union[str, List[ContentItem]]) -> None:
        """Add a new turn to the conversation history."""
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.utcnow()
        )
        self.conversation_history.append(turn)

# --- LLM Request and Response Models ---

class LLMConfig(BaseGatewayModel):
    """Configuration parameters requested for the LLM call."""
    
    model_identifier: str = Field(..., description="Requested model identifier.")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    stop_sequences: Optional[List[str]] = Field(None)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    timeout_seconds: Optional[float] = None
    system_prompt: Optional[str] = None

    model_config = ConfigDict(extra='allow')  # Allow provider-specific parameters

class LLMRequest(BaseGatewayModel):
    """Input request object for the LLM Gateway."""
    
    version: str = "1.0"
    prompt_content: Union[str, List[ContentItem]] = Field(..., description="Input prompt.")
    config: LLMConfig = Field(..., description="Requested LLM configuration.")
    initial_context: InterventionContext = Field(default_factory=InterventionContext)
    stream: bool = Field(False)
    tools: Optional[List[ToolDefinition]] = Field(None, description="Available tools for the model.")
    extensions: ExtensionPoints = Field(default_factory=ExtensionPoints)
    provider_id: Optional[str] = None
    model_identifier: Optional[str] = None  # For backward compatibility

class LLMResponse(BaseGatewayModel):
    """Output response object from the LLM Gateway."""
    model_config = ConfigDict(frozen=True)
    
    version: str = "1.0"
    request_id: str  # Matches LLMRequest's initial_context.request_id
    generated_content: Optional[Union[str, List[ContentItem]]] = None
    finish_reason: Optional[FinishReason] = None
    tool_use_requests: Optional[List[ToolUseRequest]] = Field(None, description="Tool use requested by the model.")
    usage: Optional[UsageStats] = None
    compliance_result: Optional[ComplianceResult] = None
    final_context: InterventionContext
    error_details: Optional[ErrorDetails] = None
    timestamp_end: datetime = Field(default_factory=datetime.utcnow)
    performance_metrics: Optional[PerformanceMetrics] = None
    cache_metadata: Optional[CacheMetadata] = None
    quota_info: Optional[QuotaInfo] = None
    extensions: ExtensionPoints = Field(default_factory=ExtensionPoints)
    mcp_metadata: Optional[Dict[str, Any]] = None
    provider_id: Optional[str] = None  # For backward compatibility
    model_id: Optional[str] = None  # For backward compatibility
    
class StreamChunk(BaseGatewayModel):
    """A chunk of a streaming response."""
    model_config = ConfigDict(frozen=True)
    
    chunk_id: int
    request_id: str
    delta_text: Optional[str] = None
    delta_content_items: Optional[List[ContentItem]] = None
    delta_tool_calls: Optional[List[ToolUseRequest]] = None
    finish_reason: Optional[FinishReason] = None
    usage_update: Optional[UsageStats] = None
    provider_specific_data: Optional[Dict[str, Any]] = None
    error: Optional[ErrorDetails] = None  # For backward compatibility

# --- Batch Processing Models ---

class BatchLLMRequest(BaseGatewayModel):
    """Container for batch LLM requests."""
    
    version: str = "1.0"
    batch_id: str = Field(default_factory=lambda: f"batch_{uuid.uuid4()}")
    requests: List[LLMRequest] = Field(..., min_length=1)
    parallel: bool = True

class BatchLLMResponse(BaseGatewayModel):
    """Container for batch LLM responses."""
    model_config = ConfigDict(frozen=True)
    
    version: str = "1.0"
    batch_id: str
    responses: List[LLMResponse]  # Order matches requests
    aggregated_usage: Optional[UsageStats] = None
    total_duration_ms: Optional[float] = None
    error_count: int = Field(0)
    success_count: int = Field(0)

    @model_validator(mode='after')
    def calculate_aggregated_stats(self) -> 'BatchLLMResponse':
        """Calculate aggregated usage and counts if not provided."""
        # Calculate usage
        if self.aggregated_usage is None:
            total_prompt = sum(r.usage.prompt_tokens for r in self.responses if r.usage)
            total_completion = sum(r.usage.completion_tokens for r in self.responses if r.usage)
            if total_prompt > 0 or total_completion > 0:
                object.__setattr__(self, 'aggregated_usage', UsageStats(
                    prompt_tokens=total_prompt, 
                    completion_tokens=total_completion
                ))
        
        # Calculate success/error counts
        error_count = sum(1 for r in self.responses if r.error_details is not None)
        success_count = len(self.responses) - error_count
        
        object.__setattr__(self, 'error_count', error_count)
        object.__setattr__(self, 'success_count', success_count)
        
        return self

# --- Provider Configuration Models ---

class ProviderModelInfo(BaseGatewayModel):
    """Capabilities/limits for a specific provider model."""
    model_config = ConfigDict(frozen=True)
    
    model_name: str
    context_window: int
    max_output_tokens: Optional[int] = None
    supports_streaming: bool = True
    supports_tools: bool = True
    input_modalities: List[MCPContentType] = Field(default_factory=lambda: [MCPContentType.TEXT])
    output_modalities: List[MCPContentType] = Field(default_factory=lambda: [MCPContentType.TEXT])

class ProviderConfig(BaseGatewayModel):
    """Configuration for a specific LLM provider."""
    model_config = ConfigDict(frozen=True)
    
    provider_id: str
    provider_type: str
    display_name: Optional[str] = None
    connection_params: Dict[str, Any] = Field(default_factory=dict)
    models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    default_timeout_seconds: float = 60.0
    additional_config: Dict[str, Any] = Field(default_factory=dict)

class GatewayConfig(BaseGatewayModel):
    """Overall static configuration for the LLM Gateway service."""
    model_config = ConfigDict(frozen=True)
    
    gateway_id: str
    default_provider: str = "openai"
    default_model_identifier: Optional[str] = None
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    default_timeout_seconds: float = 60.0
    allowed_providers: List[str] = Field(default_factory=list)
    default_intervention_config: InterventionConfig = Field(default_factory=InterventionConfig)
    caching_enabled: bool = True
    cache_default_ttl_seconds: int = Field(3600, ge=0)
    default_compliance_mode: str = Field("strict", pattern="^(strict|audit|permissive)$")
    logging_level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    additional_config: Dict[str, Any] = Field(default_factory=dict)
    mcp_api_endpoint: Optional[str] = Field(None, description="Global endpoint for MCP providers if applicable.")
    mcp_enabled_models: List[str] = Field(default_factory=list, description="Models supporting MCP.")
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict, description="Provider configurations.")
    model_provider_mapping: Dict[str, str] = Field(default_factory=dict, description="Model to provider mapping.")
    preload_providers: List[str] = Field(default_factory=list, description="Providers to initialize on startup.")
    interventions: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Intervention configurations.")
    routing_strategy: Union[str, Dict[str, Any]] = Field("default", description="Provider selection strategy.")
    failover_enabled: bool = Field(True, description="Whether to attempt provider failover on error.")

# --- Type Hints and Abstract Classes ---

class AsyncIterableStreamChunk(AsyncGenerator[StreamChunk, None]):
    """Type hint for async generators of stream chunks."""
    pass

class RoutingError(Exception):
    """Custom exception for provider routing failures."""
    pass

class GatewayPipeline:
    """Abstract interface for Gateway pipelines, implemented by providers."""
    
    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process a request synchronously and return a response."""
        raise NotImplementedError("Subclasses must implement process_request")
    
    async def process_stream(self, request: LLMRequest) -> AsyncIterableStreamChunk:
        """Process a request and yield stream chunks."""
        raise NotImplementedError("Subclasses must implement process_stream")
    
    async def process_batch(self, batch_request: BatchLLMRequest) -> BatchLLMResponse:
        """Process a batch of requests."""
        raise NotImplementedError("Subclasses must implement process_batch")

# --- MCP Conversion Utilities ---

class MCPConverter:
    """Utility class for converting between Gateway and MCP formats."""
    
    @staticmethod
    def gateway_to_mcp_message(turn: ConversationTurn) -> Dict[str, Any]:
        """Convert a Gateway ConversationTurn to MCP message format."""
        content_list = []
        
        # Handle different content types
        if isinstance(turn.content, str):
            # Plain text becomes a single text content block
            content_list = [{"type": "text", "text": turn.content}]
        elif isinstance(turn.content, list):
            # Already a list of ContentItems
            content_list = [MCPConverter._content_item_to_mcp(item) for item in turn.content]
        elif isinstance(turn.content, dict):
            # Single dict content that needs conversion
            if "type" in turn.content:
                # If it looks like a ContentItem-like dict
                content_list = [turn.content]
            else:
                # Default to text if structure unknown
                content_list = [{"type": "text", "text": str(turn.content)}]
        
        # Create MCP message
        return {
            "role": turn.role,
            "content": content_list,
            "id": turn.turn_id,
            "metadata": turn.metadata
        }
    
    @staticmethod
    def _content_item_to_mcp(item: Union[ContentItem, Dict[str, Any]]) -> Dict[str, Any]:
        """Convert ContentItem to MCP content block."""
        if isinstance(item, ContentItem):
            # Extract from ContentItem model
            return {"type": item.type.value, **item.data}
        else:
            # Already a dict, assume correctly structured
            return item
    
    @staticmethod
    def mcp_to_gateway_message(mcp_message: Dict[str, Any]) -> ConversationTurn:
        """Convert an MCP message to a Gateway ConversationTurn."""
        role = mcp_message.get("role", "unknown")
        
        # Process content array from MCP
        content_items = []
        mcp_content = mcp_message.get("content", [])
        
        for content_block in mcp_content:
            content_type = content_block.get("type")
            
            if content_type == "text":
                # Text is common, handle specially
                text = content_block.get("text", "")
                content_items.append(ContentItem.from_text(text))
            elif content_type == "image":
                # Handle image data
                if "url" in content_block.get("image", {}).get("source", {}):
                    url = content_block["image"]["source"]["url"]
                    content_items.append(ContentItem.from_image_url(url))
                elif "data" in content_block.get("image", {}).get("source", {}):
                    data = content_block["image"]["source"]["data"]
                    mime = content_block.get("image", {}).get("format", "jpeg")
                    content_items.append(ContentItem.from_image_base64(data, f"image/{mime}"))
            else:
                # For other types, create generic ContentItem
                try:
                    content_items.append(ContentItem(
                        type=MCPContentType(content_type),
                        data=content_block
                    ))
                except ValueError:
                    # Handle unknown content types
                    content_items.append(ContentItem(
                        type=MCPContentType.FILE,
                        data={"unknown": content_block}
                    ))
        
        # Create turn with proper metadata
        return ConversationTurn(
            turn_id=mcp_message.get("id", str(uuid.uuid4())),
            role=role,
            content=content_items,
            timestamp=datetime.fromisoformat(mcp_message.get("created_at")) if "created_at" in mcp_message else datetime.utcnow(),
            metadata=mcp_message.get("metadata", {})
        )
    
    @staticmethod
    def gateway_finish_to_mcp_stop(reason: FinishReason) -> MCPStopReason:
        """Map Gateway finish reason to MCP stop reason."""
        mapping = {
            FinishReason.STOP: MCPStopReason.END_TURN,
            FinishReason.LENGTH: MCPStopReason.MAX_TOKENS,
            FinishReason.TOOL_CALLS: MCPStopReason.TOOL_USE,
            FinishReason.CONTENT_FILTERED: MCPStopReason.CONTENT_FILTERED,
            FinishReason.ERROR: MCPStopReason.ERROR,
            FinishReason.UNKNOWN: MCPStopReason.END_TURN,
        }
        return mapping.get(reason, MCPStopReason.END_TURN)
    
    @staticmethod
    def mcp_stop_to_gateway_finish(reason: MCPStopReason) -> FinishReason:
        """Map MCP stop reason to Gateway finish reason."""
        mapping = {
            MCPStopReason.END_TURN: FinishReason.STOP,
            MCPStopReason.MAX_TOKENS: FinishReason.LENGTH,
            MCPStopReason.TOOL_USE: FinishReason.TOOL_CALLS,
            MCPStopReason.CONTENT_FILTERED: FinishReason.CONTENT_FILTERED,
            MCPStopReason.ERROR: FinishReason.ERROR,
            MCPStopReason.STOP_SEQUENCE: FinishReason.STOP
        }
        return mapping.get(reason, FinishReason.UNKNOWN)