# LLM Gateway

## Overview

The LLM Gateway provides a unified interface for interacting with various Large Language Model providers (OpenAI, Anthropic, custom MCP servers) through a consistent API. The gateway handles transport protocols, authentication, resilience patterns, and observability, allowing client applications to focus on their core logic without worrying about provider-specific implementation details.

## Consolidated Architecture

The LLM Gateway has been reorganized into a consolidated architecture with clear separation of concerns:

```
llm_gateway/
│
├── transport/              # Unified transport layer
│   ├── base.py            # Base transport interfaces and errors
│   ├── grpc_transport.py  # gRPC transport implementation
│   ├── http_transport.py  # HTTP transport implementation
│   ├── websocket_transport.py  # WebSocket transport implementation
│   ├── stdio_transport.py # StdIO transport implementation
│   └── factory.py         # Unified transport factory
│
├── resilience/             # Unified resilience components
│   ├── circuit_breaker.py # Circuit breaker implementation
│   ├── rate_limiter.py    # Rate limiting implementation
│   └── retry.py           # Retry policies
│
├── observability/          # Unified observability components
│   ├── logging.py         # Structured logging
│   ├── metrics.py         # Metrics collection
│   ├── tracing.py         # Distributed tracing
│   └── prometheus.py      # Prometheus metrics exporter
│
├── providers/              # LLM Provider implementations
│   ├── base.py            # Base provider interface
│   ├── openai_client.py   # OpenAI provider
│   ├── anthropic_client.py # Anthropic provider
│   ├── mcp_provider.py    # MCP provider
│   └── mock_client.py     # Mock provider for testing
│
├── config/                 # Configuration components
│   ├── models.py          # Configuration models
│   └── manager.py         # Configuration management
│
└── mcp/                    # MCP-specific components
    ├── session.py         # MCP session management
    ├── session_pool.py    # Advanced session pooling
    ├── transport.py       # MCP transport adaptations
    ├── errors.py          # MCP-specific errors
    └── config/            # MCP-specific configuration
```

## Key Components

### Transport Layer

The transport layer provides a unified interface for different communication protocols (HTTP, gRPC, WebSockets, StdIO). Each transport implementation adheres to the base `Transport` interface, making them interchangeable.

```python
from asf.medical.llm_gateway.transport import TransportFactory

# Create a transport instance based on configuration
transport = TransportFactory().create_transport(
    provider_id="anthropic",
    config={"transport_type": "http", "base_url": "https://api.anthropic.com"}
)

# Use the transport
await transport.initialize()
response = await transport.send_request({"prompt": "Hello, world!"})
await transport.close()
```

### Resilience Components

The resilience components provide circuit breakers, rate limiters, and retry policies for robust communication with LLM providers.

```python
from asf.medical.llm_gateway.resilience import CircuitBreaker, RetryPolicy, RateLimiter

# Create resilience components
circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)
rate_limiter = RateLimiter(requests_per_minute=60)
retry_policy = RetryPolicy(max_retries=3, backoff_factor=1.5)

# Use in a provider
async with circuit_breaker:
    async with rate_limiter:
        result = await retry_policy.execute(lambda: provider.generate_text("Hello"))
```

### Observability

The observability components provide logging, metrics, and distributed tracing for monitoring and debugging.

```python
from asf.medical.llm_gateway.observability import MetricsService, TracingService

# Record a metric
metrics = MetricsService()
metrics.increment("llm_gateway.requests_total", labels={"provider": "anthropic"})
metrics.observe("llm_gateway.latency_ms", 123.5, labels={"provider": "anthropic"})

# Create a span for tracing
tracing = TracingService()
with tracing.start_span("llm_request", attributes={"provider": "anthropic"}):
    result = await provider.generate_text("Hello")
```

### Provider Implementations

The provider implementations adapt specific LLM services to the unified gateway interface.

```python
from asf.medical.llm_gateway.providers import OpenAIClient, AnthropicClient, MCPProvider

# Create a provider
openai = OpenAIClient(api_key="your-key", model="gpt-4")
anthropic = AnthropicClient(api_key="your-key", model="claude-3-opus")
mcp = MCPProvider(endpoint="localhost:8080", transport_type="grpc")

# Use a provider
response = await openai.generate_text("Explain quantum computing")
streaming_response = await anthropic.generate_text_stream("Write a story about a robot")
for chunk in streaming_response:
    print(chunk.text, end="")
```

### MCP Support

The MCP (Model Context Protocol) components provide specialized support for MCP servers with advanced session management and pooling.

```python
from asf.medical.llm_gateway.mcp import MCPSession, EnhancedSessionPool

# Create a session pool
pool = EnhancedSessionPool(
    provider_id="mcp-local",
    create_session_func=create_session,
    close_session_func=close_session
)

# Use a session from the pool
async with pool.get_session(model_id="your-model") as session:
    response = await session.create_message(messages=[{"role": "user", "content": "Hello"}])
```

## Configuration

The LLM Gateway uses a flexible configuration system with support for environment variables, configuration files, and secrets management.

```python
from asf.medical.llm_gateway.config import ConfigManager

# Load configuration
config_manager = ConfigManager()
mcp_config = config_manager.load_mcp_config("my-provider")

# Use configuration
transport_config = mcp_config.get_transport_config()
transport = TransportFactory().create_transport(transport_config)
```

## Getting Started

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Configure your LLM providers:

```python
# Example configuration
provider_config = {
    "openai": {
        "api_key": "your-openai-api-key",
        "default_model": "gpt-4",
        "timeout_seconds": 60
    },
    "anthropic": {
        "api_key": "your-anthropic-api-key",
        "default_model": "claude-3-opus-20240229",
        "timeout_seconds": 60
    }
}
```

3. Create a gateway instance and use it:

```python
from asf.medical.llm_gateway.providers import OpenAIClient

# Create a provider
provider = OpenAIClient(
    api_key="your-openai-api-key",
    model="gpt-4"
)

# Use the provider
response = await provider.generate_text("Explain the theory of relativity")
print(response)
```

## Advanced Usage

### Custom Transport Implementation

You can create custom transport implementations by extending the base `Transport` class:

```python
from asf.medical.llm_gateway.transport import Transport, TransportConfig
from pydantic import Field

class MyCustomConfig(TransportConfig):
    transport_type: str = "custom"
    custom_param: str = Field(...)

class MyCustomTransport(Transport):
    """Custom transport implementation"""
    
    def __init__(self, provider_id: str, config: dict):
        super().__init__(provider_id, config)
        self.custom_param = config.get("custom_param")
    
    async def initialize(self) -> None:
        # Custom initialization
        pass
    
    async def send_request(self, request: dict) -> dict:
        # Custom request handling
        pass
    
    # Implement other required methods...

# Register the custom transport
from asf.medical.llm_gateway.transport import TransportFactory
factory = TransportFactory()
factory.register_transport("custom", MyCustomTransport)
```

### MCP Session Pooling

For advanced MCP server integration, you can use the enhanced session pooling:

```python
from asf.medical.llm_gateway.mcp import (
    EnhancedSessionPool, 
    SessionPoolConfig,
    SessionPriority
)

# Create session pool configuration
pool_config = SessionPoolConfig(
    min_size=2,
    max_size=10,
    adaptive_sizing=True,
    load_target_percentage=70
)

# Create the session pool
pool = EnhancedSessionPool(
    provider_id="mcp-server",
    create_session_func=create_session,
    close_session_func=close_session,
    ping_session_func=ping_session,
    config=pool_config
)

# Start the pool
await pool.start()

# Use the pool with advanced options
async with pool.get_session(
    model_id="my-model",
    tags={"purpose": "classification"},
    priority=SessionPriority.HIGH,
    timeout=10.0
) as session:
    result = await session.create_message(messages=[...])

# Get pool statistics
stats = pool.get_stats()
print(f"Active sessions: {stats['active_sessions']}/{stats['total_sessions']}")
print(f"Success rate: {stats['success_rate']:.2f}%")

# Clean up
await pool.stop()
```

## Contributing

When adding new functionality to the LLM Gateway, please follow these guidelines:

1. Maintain the separation of concerns between transport, resilience, observability, and provider layers
2. Add appropriate tests for new functionality
3. Update documentation to reflect changes
4. Follow the established patterns for error handling and configuration

## License

[Include your license information here]