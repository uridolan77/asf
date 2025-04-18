# LLM Gateway Migration Guide

This guide helps you update your code to use the new consolidated LLM Gateway structure.

## Overview of Changes

The LLM Gateway has been reorganized with a more consolidated architecture:

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
├── mcp/                    # MCP-specific components
│   ├── session.py         # MCP session management
│   ├── session_pool.py    # Advanced session pooling
│   └── transport.py       # MCP transport adaptations
│
└── config/                 # Configuration components
    ├── models.py          # Configuration models
    └── manager.py         # Configuration management
```

## Import Mapping

Update your imports according to this mapping:

| Old Import | New Import |
|------------|------------|
| `from asf.medical.llm_gateway.core.client import LLMGatewayClient` | `from asf.medical.llm_gateway.client import LLMGatewayClient` |
| `from asf.medical.llm_gateway.core.models import LLMRequest` | `from asf.medical.llm_gateway.models import LLMRequest` |
| `from asf.medical.llm_gateway.core.factory import ProviderFactory` | `from asf.medical.llm_gateway.transport.factory import TransportFactory` |
| `from asf.medical.llm_gateway.core.provider import Provider` | `from asf.medical.llm_gateway.providers.base import LLMProvider` |
| `from asf.medical.llm_gateway.mcp.observability import MetricsService` | `from asf.medical.llm_gateway.observability.metrics import MetricsService` |
| `from asf.medical.llm_gateway.mcp.resilience import CircuitBreaker` | `from asf.medical.llm_gateway.resilience.circuit_breaker import CircuitBreaker` |

## Model Changes

Several model classes have been renamed or restructured:

| Old Class | New Class |
|-----------|-----------|
| `InterventionContext` | `ConversationContext` |
| `MCPRole` | `MessageRole` |
| `GatewayConfig` | `GatewayConfig` (in config module) |
| `ProviderConfig` | `LLMProviderConfig` |
| `Transport` | `Transport` (in transport.base) |

## Gateway Client Initialization

Update your gateway client initialization:

```python
# Old initialization
from asf.medical.llm_gateway.core.client import LLMGatewayClient
from asf.medical.llm_gateway.core.factory import ProviderFactory

provider_factory = ProviderFactory()
gateway_client = LLMGatewayClient(gateway_config, provider_factory)

# New initialization
from asf.medical.llm_gateway.client import LLMGatewayClient
from asf.medical.llm_gateway.transport.factory import TransportFactory

transport_factory = TransportFactory()
gateway_client = LLMGatewayClient(config=gateway_config, transport_factory=transport_factory)
```

## MCP Provider Usage

Update your MCP provider usage:

```python
# Old usage
from asf.medical.llm_gateway.mcp import MCPSession

# New usage
from asf.medical.llm_gateway.mcp import MCPSession, EnhancedSessionPool
from asf.medical.llm_gateway.providers import MCPProvider
```

## Configuration Updates

The configuration structure has been updated to support the new consolidated components. Add these sections to your provider configurations:

```yaml
provider_name:
  # Existing configuration...
  
  # Add transport configuration
  transport:
    type: http  # or grpc, stdio, websocket, local
    timeout_seconds: 60
    
  # Add resilience configuration
  resilience:
    circuit_breaker:
      enabled: true
      failure_threshold: 5
    retry:
      max_retries: 3
      base_delay: 1.0
      
  # Add observability configuration
  observability:
    metrics_enabled: true
    tracing_enabled: true
```

## Working with Transports

The new transport layer provides a unified interface for different communication protocols:

```python
from asf.medical.llm_gateway.transport import TransportFactory

# Create a transport instance
transport = TransportFactory().create_transport(
    provider_id="anthropic",
    config={"type": "http", "base_url": "https://api.anthropic.com"}
)

# Use the transport
await transport.initialize()
response = await transport.send_request({"prompt": "Hello, world!"})
await transport.close()
```

## Working with Resilience Components

Use the new resilience components for robust communication:

```python
from asf.medical.llm_gateway.resilience import CircuitBreaker, RetryPolicy

circuit_breaker = CircuitBreaker(failure_threshold=5)
retry_policy = RetryPolicy(max_retries=3)

async with circuit_breaker:
    result = await retry_policy.execute(lambda: provider.generate_text("Hello"))
```

## Working with Observability

Use the new observability components for monitoring:

```python
from asf.medical.llm_gateway.observability import MetricsService, TracingService

# Record metrics
metrics = MetricsService()
metrics.increment("llm_gateway.requests_total", {"provider": "anthropic"})

# Create traces
tracing = TracingService()
with tracing.start_span("llm_request", {"provider": "anthropic"}):
    result = await provider.generate_text("Hello")
```

## Need Help?

If you encounter any issues during migration, please refer to the [full documentation](../../../README.md) or contact the LLM Gateway team.