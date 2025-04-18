# MCP Provider: Production-Ready Gateway Implementation

A comprehensive, production-grade MCP (Mind-Control-Protocol) gateway provider implementation with advanced features for scalability, reliability, and observability.

## Features

- **Robust Connection Management**:
  - Connection pooling with lifecycle management
  - Multiple transport options (stdio, gRPC, HTTP/REST)
  - Automatic reconnection and health monitoring

- **Streaming & Non-Streaming Support**:
  - Full duplex streaming with backpressure control
  - Cancellation propagation
  - Efficient token-by-token or chunked streaming

- **Advanced Resilience**:
  - Circuit breaker pattern to prevent cascading failures
  - Exponential backoff with jitter
  - Pluggable retry policies based on error categories

- **Comprehensive Observability**:
  - Structured logging with contextual data
  - OpenTelemetry metrics and tracing
  - Prometheus integration for monitoring
  - Detailed error classification and tracking

- **Configuration Management**:
  - Validated configurations with Pydantic
  - Environment variable interpolation
  - Secret management integration (Vault, AWS Secrets Manager)
  - Configuration profiles for different environments

- **Transport Flexibility**:
  - stdio for local process communication
  - gRPC for high-performance bi-directional streaming
  - HTTP/REST for cloud-based services
  - Pluggable transport architecture for custom implementations

## Installation

### Requirements

- Python 3.8+
- Required packages:
  - `pydantic` for configuration validation
  - `structlog` for structured logging
  - `tenacity` for retry mechanisms
  - `opentelemetry-api` and related packages for observability
  - Transport-specific libraries:
    - `httpx` for HTTP transport
    - `grpcio` for gRPC transport

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-org/mcp-provider.git
cd mcp-provider

# Install with all dependencies
pip install -e ".[all]"

# Or install with specific features
pip install -e ".[http,metrics]"
```

## Usage

### Basic Example

```python
import asyncio
from asf.medical.llm_gateway.core.models import LLMRequest, LLMConfig
from asf.medical.llm_gateway.providers.mcp_provider import MCPProvider
from asf.medical.llm_gateway.config.models import MCPConnectionConfig

async def main():
    # Create configuration
    config = MCPConnectionConfig(
        transport_type="stdio",
        stdio_config={
            "command": "npx",
            "args": ["@anthropic/mcp-starter", "--no-color"],
            "env": {}
        },
        enable_streaming=True,
        timeout_seconds=60,
        max_retries=3
    )

    # Create provider
    provider = MCPProvider(
        provider_config={
            "provider_id": "anthropic",
            "provider_type": "mcp",
            "connection_params": config.dict()
        }, 
        gateway_config={
            "max_retries": 3,
            "retry_delay_seconds": 1,
            "default_timeout_seconds": 30
        }
    )

    # Initialize provider
    await provider.initialize_async()

    try:
        # Create request
        request = LLMRequest(
            prompt_content="Tell me a short story about a robot learning to paint.",
            config=LLMConfig(
                model_identifier="claude-3-haiku",
                max_tokens=300,
                temperature=0.7
            )
        )

        # Generate response
        response = await provider.generate(request)
        print(f"Finish reason: {response.finish_reason}")
        print(f"Content: {response.generated_content}")
        
    finally:
        # Clean up provider
        await provider.cleanup()

# Run the example
asyncio.run(main())
```

### Streaming Example

```python
import asyncio
from asf.medical.llm_gateway.core.models import LLMRequest, LLMConfig
from asf.medical.llm_gateway.providers.mcp_provider import MCPProvider
from asf.medical.llm_gateway.config.models import MCPConnectionConfig

async def main():
    # Create configuration with streaming enabled
    config = MCPConnectionConfig(
        transport_type="stdio",
        stdio_config={
            "command": "npx",
            "args": ["@anthropic/mcp-starter", "--no-color"],
            "env": {}
        },
        enable_streaming=True,
        timeout_seconds=60
    )

    # Create provider
    provider = MCPProvider(
        provider_config={
            "provider_id": "anthropic",
            "provider_type": "mcp",
            "connection_params": config.dict()
        }, 
        gateway_config={
            "max_retries": 3,
            "retry_delay_seconds": 1,
            "default_timeout_seconds": 30
        }
    )

    # Initialize provider
    await provider.initialize_async()

    try:
        # Create request
        request = LLMRequest(
            prompt_content="Write a poem about artificial intelligence.",
            config=LLMConfig(
                model_identifier="claude-3-haiku",
                max_tokens=300,
                temperature=0.7
            )
        )

        # Generate streaming response
        async for chunk in provider.generate_stream(request):
            if chunk.delta_text:
                print(chunk.delta_text, end="", flush=True)
            
            if chunk.finish_reason:
                print(f"\n\nFinish reason: {chunk.finish_reason}")
        
    finally:
        # Clean up provider
        await provider.cleanup()

# Run the example
asyncio.run(main())
```

### Using gRPC Transport

```python
from asf.medical.llm_gateway.config.models import MCPConnectionConfig, TransportType

# Create configuration with gRPC transport
config = MCPConnectionConfig(
    transport_type=TransportType.GRPC,
    grpc_config={
        "endpoint": "localhost:50051",
        "use_tls": False
    },
    enable_streaming=True,
    timeout_seconds=60
)

# Rest of the code remains the same
```

### Using HTTP Transport

```python
from asf.medical.llm_gateway.config.models import MCPConnectionConfig, TransportType

# Create configuration with HTTP transport
config = MCPConnectionConfig(
    transport_type=TransportType.HTTP,
    http_config={
        "base_url": "https://api.example.com/mcp",
        "headers": {
            "Content-Type": "application/json"
        },
        "verify_ssl": True
    },
    enable_streaming=True,
    timeout_seconds=60
)

# Rest of the code remains the same
```

## Configuration

The MCP Provider can be configured using a configuration file, environment variables, or programmatically:

### Using Configuration Files

```yaml
# config/mcp_connection_config.yaml
profiles:
  default:
    transport_type: stdio
    enable_streaming: true
    timeout_seconds: 60
    stdio_config:
      command: npx
      args: ["@anthropic/mcp-starter", "--no-color"]
      env: {}
    
  production:
    transport_type: http
    enable_streaming: true
    timeout_seconds: 60
    http_config:
      base_url: "https://api.example.com/mcp"
      headers:
        Content-Type: "application/json"
      verify_ssl: true
    api_key_env_var: "MCP_API_KEY"
    observability:
      enable_metrics: true
      enable_tracing: true
      enable_prometheus: true
      prometheus_port: 8000
```

```python
from asf.medical.llm_gateway.config.manager import ConfigManager

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_mcp_config(
    provider_id="anthropic",
    profile="production"
)

# Create provider with loaded config
provider = MCPProvider(
    provider_config={
        "provider_id": "anthropic",
        "provider_type": "mcp",
        "connection_params": config.dict()
    }, 
    gateway_config=gateway_config
)
```

### Environment Variable Configuration

Environment variables can override configuration values:

```bash
# Set environment variables
export MCP_TRANSPORT_TYPE=http
export MCP_HTTP_CONFIG_BASE_URL=https://api.example.com/mcp
export MCP_TIMEOUT_SECONDS=120
export MCP_API_KEY=your-api-key

# Or use a .env file with your favorite dotenv loader
```

## Advanced Features

### Circuit Breaker

The circuit breaker prevents cascading failures by failing fast when a service is unhealthy:

```python
from asf.medical.llm_gateway.config.models import MCPConnectionConfig, CircuitBreakerConfig

config = MCPConnectionConfig(
    # ... other config ...
    circuit_breaker=CircuitBreakerConfig(
        enabled=True,
        failure_threshold=5,  # Open circuit after 5 failures
        recovery_timeout=30,  # Wait 30 seconds before trying again
        half_open_max_calls=2  # Allow 2 test calls in half-open state
    )
)
```

### Custom Retry Policies

Configure advanced retry behavior with exponential backoff and jitter:

```python
from asf.medical.llm_gateway.config.models import MCPConnectionConfig, RetryConfig

config = MCPConnectionConfig(
    # ... other config ...
    retry=RetryConfig(
        max_retries=5,
        base_delay=1.0,
        max_delay=30.0,
        jitter_factor=0.2,
        retry_codes={"RATE_LIMIT_EXCEEDED", "SERVICE_UNAVAILABLE"}
    )
)
```

### Metrics and Tracing

Enable comprehensive observability:

```python
from asf.medical.llm_gateway.config.models import MCPConnectionConfig, ObservabilityConfig

config = MCPConnectionConfig(
    # ... other config ...
    observability=ObservabilityConfig(
        enable_metrics=True,
        enable_tracing=True,
        enable_prometheus=True,
        prometheus_port=8000,
        otlp_endpoint="localhost:4317",
        service_name="mcp-provider",
        log_level="INFO",
        structured_logging=True
    )
)
```

## Architecture

The MCP Provider is designed with a modular architecture following SOLID principles:

1. **Provider Layer**: High-level interface for LLM requests
2. **Transport Layer**: Pluggable communication mechanisms
3. **Resilience Layer**: Circuit breakers and retry mechanisms
4. **Observability Layer**: Metrics, tracing, and logging
5. **Configuration Layer**: Validated configurations and secrets

This layered approach allows for easy extension and customization.

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=asf.medical.llm_gateway
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.