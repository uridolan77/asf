# LLM Gateway Service Abstraction Layer

## Overview

The Service Abstraction Layer provides a clean interface for interacting with different LLM providers. It abstracts away the provider-specific implementation details and exposes a unified API for text generation, embedding generation, and chat capabilities.

## Architecture

The Service Abstraction Layer follows a layered architecture:

```
                  +------------------------+
                  |  LLMGatewayManager     |
                  +------------------------+
                            |
                            v
                  +------------------------+
                  |  ServiceFactory        |
                  +------------------------+
                            |
                            v
+--------------------------------------------------------+
|                    LLMServiceInterface                  |
+--------------------------------------------------------+
          |                 |                  |
          v                 v                  v
+------------------+ +---------------+ +------------------+
| MCPService       | | OtherService1 | | OtherService2    |
+------------------+ +---------------+ +------------------+
          |
          v
+------------------+
| MCPProvider      |
+------------------+
          |
          v
+------------------+
| Transport Layer  |
+------------------+
```

## Components

### Interfaces

- **LLMServiceInterface**: Abstract interface defining the contract for all LLM services
- **Exceptions**: Service-level exceptions for error handling

### Services

- **BaseService**: Base implementation with common functionality and error handling
- **MCPService**: Implementation for the Model Context Protocol
- **ServiceFactory**: Factory for creating the appropriate service implementation

### Integration

The Service Abstraction Layer integrates with the existing LLM Gateway components:

- The `LLMGatewayManager` uses the `ServiceFactory` to route requests to the appropriate service
- Services adapt the underlying providers to the common interface
- Error handling is standardized across all services

## Usage

To use the Service Abstraction Layer:

```python
# Initialize the gateway manager with configuration
manager = LLMGatewayManager(config)

# Generate text
text = await manager.generate_text("Write a poem about AI", "mcp-model-name")

# Generate embeddings
embeddings = await manager.get_embeddings(["text to embed"], "mcp-embedding-model")

# Chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, who are you?"}
]
response = await manager.chat(messages, "mcp-chat-model")
```

## Design Principles

1. **Separation of Concerns**: Clear separation between interface and implementation
2. **Extensibility**: Easy to add new service implementations
3. **Error Handling**: Standardized error handling across all services
4. **Resource Management**: Proper handling of resources and connections

## Future Enhancements

1. **Additional Services**: Support for more LLM providers
2. **Advanced Features**: Support for function calling, tools, etc.
3. **Caching**: Response caching for improved performance
4. **Load Balancing**: Load balancing across multiple providers