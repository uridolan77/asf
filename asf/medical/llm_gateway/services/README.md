# Service Abstraction Layer for LLM Gateway

This directory contains the Service Abstraction Layer (SAL) for the LLM Gateway, which provides a high-level interface for interacting with LLMs through the gateway.

## Overview

The Service Abstraction Layer is designed to provide a consistent interface for LLM operations while adding cross-cutting concerns like caching, resilience, observability, events, and progress tracking. It follows a modular design where each concern is implemented as a separate component.

## Components

The SAL is divided into the following components:

- **Core Operations**: Basic LLM operations like text generation, chat, and embeddings
- **Caching**: Semantic caching for LLM requests
- **Resilience**: Retry logic, circuit breakers, and timeouts
- **Observability**: Metrics recording and tracing
- **Events**: Event publishing and subscription
- **Progress Tracking**: Tracking and reporting on long-running operations

## Usage

### Basic Usage

```python
from asf.medical.llm_gateway.services.enhanced_llm_service import EnhancedLLMService

# Create a service instance
service = EnhancedLLMService()

# Initialize the service
await service.initialize()

# Use the service
response = await service.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    model="gpt-4",
    params={"temperature": 0.7}
)

# Shut down the service when done
await service.shutdown()
```

### Configuration

You can configure the service by passing options to the constructor:

```python
service = EnhancedLLMService(
    config={
        "service_id": "my_llm_service",
        "cache": {
            "similarity_threshold": 0.95,
            "max_entries": 5000,
            "ttl_seconds": 1800
        }
    },
    enable_caching=True,
    enable_resilience=True,
    enable_observability=True,
    enable_events=True,
    enable_progress_tracking=True
)
```

### Advanced Features

#### Caching

```python
# Manually interact with the cache
value = await service.get_from_cache("my_key")
await service.store_in_cache("my_key", "my_value", ttl=3600)
await service.invalidate_cache("my_key")
await service.clear_cache()
```

#### Resilience

```python
# Use retry logic
result = await service.with_retry(
    lambda: some_operation(),
    max_retries=5,
    retry_delay=1.0,
    backoff_factor=2.0
)

# Use circuit breaker
result = await service.with_circuit_breaker(
    lambda: some_operation(),
    circuit_name="my_circuit",
    fallback=lambda: fallback_operation()
)

# Use timeout
result = await service.with_timeout(
    lambda: some_operation(),
    timeout_seconds=10.0
)
```

#### Observability

```python
# Record metrics
service.record_metric("my_metric", 1.0, {"tag": "value"})

# Use tracing
span = service.start_span("my_operation")
try:
    # Do something
    pass
finally:
    service.end_span(span)
```

#### Events

```python
# Publish events
await service.publish_event("my_event", {"key": "value"})

# Subscribe to events
async def handle_event(payload):
    print(f"Received event: {payload}")

await service.subscribe_to_events("my_event", handle_event)
```

#### Progress Tracking

```python
# Create a progress tracker
tracker = service.create_progress_tracker(
    operation_id="my_operation",
    total_steps=10,
    operation_type="my_operation_type"
)

# Update progress
service.update_progress(tracker, 1, "Step 1 completed")
service.update_progress(tracker, 2, "Step 2 completed")
# ...
service.update_progress(tracker, 10, "Operation completed")
```

## Architecture

The Service Abstraction Layer follows a component-based architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Service Abstraction Layer                     │
├─────────┬─────────┬─────────┬─────────┬─────────┬──────────────┤
│  Core   │ Caching │Resilience│Observ- │ Events  │  Progress    │
│Operations│        │         │ability  │         │  Tracking    │
└─────────┴─────────┴─────────┴─────────┴─────────┴──────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LLM Gateway Client                        │
├─────────────────────────────────────────────────────────────────┤
│                        Provider Router                          │
├─────────┬─────────┬─────────┬─────────┬─────────┬──────────────┤
│ OpenAI  │Anthropic│ Mistral │  Azure  │  Other  │    Mock      │
│Provider │Provider │Provider │Provider │Providers│  Provider    │
└─────────┴─────────┴─────────┴─────────┴─────────┴──────────────┘
```

## Files

- `service_abstraction_layer.py`: Abstract base class defining the enhanced interface
- `enhanced_llm_service.py`: Concrete implementation of the Service Abstraction Layer
- `components/`: Directory containing the individual components
  - `core_operations.py`: Core LLM operations
  - `caching.py`: Caching functionality
  - `resilience.py`: Resilience patterns
  - `observability.py`: Metrics and tracing
  - `events.py`: Event publishing and subscription
  - `progress_tracking.py`: Progress tracking for long-running operations
