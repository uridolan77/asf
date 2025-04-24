# LLM Gateway Client

This module provides a client for interacting with the LLM Gateway API.

## Overview

The LLM Gateway Client allows you to interact with the LLM Gateway API from Python code. It provides methods for managing providers, models, API keys, and connection parameters, as well as for generating responses from LLMs.

## Installation

The client is part of the LLM Gateway package and is installed along with it.

```bash
# From the root directory
pip install -e .
```

## Usage

### Basic Usage

```python
from asf.medical.llm_gateway.client.api_client import LLMGatewayClient

# Create a client
client = LLMGatewayClient("http://localhost:8000", "your-api-key")

# Get all providers
providers = client.get_providers()
print(f"Found {len(providers)} providers")

# Get a specific provider
provider = client.get_provider("openai")
print(f"Provider: {provider['display_name']}")

# Generate a response from an LLM
response = client.generate({
    "prompt": "Hello, world!",
    "model": "gpt-4-turbo-preview",
    "provider_id": "openai",
    "max_tokens": 100,
    "temperature": 0.7
})
print(f"Response: {response['content']}")
```

### Provider Management

```python
# Create a new provider
new_provider = {
    "provider_id": "anthropic",
    "display_name": "Anthropic",
    "provider_type": "anthropic",
    "description": "Anthropic Claude models",
    "enabled": True,
    "connection_params": {
        "api_base": "https://api.anthropic.com"
    },
    "models": [
        {
            "model_id": "claude-3-opus-20240229",
            "display_name": "Claude 3 Opus",
            "model_type": "chat",
            "context_window": 200000,
            "max_tokens": 4096
        }
    ],
    "api_key": {
        "key_value": "your-api-key",
        "is_encrypted": True
    }
}
created_provider = client.create_provider(new_provider)

# Update a provider
update_data = {
    "display_name": "Anthropic AI",
    "enabled": True
}
updated_provider = client.update_provider("anthropic", update_data)

# Delete a provider
client.delete_provider("anthropic")

# Test a provider connection
test_result = client.test_provider("anthropic")
print(f"Test result: {test_result['success']}")

# Synchronize a provider with the LLM Gateway configuration
sync_result = client.sync_provider("anthropic")
```

### Model Management

```python
# Get all models for a provider
models = client.get_models("anthropic")
print(f"Found {len(models)} models for provider 'anthropic'")

# Get a specific model
model = client.get_model("anthropic", "claude-3-opus-20240229")
print(f"Model: {model['display_name']}")

# Create a new model
new_model = {
    "model_id": "claude-3-haiku-20240307",
    "display_name": "Claude 3 Haiku",
    "model_type": "chat",
    "context_window": 200000,
    "max_tokens": 4096,
    "enabled": True
}
created_model = client.create_model("anthropic", new_model)

# Update a model
update_data = {
    "display_name": "Claude 3 Haiku (2024)",
    "context_window": 256000
}
updated_model = client.update_model("anthropic", "claude-3-haiku-20240307", update_data)

# Delete a model
client.delete_model("anthropic", "claude-3-haiku-20240307")
```

### API Key Management

```python
# Get all API keys for a provider
api_keys = client.get_api_keys("anthropic")
print(f"Found {len(api_keys)} API keys for provider 'anthropic'")

# Create a new API key
new_api_key = {
    "key_value": "your-api-key",
    "is_encrypted": True,
    "environment": "development"
}
created_api_key = client.create_api_key("anthropic", new_api_key)

# Get the actual API key value
key_value = client.get_api_key_value("anthropic", created_api_key["key_id"])
print(f"API key value: {key_value['key_value']}")
```

### Connection Parameter Management

```python
# Get all connection parameters for a provider
params = client.get_connection_params("anthropic")
print(f"Connection parameters: {params}")

# Set a connection parameter
param_data = {
    "param_name": "api_version",
    "param_value": "2023-06-01",
    "is_sensitive": False,
    "environment": "development"
}
set_param = client.set_connection_param("anthropic", param_data)
```

### LLM Interaction

```python
# Generate a response from an LLM
request_data = {
    "prompt": "Hello, world!",
    "model": "claude-3-opus-20240229",
    "provider_id": "anthropic",
    "max_tokens": 100,
    "temperature": 0.7
}
response = client.generate(request_data)
print(f"Generated response: {response['content']}")
```

## Error Handling

The client raises exceptions for HTTP errors. You can catch these exceptions to handle errors:

```python
try:
    provider = client.get_provider("nonexistent-provider")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
except Exception as e:
    print(f"Error: {e}")
```

## Environment Variables

The client can be configured using environment variables:

- `LLM_GATEWAY_API_URL`: Base URL for the LLM Gateway API (default: `http://localhost:8000`)
- `LLM_GATEWAY_API_KEY`: API key for authentication

```python
import os

# Set environment variables
os.environ["LLM_GATEWAY_API_URL"] = "http://localhost:8000"
os.environ["LLM_GATEWAY_API_KEY"] = "your-api-key"

# Create client using environment variables
api_url = os.environ.get("LLM_GATEWAY_API_URL", "http://localhost:8000")
api_key = os.environ.get("LLM_GATEWAY_API_KEY")
client = LLMGatewayClient(api_url, api_key)
```

## Example Script

See the `example.py` script for a complete example of using the client.

```bash
# Run the example script
python -m asf.medical.llm_gateway.client.example
