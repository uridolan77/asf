# LLM Gateway

The LLM Gateway is a centralized service for managing and interacting with various LLM providers. It provides a unified interface for sending requests to different LLM models, managing provider configurations, and handling common concerns like caching, resilience, and observability.

## Architecture

The LLM Gateway is designed with a modular architecture that separates concerns and allows for easy extension:

```
LLM Gateway
├── API Layer (FastAPI)
│   ├── Provider Management
│   ├── Model Management
│   └── LLM Interaction
├── Service Layer
│   ├── Provider Service
│   ├── Enhanced LLM Service
│   └── Service Abstraction Layer
├── Repository Layer
│   ├── Provider Repository
│   └── Audit Repository
├── Core Components
│   ├── Client
│   ├── Config Loader
│   ├── Factory
│   └── Models
└── Providers
    ├── OpenAI
    ├── Anthropic
    └── Custom Providers
```

### Key Components

- **API Layer**: Provides RESTful endpoints for managing providers, models, and interacting with LLMs.
- **Service Layer**: Contains business logic for provider management and LLM interactions.
- **Repository Layer**: Handles data access and persistence.
- **Core Components**: Provides core functionality like configuration management and client interfaces.
- **Providers**: Implements specific provider integrations.

## Provider Management

The LLM Gateway includes a comprehensive provider management system that allows you to:

1. **Add and configure providers**: Register new LLM providers with their connection details.
2. **Manage models**: Configure which models are available for each provider.
3. **Handle API keys**: Securely store and manage API keys for different providers.
4. **Set connection parameters**: Configure provider-specific connection parameters.

### Provider Data Model

The provider management system uses the following data model:

- **Provider**: Represents an LLM provider like OpenAI or Anthropic.
- **ProviderModel**: Represents a specific model offered by a provider.
- **ApiKey**: Stores API keys for providers with optional encryption.
- **ConnectionParameter**: Stores connection parameters for providers.
- **AuditLog**: Tracks changes to providers and related entities.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- SQLite or another supported database
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize the database:
   ```bash
   python -m asf.medical.llm_gateway.scripts.init_db --sync
   ```

4. Start the API:
   ```bash
   python -m asf.medical.llm_gateway.run_api
   ```

### Configuration

The LLM Gateway can be configured using environment variables:

- `LLM_GATEWAY_DATABASE_URL`: Database connection URL (default: `sqlite:///./llm_gateway.db`)
- `LLM_GATEWAY_CONFIG_PATH`: Path to the configuration file (default: `config/gateway_config.yaml`)
- `LLM_GATEWAY_ENCRYPTION_KEY`: Key for encrypting sensitive data
- `LLM_GATEWAY_API_HOST`: Host for the API server (default: `0.0.0.0`)
- `LLM_GATEWAY_API_PORT`: Port for the API server (default: `8000`)

### API Usage

#### Provider Management

```python
import requests

# Base URL for the API
base_url = "http://localhost:8000"

# Get all providers
response = requests.get(f"{base_url}/providers")
providers = response.json()

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
response = requests.post(f"{base_url}/providers", json=new_provider)
created_provider = response.json()

# Update a provider
update_data = {
    "display_name": "Anthropic AI",
    "enabled": True
}
response = requests.put(f"{base_url}/providers/anthropic", json=update_data)
updated_provider = response.json()

# Delete a provider
response = requests.delete(f"{base_url}/providers/anthropic")
```

#### Model Management

```python
import requests

# Base URL for the API
base_url = "http://localhost:8000"

# Get all models for a provider
response = requests.get(f"{base_url}/providers/openai/models")
models = response.json()

# Create a new model
new_model = {
    "model_id": "gpt-4o",
    "provider_id": "openai",
    "display_name": "GPT-4o",
    "model_type": "chat",
    "context_window": 128000,
    "max_tokens": 4096,
    "enabled": True
}
response = requests.post(f"{base_url}/providers/openai/models", json=new_model)
created_model = response.json()

# Update a model
update_data = {
    "display_name": "GPT-4o (2024)",
    "context_window": 256000
}
response = requests.put(f"{base_url}/providers/openai/models/gpt-4o", json=update_data)
updated_model = response.json()

# Delete a model
response = requests.delete(f"{base_url}/providers/openai/models/gpt-4o")
```

## Integration with Backoffice Backend

The LLM Gateway is designed to be integrated with the Backoffice Backend. The Backoffice Backend can use the LLM Gateway API to manage providers and models, and to send requests to LLMs.

### Provider Management Integration

The Backoffice Backend can use the LLM Gateway API to manage providers and models. This allows the Backoffice Backend to offload provider management to the LLM Gateway, which provides a more robust and feature-rich provider management system.

### LLM Interaction Integration

The Backoffice Backend can use the LLM Gateway Client to send requests to LLMs. This allows the Backoffice Backend to benefit from the enhanced capabilities of the LLM Gateway, such as caching, resilience, and observability.

```python
from asf.medical.llm_gateway.core.client import LLMGatewayClient
from asf.medical.llm_gateway.core.models import LLMRequest, LLMConfig, InterventionContext
from asf.medical.llm_gateway.core.factory import ProviderFactory
from asf.medical.llm_gateway.core.config_loader import ConfigLoader

# Create a gateway client
config_loader = ConfigLoader()
gateway_config = config_loader.load_config_as_object()
provider_factory = ProviderFactory()
client = LLMGatewayClient(gateway_config, provider_factory)

# Create a request
llm_config = LLMConfig(model_identifier="gpt-4-turbo-preview")
context = InterventionContext(session_id="example-session")
llm_req = LLMRequest(
    prompt_content="Hello, world!",
    config=llm_config,
    initial_context=context
)

# Send the request
response = await client.generate(llm_req)
print(response.generated_content)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
