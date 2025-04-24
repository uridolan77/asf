# Migration Guide: Backoffice Backend to LLM Gateway

This guide provides instructions for migrating provider management and LLM interaction functionality from the Backoffice Backend to the LLM Gateway.

## Overview

The migration involves moving the following components from the Backoffice Backend to the LLM Gateway:

1. **Provider Management**: Moving provider data models, repositories, and services
2. **LLM Interaction**: Using the LLM Gateway client for LLM requests
3. **API Integration**: Updating API endpoints to use the LLM Gateway

## Migration Steps

### 1. Provider Management Migration

#### 1.1 Data Model Migration

The Backoffice Backend provider data models should be replaced with the LLM Gateway models:

| Backoffice Backend | LLM Gateway |
|-------------------|-------------|
| `Provider` | `asf.medical.llm_gateway.models.provider.Provider` |
| `ProviderModel` | `asf.medical.llm_gateway.models.provider.ProviderModel` |
| `ApiKey` | `asf.medical.llm_gateway.models.provider.ApiKey` |
| `ConnectionParameter` | `asf.medical.llm_gateway.models.provider.ConnectionParameter` |

#### 1.2 Repository Migration

Replace the Backoffice Backend provider repositories with the LLM Gateway repositories:

```python
# Before
from asf.bo.backend.repositories.provider_repository import ProviderRepository
provider_repo = ProviderRepository(db)

# After
from asf.medical.llm_gateway.repositories.provider_repository import ProviderRepository
provider_repo = ProviderRepository(db, encryption_key)
```

#### 1.3 Service Migration

Replace the Backoffice Backend provider services with the LLM Gateway services:

```python
# Before
from asf.bo.backend.services.provider_service import ProviderService
provider_service = ProviderService(db)

# After
from asf.medical.llm_gateway.services.provider_service import ProviderService
provider_service = ProviderService(db, encryption_key, current_user_id)
```

### 2. LLM Interaction Migration

#### 2.1 Client Migration

Replace direct LLM provider interactions with the LLM Gateway client:

```python
# Before
from asf.bo.backend.services.llm_service import LLMService
llm_service = LLMService()
response = await llm_service.generate_text(prompt, model, provider_id)

# After
from asf.medical.llm_gateway.core.client import LLMGatewayClient
from asf.medical.llm_gateway.core.models import LLMRequest, LLMConfig, InterventionContext
from asf.medical.llm_gateway.core.factory import ProviderFactory
from asf.medical.llm_gateway.core.config_loader import ConfigLoader

# Create a gateway client
config_loader = ConfigLoader(db)
gateway_config = config_loader.load_config_as_object()
provider_factory = ProviderFactory()
client = LLMGatewayClient(gateway_config, provider_factory, db)

# Create a request
llm_config = LLMConfig(model_identifier=model, provider_id=provider_id)
context = InterventionContext(session_id=session_id)
llm_req = LLMRequest(
    prompt_content=prompt,
    config=llm_config,
    initial_context=context
)

# Send the request
response = await client.generate(llm_req)
generated_text = response.generated_content
```

#### 2.2 Configuration Migration

Migrate provider configuration from the Backoffice Backend to the LLM Gateway:

```python
# Before
provider_config = {
    "provider_id": "openai",
    "api_key": "your-api-key",
    "api_base": "https://api.openai.com/v1",
    "models": {
        "gpt-4-turbo-preview": {
            "max_tokens": 4096
        }
    }
}
provider_service.update_provider_config("openai", provider_config)

# After
from asf.medical.llm_gateway.core.config_loader import ConfigLoader

config_loader = ConfigLoader(db)
provider_config = {
    "provider_id": "openai",
    "provider_type": "openai",
    "display_name": "OpenAI",
    "enabled": True,
    "api_base": "https://api.openai.com/v1",
    "models": {
        "gpt-4-turbo-preview": {
            "display_name": "GPT-4 Turbo",
            "context_window": 128000,
            "max_tokens": 4096
        }
    }
}
config_loader.update_provider_config("openai", provider_config)

# Or use the Provider Service
from asf.medical.llm_gateway.services.provider_service import ProviderService

provider_service = ProviderService(db, encryption_key, current_user_id)
provider_service.create_provider({
    "provider_id": "openai",
    "provider_type": "openai",
    "display_name": "OpenAI",
    "enabled": True,
    "connection_params": {
        "api_base": "https://api.openai.com/v1"
    },
    "models": [
        {
            "model_id": "gpt-4-turbo-preview",
            "display_name": "GPT-4 Turbo",
            "context_window": 128000,
            "max_tokens": 4096
        }
    ],
    "api_key": {
        "key_value": "your-api-key",
        "is_encrypted": True
    }
})
```

### 3. API Integration

#### 3.1 API Client Migration

Create an API client for the LLM Gateway:

```python
# api_client.py
import requests
from typing import Dict, Any, List, Optional

class LLMGatewayClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def get_providers(self) -> List[Dict[str, Any]]:
        response = requests.get(f"{self.base_url}/providers", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_provider(self, provider_id: str) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/providers/{provider_id}", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def create_provider(self, provider_data: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/providers", json=provider_data, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def update_provider(self, provider_id: str, provider_data: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.put(f"{self.base_url}/providers/{provider_id}", json=provider_data, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def delete_provider(self, provider_id: str) -> Dict[str, Any]:
        response = requests.delete(f"{self.base_url}/providers/{provider_id}", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    # Add methods for models, API keys, connection parameters, etc.
```

#### 3.2 API Endpoint Migration

Update the Backoffice Backend API endpoints to use the LLM Gateway API:

```python
# Before
@router.get("/providers")
async def get_providers(db: Session = Depends(get_db)):
    provider_service = ProviderService(db)
    return provider_service.get_all_providers()

# After
@router.get("/providers")
async def get_providers():
    client = LLMGatewayClient(settings.LLM_GATEWAY_URL, settings.LLM_GATEWAY_API_KEY)
    return client.get_providers()
```

### 4. Database Migration

#### 4.1 Data Migration

Migrate existing provider data from the Backoffice Backend to the LLM Gateway:

```python
# migration_script.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sqlalchemy.orm import Session
from asf.bo.backend.repositories.provider_repository import ProviderRepository as BOProviderRepository
from asf.medical.llm_gateway.repositories.provider_repository import ProviderRepository as GWProviderRepository
from asf.bo.backend.db.session import get_db as get_bo_db
from asf.medical.llm_gateway.db.session import get_db_session as get_gw_db

def migrate_providers():
    # Get database sessions
    bo_db = next(get_bo_db())
    gw_db = get_gw_db()
    
    try:
        # Get repositories
        bo_repo = BOProviderRepository(bo_db)
        gw_repo = GWProviderRepository(gw_db)
        
        # Get all providers from Backoffice Backend
        providers = bo_repo.get_all_providers()
        
        # Migrate each provider
        for provider in providers:
            # Convert provider data
            provider_data = {
                "provider_id": provider.provider_id,
                "provider_type": provider.provider_type,
                "display_name": provider.display_name,
                "description": provider.description,
                "enabled": provider.enabled,
                "connection_params": provider.connection_params,
                "request_settings": provider.request_settings
            }
            
            # Create provider in LLM Gateway
            gw_provider = gw_repo.create_provider(provider_data)
            
            # Get models for provider
            models = bo_repo.get_models_by_provider_id(provider.provider_id)
            
            # Migrate each model
            for model in models:
                model_data = {
                    "model_id": model.model_id,
                    "provider_id": model.provider_id,
                    "display_name": model.display_name,
                    "model_type": model.model_type,
                    "context_window": model.context_window,
                    "max_tokens": model.max_tokens,
                    "enabled": model.enabled,
                    "capabilities": model.capabilities,
                    "parameters": model.parameters
                }
                
                # Create model in LLM Gateway
                gw_repo.create_model(model_data)
            
            # Get API keys for provider
            api_keys = bo_repo.get_api_keys_by_provider_id(provider.provider_id)
            
            # Migrate each API key
            for api_key in api_keys:
                api_key_data = {
                    "provider_id": api_key.provider_id,
                    "key_value": api_key.key_value,
                    "is_encrypted": api_key.is_encrypted,
                    "environment": api_key.environment,
                    "expires_at": api_key.expires_at
                }
                
                # Create API key in LLM Gateway
                gw_repo.create_api_key(api_key_data)
            
            # Get connection parameters for provider
            params = bo_repo.get_connection_parameters_by_provider_id(provider.provider_id)
            
            # Migrate each connection parameter
            for param in params:
                param_data = {
                    "provider_id": param.provider_id,
                    "param_name": param.param_name,
                    "param_value": param.param_value,
                    "is_sensitive": param.is_sensitive,
                    "environment": param.environment
                }
                
                # Create connection parameter in LLM Gateway
                gw_repo.set_connection_parameter(param_data)
        
        print(f"Successfully migrated {len(providers)} providers")
    finally:
        bo_db.close()
        gw_db.close()

if __name__ == "__main__":
    migrate_providers()
```

## Testing the Migration

### 1. Run Both Systems in Parallel

During the migration, run both the Backoffice Backend and the LLM Gateway in parallel to ensure that the migration is working correctly.

### 2. Verify Provider Data

Verify that all provider data has been migrated correctly:

```python
# Verify providers
bo_providers = bo_provider_service.get_all_providers()
gw_providers = gw_provider_service.get_all_providers()
assert len(bo_providers) == len(gw_providers)

# Verify models
for provider in bo_providers:
    bo_models = bo_provider_service.get_models_by_provider_id(provider["provider_id"])
    gw_models = gw_provider_service.get_models_by_provider_id(provider["provider_id"])
    assert len(bo_models) == len(gw_models)
```

### 3. Test LLM Requests

Test LLM requests using the LLM Gateway client:

```python
# Create a gateway client
config_loader = ConfigLoader(db)
gateway_config = config_loader.load_config_as_object()
provider_factory = ProviderFactory()
client = LLMGatewayClient(gateway_config, provider_factory, db)

# Create a request
llm_config = LLMConfig(model_identifier="gpt-4-turbo-preview")
context = InterventionContext(session_id="test-session")
llm_req = LLMRequest(
    prompt_content="Hello, world!",
    config=llm_config,
    initial_context=context
)

# Send the request
response = await client.generate(llm_req)
assert response.generated_content
```

## Rollback Plan

In case of issues during the migration, have a rollback plan in place:

1. Keep the original Backoffice Backend code and database intact
2. Implement feature flags to switch between the Backoffice Backend and LLM Gateway
3. Monitor the LLM Gateway for errors and performance issues

## Post-Migration Tasks

After the migration is complete:

1. Remove the old provider management code from the Backoffice Backend
2. Update documentation to reflect the new architecture
3. Train developers on using the LLM Gateway
4. Monitor the LLM Gateway for performance and errors

## Conclusion

By following this migration guide, you can successfully move provider management and LLM interaction functionality from the Backoffice Backend to the LLM Gateway. This will provide a more robust and feature-rich provider management system, as well as enhanced LLM interaction capabilities.
