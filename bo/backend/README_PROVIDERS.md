# Provider Management System

This document describes the provider management system implemented in the ASF backend.

## Overview

The provider management system allows users to:

1. Store and manage LLM providers (OpenAI, Anthropic, etc.)
2. Securely store API keys for these providers
3. Configure connection parameters for providers
4. Assign users to providers with specific roles

## Database Schema

The system uses the following database tables:

- `providers`: Stores information about LLM providers
- `provider_models`: Stores information about the models available for each provider
- `api_keys`: Securely stores API keys for providers
- `connection_parameters`: Stores connection parameters for providers
- `users_providers`: Many-to-many association table between users and providers

## Setup Instructions

### 1. Create the Association Table

Run the migration script to create the users_providers association table:

```bash
python -m asf.bo.backend.db.migrations.run_add_users_providers --user <db_user> --password <db_password> --database <db_name>
```

### 2. Migrate Configuration to Database

If you haven't already, run the migration script to migrate your configuration from YAML files to the database:

```bash
python -m asf.bo.backend.db.migrations.migrate_config_to_db --user <db_user> --password <db_password> --database <db_name> --create-tables --migrate-data
```

### 3. Test the LLM Gateway with Database Configuration

Run the test script to verify that the LLM Gateway can use the database for configuration:

```bash
python -m asf.medical.llm_gateway.test_with_db
```

## API Endpoints

### Provider Management

- `GET /api/providers`: Get all providers
- `GET /api/providers/{provider_id}`: Get a provider by ID
- `POST /api/providers`: Create a new provider
- `PUT /api/providers/{provider_id}`: Update a provider
- `DELETE /api/providers/{provider_id}`: Delete a provider

### API Key Management

- `GET /api/providers/{provider_id}/api-keys`: Get all API keys for a provider
- `POST /api/providers/{provider_id}/api-keys`: Create a new API key for a provider
- `GET /api/providers/{provider_id}/api-keys/{key_id}/value`: Get the actual API key value

### Connection Parameter Management

- `GET /api/providers/{provider_id}/connection-params`: Get all connection parameters for a provider
- `POST /api/providers/{provider_id}/connection-params`: Set a connection parameter for a provider

### User-Provider Management

- `POST /api/user-providers/assign`: Assign a user to a provider with a specific role
- `DELETE /api/user-providers/{provider_id}/users/{user_id}`: Remove a user from a provider
- `GET /api/user-providers/providers/{provider_id}/users`: Get all users assigned to a provider
- `GET /api/user-providers/users/{user_id}/providers`: Get all providers assigned to a user
- `GET /api/user-providers/check-access/{provider_id}`: Check if the current user has access to a provider

### Configuration Management

- `GET /api/config`: Get all configurations
- `GET /api/config/{config_key}`: Get a configuration by key
- `POST /api/config`: Create a new configuration
- `PUT /api/config/{config_key}`: Update a configuration
- `DELETE /api/config/{config_key}`: Delete a configuration

### User Settings Management

- `GET /api/config/user/settings`: Get all settings for a user
- `GET /api/config/user/settings/{setting_key}`: Get a user setting
- `POST /api/config/user/settings`: Create a new user setting
- `PUT /api/config/user/settings/{setting_key}`: Update a user setting
- `DELETE /api/config/user/settings/{setting_key}`: Delete a user setting

## Security Considerations

- API keys are encrypted in the database using a secure encryption key
- Access to providers is controlled through the users_providers association table
- Only administrators can assign users to providers
- Users can only access providers they have been assigned to
- Audit logs track changes to sensitive data

## Usage Examples

### Assigning a User to a Provider

```python
import requests

# Assign a user to a provider
response = requests.post(
    "http://localhost:8000/api/user-providers/assign",
    json={
        "user_id": 1,
        "provider_id": "openai",
        "role": "admin"
    },
    headers={"Authorization": f"Bearer {token}"}
)
```

### Getting Providers for a User

```python
import requests

# Get all providers for a user
response = requests.get(
    "http://localhost:8000/api/user-providers/users/1/providers",
    headers={"Authorization": f"Bearer {token}"}
)
providers = response.json()
```

### Creating a New Provider

```python
import requests

# Create a new provider
response = requests.post(
    "http://localhost:8000/api/providers",
    json={
        "provider_id": "anthropic",
        "display_name": "Anthropic",
        "provider_type": "llm",
        "description": "Anthropic Claude API",
        "enabled": True,
        "models": [
            {
                "model_id": "claude-3-opus-20240229",
                "display_name": "Claude 3 Opus",
                "model_type": "chat",
                "context_window": 200000,
                "max_tokens": 4096,
                "enabled": True
            }
        ],
        "api_key": {
            "key_value": "your-api-key",
            "is_encrypted": True,
            "environment": "development"
        }
    },
    headers={"Authorization": f"Bearer {token}"}
)
```
