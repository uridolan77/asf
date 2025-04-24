"""
Migration script for transferring provider data from Backoffice Backend to LLM Gateway.

This script migrates provider data, including models, API keys, and connection parameters,
from the Backoffice Backend to the LLM Gateway.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def migrate_providers(bo_db_url=None, gw_db_url=None, encryption_key=None, dry_run=False):
    """
    Migrate providers from Backoffice Backend to LLM Gateway.
    
    Args:
        bo_db_url: Database URL for Backoffice Backend
        gw_db_url: Database URL for LLM Gateway
        encryption_key: Encryption key for sensitive data
        dry_run: Whether to perform a dry run without making changes
    """
    try:
        # Set environment variables for database URLs
        if bo_db_url:
            os.environ["BO_DATABASE_URL"] = bo_db_url
        if gw_db_url:
            os.environ["LLM_GATEWAY_DATABASE_URL"] = gw_db_url
        if encryption_key:
            os.environ["LLM_GATEWAY_ENCRYPTION_KEY"] = encryption_key
        
        # Import database sessions
        from asf.bo.backend.db.session import get_db as get_bo_db
        from asf.medical.llm_gateway.db.session import get_db_session as get_gw_db
        
        # Import repositories
        from asf.bo.backend.repositories.provider_repository import ProviderRepository as BOProviderRepository
        from asf.medical.llm_gateway.repositories.provider_repository import ProviderRepository as GWProviderRepository
        
        # Get database sessions
        bo_db = next(get_bo_db())
        gw_db = get_gw_db()
        
        try:
            # Get repositories
            bo_repo = BOProviderRepository(bo_db)
            gw_repo = GWProviderRepository(gw_db, encryption_key.encode() if encryption_key else None)
            
            # Get all providers from Backoffice Backend
            logger.info("Getting providers from Backoffice Backend...")
            providers = bo_repo.get_all_providers()
            logger.info(f"Found {len(providers)} providers in Backoffice Backend")
            
            # Migrate each provider
            for provider in providers:
                logger.info(f"Migrating provider '{provider.provider_id}'...")
                
                # Convert provider data
                provider_data = {
                    "provider_id": provider.provider_id,
                    "provider_type": provider.provider_type,
                    "display_name": provider.display_name,
                    "description": provider.description,
                    "enabled": provider.enabled
                }
                
                # Add connection params if available
                if hasattr(provider, "connection_params") and provider.connection_params:
                    try:
                        if isinstance(provider.connection_params, str):
                            provider_data["connection_params"] = json.loads(provider.connection_params)
                        else:
                            provider_data["connection_params"] = provider.connection_params
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse connection_params for provider '{provider.provider_id}'")
                
                # Add request settings if available
                if hasattr(provider, "request_settings") and provider.request_settings:
                    try:
                        if isinstance(provider.request_settings, str):
                            provider_data["request_settings"] = json.loads(provider.request_settings)
                        else:
                            provider_data["request_settings"] = provider.request_settings
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse request_settings for provider '{provider.provider_id}'")
                
                # Create provider in LLM Gateway if not a dry run
                if not dry_run:
                    try:
                        # Check if provider already exists
                        existing_provider = gw_repo.get_provider_by_id(provider.provider_id)
                        if existing_provider:
                            logger.info(f"Provider '{provider.provider_id}' already exists in LLM Gateway, updating...")
                            gw_provider = gw_repo.update_provider(provider.provider_id, provider_data)
                        else:
                            logger.info(f"Creating provider '{provider.provider_id}' in LLM Gateway...")
                            gw_provider = gw_repo.create_provider(provider_data)
                    except Exception as e:
                        logger.error(f"Error creating/updating provider '{provider.provider_id}' in LLM Gateway: {e}")
                        continue
                
                # Get models for provider
                logger.info(f"Getting models for provider '{provider.provider_id}'...")
                models = bo_repo.get_models_by_provider_id(provider.provider_id)
                logger.info(f"Found {len(models)} models for provider '{provider.provider_id}'")
                
                # Migrate each model
                for model in models:
                    logger.info(f"Migrating model '{model.model_id}' for provider '{provider.provider_id}'...")
                    
                    # Convert model data
                    model_data = {
                        "model_id": model.model_id,
                        "provider_id": model.provider_id,
                        "display_name": model.display_name,
                        "model_type": model.model_type if hasattr(model, "model_type") else "chat",
                        "context_window": model.context_window if hasattr(model, "context_window") else None,
                        "max_tokens": model.max_tokens if hasattr(model, "max_tokens") else None,
                        "enabled": model.enabled if hasattr(model, "enabled") else True
                    }
                    
                    # Add capabilities if available
                    if hasattr(model, "capabilities") and model.capabilities:
                        try:
                            if isinstance(model.capabilities, str):
                                model_data["capabilities"] = json.loads(model.capabilities)
                            else:
                                model_data["capabilities"] = model.capabilities
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse capabilities for model '{model.model_id}'")
                    
                    # Add parameters if available
                    if hasattr(model, "parameters") and model.parameters:
                        try:
                            if isinstance(model.parameters, str):
                                model_data["parameters"] = json.loads(model.parameters)
                            else:
                                model_data["parameters"] = model.parameters
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse parameters for model '{model.model_id}'")
                    
                    # Create model in LLM Gateway if not a dry run
                    if not dry_run:
                        try:
                            # Check if model already exists
                            existing_model = gw_repo.get_model_by_id(model.model_id, model.provider_id)
                            if existing_model:
                                logger.info(f"Model '{model.model_id}' already exists in LLM Gateway, updating...")
                                gw_model = gw_repo.update_model(model.model_id, model.provider_id, model_data)
                            else:
                                logger.info(f"Creating model '{model.model_id}' in LLM Gateway...")
                                gw_model = gw_repo.create_model(model_data)
                        except Exception as e:
                            logger.error(f"Error creating/updating model '{model.model_id}' in LLM Gateway: {e}")
                            continue
                
                # Get API keys for provider
                logger.info(f"Getting API keys for provider '{provider.provider_id}'...")
                api_keys = bo_repo.get_api_keys_by_provider_id(provider.provider_id)
                logger.info(f"Found {len(api_keys)} API keys for provider '{provider.provider_id}'")
                
                # Migrate each API key
                for api_key in api_keys:
                    logger.info(f"Migrating API key for provider '{provider.provider_id}'...")
                    
                    # Convert API key data
                    api_key_data = {
                        "provider_id": api_key.provider_id,
                        "key_value": api_key.key_value,
                        "is_encrypted": api_key.is_encrypted if hasattr(api_key, "is_encrypted") else True,
                        "environment": api_key.environment if hasattr(api_key, "environment") else "development",
                        "expires_at": api_key.expires_at if hasattr(api_key, "expires_at") else None
                    }
                    
                    # Create API key in LLM Gateway if not a dry run
                    if not dry_run:
                        try:
                            logger.info(f"Creating API key for provider '{provider.provider_id}' in LLM Gateway...")
                            gw_api_key = gw_repo.create_api_key(api_key_data)
                        except Exception as e:
                            logger.error(f"Error creating API key for provider '{provider.provider_id}' in LLM Gateway: {e}")
                            continue
                
                # Get connection parameters for provider
                logger.info(f"Getting connection parameters for provider '{provider.provider_id}'...")
                params = bo_repo.get_connection_parameters_by_provider_id(provider.provider_id)
                logger.info(f"Found {len(params)} connection parameters for provider '{provider.provider_id}'")
                
                # Migrate each connection parameter
                for param in params:
                    logger.info(f"Migrating connection parameter '{param.param_name}' for provider '{provider.provider_id}'...")
                    
                    # Convert connection parameter data
                    param_data = {
                        "provider_id": param.provider_id,
                        "param_name": param.param_name,
                        "param_value": param.param_value,
                        "is_sensitive": param.is_sensitive if hasattr(param, "is_sensitive") else False,
                        "environment": param.environment if hasattr(param, "environment") else "development"
                    }
                    
                    # Create connection parameter in LLM Gateway if not a dry run
                    if not dry_run:
                        try:
                            logger.info(f"Creating connection parameter '{param.param_name}' for provider '{provider.provider_id}' in LLM Gateway...")
                            gw_param = gw_repo.set_connection_parameter(param_data)
                        except Exception as e:
                            logger.error(f"Error creating connection parameter '{param.param_name}' for provider '{provider.provider_id}' in LLM Gateway: {e}")
                            continue
            
            logger.info(f"Successfully migrated {len(providers)} providers")
        finally:
            bo_db.close()
            gw_db.close()
    except Exception as e:
        logger.error(f"Error migrating providers: {e}")
        raise

def main():
    """
    Main function for the migration script.
    """
    parser = argparse.ArgumentParser(description="Migrate providers from Backoffice Backend to LLM Gateway")
    parser.add_argument("--bo-db-url", help="Database URL for Backoffice Backend")
    parser.add_argument("--gw-db-url", help="Database URL for LLM Gateway")
    parser.add_argument("--encryption-key", help="Encryption key for sensitive data")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without making changes")
    args = parser.parse_args()
    
    migrate_providers(
        bo_db_url=args.bo_db_url,
        gw_db_url=args.gw_db_url,
        encryption_key=args.encryption_key,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
