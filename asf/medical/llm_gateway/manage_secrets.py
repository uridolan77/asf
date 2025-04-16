#!/usr/bin/env python
# manage_secrets.py

"""
Utility script for managing LLM Gateway secrets.
This script provides commands for setting, getting, and listing secrets.
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    logger.info(f"Added {project_root} to Python path")

# Import secret manager
from asf.medical.core.secrets import SecretManager

def set_secret(args):
    """Set a secret in the Secret Manager."""
    if ":" not in args.key:
        logger.error("Secret key must be in the format 'category:name'")
        return False
    
    category, name = args.key.split(":", 1)
    
    # Initialize Secret Manager
    secret_manager = SecretManager()
    
    # Set secret
    secret_manager._secrets.setdefault(category, {})
    secret_manager._secrets[category][name] = args.value
    
    logger.info(f"Secret '{args.key}' set successfully")
    return True

def get_secret(args):
    """Get a secret from the Secret Manager."""
    if ":" not in args.key:
        logger.error("Secret key must be in the format 'category:name'")
        return False
    
    category, name = args.key.split(":", 1)
    
    # Initialize Secret Manager
    secret_manager = SecretManager()
    
    # Get secret
    value = secret_manager.get_secret(category, name)
    
    if value:
        if args.mask:
            # Mask the value but keep first and last few characters
            if len(value) > 8:
                masked_value = f"{value[:5]}...{value[-4:]}"
            else:
                masked_value = "***MASKED***"
            logger.info(f"Secret '{args.key}': {masked_value}")
        else:
            logger.info(f"Secret '{args.key}': {value}")
        return True
    else:
        logger.error(f"Secret '{args.key}' not found")
        return False

def list_secrets(args):
    """List all secrets in the Secret Manager."""
    # Initialize Secret Manager
    secret_manager = SecretManager()
    
    # List secrets
    for category, secrets in secret_manager._secrets.items():
        logger.info(f"Category: {category}")
        for name, value in secrets.items():
            if args.mask:
                # Mask the value but keep first and last few characters
                if isinstance(value, str) and len(value) > 8:
                    masked_value = f"{value[:5]}...{value[-4:]}"
                else:
                    masked_value = "***MASKED***"
                logger.info(f"  {name}: {masked_value}")
            else:
                logger.info(f"  {name}: {value}")
    
    return True

def import_from_config(args):
    """Import secrets from the LLM Gateway configuration."""
    # Load configuration
    config_path = args.config_path or os.path.join(project_root, "bo", "backend", "config", "llm", "llm_gateway_config.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at {config_path}")
        return False
    
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Initialize Secret Manager
    secret_manager = SecretManager()
    
    # Extract API keys from configuration
    providers = config_dict.get("additional_config", {}).get("providers", {})
    imported_count = 0
    
    for provider_id, provider_config in providers.items():
        connection_params = provider_config.get("connection_params", {})
        
        # Check for API key
        api_key = connection_params.get("api_key")
        api_key_secret = connection_params.get("api_key_secret")
        
        if api_key and api_key_secret and ":" in api_key_secret:
            category, name = api_key_secret.split(":", 1)
            
            # Set secret
            secret_manager._secrets.setdefault(category, {})
            secret_manager._secrets[category][name] = api_key
            
            logger.info(f"Imported secret '{api_key_secret}' for provider '{provider_id}'")
            imported_count += 1
    
    logger.info(f"Imported {imported_count} secrets from configuration")
    return True

def export_to_env(args):
    """Export secrets to environment variables."""
    # Initialize Secret Manager
    secret_manager = SecretManager()
    
    # Export secrets
    exported_count = 0
    
    for category, secrets in secret_manager._secrets.items():
        for name, value in secrets.items():
            # Convert to environment variable format
            env_var = f"{category.upper()}_{name.upper()}"
            
            # Set environment variable
            os.environ[env_var] = value
            
            logger.info(f"Exported secret '{category}:{name}' to environment variable '{env_var}'")
            exported_count += 1
    
    logger.info(f"Exported {exported_count} secrets to environment variables")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Manage LLM Gateway secrets")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Set secret command
    set_parser = subparsers.add_parser("set", help="Set a secret")
    set_parser.add_argument("key", help="Secret key in the format 'category:name'")
    set_parser.add_argument("value", help="Secret value")
    set_parser.set_defaults(func=set_secret)
    
    # Get secret command
    get_parser = subparsers.add_parser("get", help="Get a secret")
    get_parser.add_argument("key", help="Secret key in the format 'category:name'")
    get_parser.add_argument("--mask", action="store_true", help="Mask the secret value")
    get_parser.set_defaults(func=get_secret)
    
    # List secrets command
    list_parser = subparsers.add_parser("list", help="List all secrets")
    list_parser.add_argument("--mask", action="store_true", help="Mask the secret values")
    list_parser.set_defaults(func=list_secrets)
    
    # Import from config command
    import_parser = subparsers.add_parser("import", help="Import secrets from the LLM Gateway configuration")
    import_parser.add_argument("--config-path", help="Path to the LLM Gateway configuration file")
    import_parser.set_defaults(func=import_from_config)
    
    # Export to environment variables command
    export_parser = subparsers.add_parser("export", help="Export secrets to environment variables")
    export_parser.set_defaults(func=export_to_env)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if hasattr(args, "func"):
        success = args.func(args)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
