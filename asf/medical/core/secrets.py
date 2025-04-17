"""
Secrets management for ASF Medical.

This module provides a simple secrets manager for storing and retrieving sensitive information
such as API keys and credentials.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SecretManager:
    """
    Manages secrets for the application.
    
    Supports loading secrets from:
    1. Environment variables
    2. Local secrets file
    3. Default values (for development only)
    """
    
    def __init__(self, secrets_file: Optional[str] = None):
        """
        Initialize the SecretManager.
        
        Args:
            secrets_file: Path to secrets file. If None, uses default location.
        """
        if secrets_file:
            self.secrets_file = Path(secrets_file)
        else:
            # Default location: ~/.asf/secrets.json
            home_dir = Path.home()
            self.secrets_file = home_dir / ".asf" / "secrets.json"
            
        self.secrets: Dict[str, Dict[str, str]] = {}
        self._load_secrets()
    
    def _load_secrets(self) -> None:
        """Load secrets from the secrets file if it exists."""
        if not self.secrets_file.exists():
            logger.warning(f"Secrets file not found at {self.secrets_file}. Creating a new one.")
            # Ensure directory exists
            self.secrets_file.parent.mkdir(parents=True, exist_ok=True)
            # Create empty secrets file
            self._save_secrets()
            return
        
        try:
            with open(self.secrets_file, 'r') as f:
                self.secrets = json.load(f)
            logger.debug(f"Loaded secrets from {self.secrets_file}")
        except Exception as e:
            logger.error(f"Failed to load secrets from {self.secrets_file}: {e}")
            self.secrets = {}
    
    def _save_secrets(self) -> None:
        """Save secrets to the secrets file."""
        try:
            self.secrets_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.secrets_file, 'w') as f:
                json.dump(self.secrets, f, indent=2)
            # Set appropriate permissions (read/write only for owner)
            os.chmod(self.secrets_file, 0o600)
            logger.debug(f"Saved secrets to {self.secrets_file}")
        except Exception as e:
            logger.error(f"Failed to save secrets to {self.secrets_file}: {e}")
    
    def get_secret(self, category: str, name: str) -> Optional[str]:
        """
        Get a secret value.
        
        Args:
            category: Secret category (e.g., 'llm', 'database')
            name: Secret name (e.g., 'openai_api_key')
            
        Returns:
            Secret value or None if not found
        """
        # First try environment variable with combined name
        env_var = f"{category.upper()}_{name.upper()}"
        value = os.environ.get(env_var)
        if value:
            logger.debug(f"Found secret {category}:{name} in environment variable {env_var}")
            return value
        
        # Then check the secrets file
        category_dict = self.secrets.get(category, {})
        value = category_dict.get(name)
        if value:
            logger.debug(f"Found secret {category}:{name} in secrets file")
            return value
        
        logger.warning(f"Secret {category}:{name} not found in environment or secrets file")
        return None
    
    def set_secret(self, category: str, name: str, value: str) -> None:
        """
        Set a secret value.
        
        Args:
            category: Secret category (e.g., 'llm', 'database')
            name: Secret name (e.g., 'openai_api_key')
            value: Secret value
        """
        if category not in self.secrets:
            self.secrets[category] = {}
        
        self.secrets[category][name] = value
        self._save_secrets()
        logger.debug(f"Saved secret {category}:{name}")
    
    def delete_secret(self, category: str, name: str) -> bool:
        """
        Delete a secret.
        
        Args:
            category: Secret category
            name: Secret name
            
        Returns:
            True if secret was deleted, False otherwise
        """
        if category in self.secrets and name in self.secrets[category]:
            del self.secrets[category][name]
            # Remove empty categories
            if not self.secrets[category]:
                del self.secrets[category]
            self._save_secrets()
            logger.debug(f"Deleted secret {category}:{name}")
            return True
        
        logger.warning(f"Secret {category}:{name} not found, nothing to delete")
        return False