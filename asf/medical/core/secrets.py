"""
Secret management for the Medical Research Synthesizer.
This module provides a secure way to manage secrets for the application.
It supports loading secrets from environment variables, .env files, or
external secret management services like HashiCorp Vault or AWS Secrets Manager.
"""
import os
import json
import logging
import base64
from typing import Optional
from pathlib import Path
import boto3
import hvac
from dotenv import load_dotenv
logger = logging.getLogger(__name__)
class SecretManager:
    """
    Secret manager for the Medical Research Synthesizer.
    This class provides a secure way to manage secrets for the application.
    It supports loading secrets from environment variables, .env files, or
    external secret management services like HashiCorp Vault or AWS Secrets Manager.
    """
    _instance = None
    def __new__(cls):
        """Implement singleton pattern.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        if cls._instance is None:
            cls._instance = super(SecretManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    def __init__(self):
        """Initialize the secret manager.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        if self._initialized:
            return
        self._initialized = True
        self._secrets = {}
        self._provider = os.environ.get("SECRET_PROVIDER", "env")
        if self._provider == "env":
            self._load_from_env()
        elif self._provider == "vault":
            self._load_from_vault()
        elif self._provider == "aws":
            self._load_from_aws()
        else:
            logger.warning(f"Unknown secret provider: {self._provider}, falling back to env")
            self._load_from_env()
    def _load_from_env(self):
        """Load secrets from environment variables or .env file.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        env_file = Path(os.environ.get("ENV_FILE", ".env"))
        if env_file.exists():
            load_dotenv(env_file)
        self._secrets = {
            "database": {
                "username": os.environ.get("DB_USERNAME", ""),
                "password": os.environ.get("DB_PASSWORD", ""),
            },
            "api": {
                "secret_key": os.environ.get("SECRET_KEY", ""),
            },
            "external": {
                "ncbi_api_key": os.environ.get("NCBI_API_KEY", ""),
            },
            "redis": {
                "password": os.environ.get("REDIS_PASSWORD", ""),
            },
            "graph": {
                "neo4j_password": os.environ.get("NEO4J_PASSWORD", ""),
            },
        }
    def _load_from_vault(self):
        """Load secrets from HashiCorp Vault.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        try:
            vault_url = os.environ.get("VAULT_ADDR", "http://localhost:8200")
            vault_token = os.environ.get("VAULT_TOKEN", "")
            vault_path = os.environ.get("VAULT_PATH", "secret/medical-research-synthesizer")
            client = hvac.Client(url=vault_url, token=vault_token)
            if not client.is_authenticated():
                logger.error("Failed to authenticate with Vault")
                self._load_from_env()
                return
            secret = client.secrets.kv.v2.read_secret_version(path=vault_path)
            if secret and "data" in secret and "data" in secret["data"]:
                self._secrets = secret["data"]["data"]
            else:
                logger.error("Failed to read secrets from Vault")
                self._load_from_env()
        except Exception as e:
            logger.error(f"Error loading secrets from Vault: {e}")
            self._load_from_env()
    def _load_from_aws(self):
        """Load secrets from AWS Secrets Manager.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        try:
            region_name = os.environ.get("AWS_REGION", "us-east-1")
            secret_name = os.environ.get("AWS_SECRET_NAME", "medical-research-synthesizer")
            session = boto3.session.Session()
            client = session.client(
                service_name='secretsmanager',
                region_name=region_name
            )
            get_secret_value_response = client.get_secret_value(
                SecretId=secret_name
            )
            if 'SecretString' in get_secret_value_response:
                secret = get_secret_value_response['SecretString']
                self._secrets = json.loads(secret)
            else:
                decoded_binary_secret = base64.b64decode(get_secret_value_response['SecretBinary'])
                self._secrets = json.loads(decoded_binary_secret)
        except Exception as e:
            logger.error(f"Error loading secrets from AWS Secrets Manager: {e}")
            self._load_from_env()
    def get_secret(self, category: str, name: str, default: Optional[str] = None) -> str:
        """
        Get a secret by category and name.
        Args:
            category: The category of the secret (e.g., "database", "api")
            name: The name of the secret (e.g., "password", "secret_key")
            default: The default value to return if the secret is not found
        Returns:
            The secret value or the default value if not found
        Get all secrets in a category.
        Args:
            category: The category of the secrets (e.g., "database", "api")
        Returns:
            A dictionary of secret names and values