#!/usr/bin/env python
# test_with_secrets.py

"""
Test script to verify OpenAI API connection using the Secret Manager.
"""

import asyncio
import logging
import os
import sys
import yaml
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from asf.bo.backend.api.routers.llm.utils import load_config
from datetime import datetime, timezone
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    logger.info(f"Added {project_root} to Python path")

# Import gateway components
from asf.medical.core.secrets import SecretManager
from asf.medical.llm_gateway.core.models import (
    GatewayConfig,
    InterventionConfig,
    InterventionContext,
    LLMConfig,
    LLMRequest,
    ProviderConfig,
)
from asf.medical.llm_gateway.core.client import LLMGatewayClient
from asf.medical.llm_gateway.core.factory import ProviderFactory
from asf.medical.llm_gateway.providers.openai_client import OpenAIClient

async def test_with_secrets():
    """Test OpenAI connection using the Secret Manager."""

    try:
        # Load configuration
        config_path = os.path.join(project_root, "bo", "backend", "config", "llm", "llm_gateway_config.yaml")
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found at {config_path}")
            return False

        logger.info(f"Loading configuration from {config_path}")
        # Load configuration using the utility function that handles local config
        config_dict = load_config(config_path)

        # Initialize Secret Manager
        secret_manager = SecretManager()

        # Extract API key from configuration
        providers = config_dict.get("additional_config", {}).get("providers", {})
        openai_provider = providers.get("openai_gpt4_default", {})

        # Get API key from configuration first
        api_key = openai_provider.get("connection_params", {}).get("api_key")
        if api_key:
            logger.info("Using API key from configuration")
        else:
            logger.error("API key not found in configuration")
            return False

        # Store API key in Secret Manager
        api_key_secret = openai_provider.get("connection_params", {}).get("api_key_secret")
        if api_key_secret and ":" in api_key_secret:
            category, name = api_key_secret.split(":", 1)
            # Ensure the category exists
            if category not in secret_manager._secrets:
                secret_manager._secrets[category] = {}
            # Store the API key
            secret_manager._secrets[category][name] = api_key
            logger.info(f"Stored API key in Secret Manager: {category}:{name}")

            # Verify it was stored correctly
            stored_key = secret_manager.get_secret(category, name)
            if stored_key == api_key:
                logger.info(f"Verified API key in Secret Manager: {category}:{name}")
            else:
                logger.error(f"Failed to store API key in Secret Manager: {category}:{name}")
                return False

        # Now use the Secret Manager for the test
        if api_key_secret and ":" in api_key_secret:
            category, name = api_key_secret.split(":", 1)
            # Get the API key from the Secret Manager
            api_key_from_secret = secret_manager.get_secret(category, name)
            logger.info(f"Retrieved API key from Secret Manager for test: {category}:{name}")

            # Use the API key from the Secret Manager for the test
            if api_key_from_secret:
                api_key = api_key_from_secret
                logger.info("Using API key from Secret Manager for the test")

        # Create gateway config
        gateway_config = GatewayConfig(**config_dict)

        # Create provider factory
        provider_factory = ProviderFactory()

        # Create gateway client
        client = LLMGatewayClient(gateway_config, provider_factory)

        # Create request
        request_id = f"test_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        # Create intervention context
        context = InterventionContext(
            request_id=request_id,
            conversation_history=[],
            user_id="test_user",
            session_id="test_session",
            timestamp_start=datetime.now(timezone.utc),
            intervention_config=InterventionConfig(
                enabled_pre_interventions=[],
                enabled_post_interventions=[],
                fail_open=True
            ),
            intervention_data={}
        )

        # Create LLM config
        llm_config = LLMConfig(
            model_identifier="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            system_prompt="You are a helpful assistant."
        )

        # Create the request
        request = LLMRequest(
            version="1.0",
            initial_context=context,
            config=llm_config,
            prompt_content="Hello, can you tell me what time it is?",
            tools=[]
        )

        # Send request
        logger.info(f"Sending test request with request ID: {request_id}")
        response = await client.generate(request)

        # Check response
        if response.error_details:
            logger.error(f"Error in response: {response.error_details.code} - {response.error_details.message}")
            logger.error(f"Provider details: {response.error_details.provider_error_details}")
            return False

        logger.info(f"Response received successfully!")
        logger.info(f"Generated content: {response.generated_content}")
        logger.info(f"Finish reason: {response.finish_reason}")

        if response.usage:
            logger.info(f"Usage: {response.usage.prompt_tokens} prompt tokens, "
                       f"{response.usage.completion_tokens} completion tokens, "
                       f"{response.usage.total_tokens} total tokens")

        if response.performance_metrics:
            logger.info(f"LLM latency: {response.performance_metrics.llm_latency_ms:.2f}ms")
            logger.info(f"Total duration: {response.performance_metrics.total_duration_ms:.2f}ms")

        # Clean up
        await client.close()

        return True

    except Exception as e:
        logger.exception(f"Error testing OpenAI connection: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting OpenAI connection test with Secret Manager...")

    # Run the async test
    success = asyncio.run(test_with_secrets())

    if success:
        logger.info("✅ OpenAI connection test with Secret Manager PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ OpenAI connection test with Secret Manager FAILED!")
        sys.exit(1)
