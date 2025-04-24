"""
Test the mock provider creation in the InterventionManager.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variable for OpenAI API key
os.environ["OPENAI_API_KEY"] = "dummy_key"

async def test_mock_provider():
    """Test creating a mock provider."""
    try:
        logger.info("Importing modules...")
        from asf.medical.llm_gateway.interventions.manager import InterventionManager
        from asf.medical.llm_gateway.core.factory import ProviderFactory
        from asf.medical.llm_gateway.core.models import GatewayConfig
        from asf.medical.llm_gateway.providers.mock import MockProvider

        logger.info("Creating mock provider directly...")
        # Test creating a mock provider directly
        from asf.medical.llm_gateway.core.models import ProviderConfig
        mock_provider_id = "test_mock_provider"
        mock_config = ProviderConfig(
            provider_id=mock_provider_id,
            provider_type="mock",
            connection_params={"simulate_delay_ms": 200},
            models={"gpt-3.5-turbo": {}}
        )
        direct_provider = MockProvider(mock_provider_id, mock_config)
        logger.info(f"Direct mock provider created successfully: {direct_provider.provider_id}")

        # Create a minimal gateway config
        logger.info("Creating gateway config...")
        gateway_config = GatewayConfig(
            gateway_id="test_gateway",
            default_provider="mock_provider",
            allowed_providers=["mock_provider"]
        )

        # Create provider factory
        logger.info("Creating provider factory...")
        provider_factory = ProviderFactory()

        # Create intervention manager
        logger.info("Creating intervention manager...")
        manager = InterventionManager(provider_factory, gateway_config)
        logger.info("InterventionManager created successfully")

        # Create mock provider
        logger.info("Creating mock provider through manager...")
        model_identifier = "gpt-3.5-turbo"
        provider, provider_id = await manager._create_mock_provider(model_identifier)

        logger.info(f"Mock provider created successfully: {provider_id}")
        logger.info(f"Provider config: {provider.provider_config}")
        logger.info(f"Provider models: {provider.provider_config.models}")

        return True
    except Exception as e:
        logger.error(f"Error creating mock provider: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_mock_provider())
