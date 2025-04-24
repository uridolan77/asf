"""
Test the LLM Gateway with a mock provider.
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

async def test_llm_gateway():
    """Test the LLM Gateway with a mock provider."""
    try:
        logger.info("Importing modules...")
        from asf.medical.llm_gateway.core.client import LLMGatewayClient
        from asf.medical.llm_gateway.core.factory import ProviderFactory
        from asf.medical.llm_gateway.core.models import (
            GatewayConfig, LLMRequest, LLMConfig, InterventionContext
        )
        
        # Create a minimal gateway config
        logger.info("Creating gateway config...")
        gateway_config = GatewayConfig(
            gateway_id="test_gateway",
            default_provider="mock_provider",
            allowed_providers=["mock_provider"],
            providers={
                "mock_provider": {
                    "provider_id": "mock_provider",
                    "provider_type": "mock",
                    "connection_params": {"simulate_delay_ms": 200},
                    "models": {"gpt-3.5-turbo": {}}
                }
            }
        )
        
        # Create provider factory
        logger.info("Creating provider factory...")
        provider_factory = ProviderFactory()
        
        # Create LLM Gateway client
        logger.info("Creating LLM Gateway client...")
        client = LLMGatewayClient(gateway_config, provider_factory)
        logger.info("LLM Gateway client created successfully")
        
        # Create a request
        logger.info("Creating request...")
        request = LLMRequest(
            prompt_content="Hello, world!",
            config=LLMConfig(
                model_identifier="gpt-3.5-turbo"
            ),
            initial_context=InterventionContext(
                request_id="test_request"
            )
        )
        
        # Generate a response
        logger.info("Generating response...")
        response = await client.generate(request)
        
        logger.info(f"Response generated successfully: {response.generated_content}")
        logger.info(f"Finish reason: {response.finish_reason}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing LLM Gateway: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_llm_gateway())
