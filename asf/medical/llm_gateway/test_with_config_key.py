#!/usr/bin/env python
# test_with_config_key.py

"""
Test script to verify OpenAI API connection using the API key from the configuration file.
"""

import asyncio
import logging
import os
import sys
import yaml
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

async def test_with_config_key():
    """Test OpenAI connection using the API key from the configuration file."""
    
    try:
        # Load configuration
        config_path = os.path.join(project_root, "bo", "backend", "config", "llm", "llm_gateway_config.yaml")
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found at {config_path}")
            return False
        
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract API key from configuration
        providers = config_dict.get("additional_config", {}).get("providers", {})
        openai_provider = providers.get("openai_gpt4_default", {})
        api_key = openai_provider.get("connection_params", {}).get("api_key")
        
        if not api_key:
            logger.error("API key not found in configuration")
            return False
        
        # Set API key in environment variable
        os.environ["OPENAI_API_KEY"] = api_key
        logger.info(f"Using API key from configuration: {api_key[:5]}...{api_key[-4:]}")
        
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
    logger.info("Starting OpenAI connection test with config key...")
    
    # Run the async test
    success = asyncio.run(test_with_config_key())
    
    if success:
        logger.info("✅ OpenAI connection test PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ OpenAI connection test FAILED!")
        sys.exit(1)
