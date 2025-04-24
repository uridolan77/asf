#!/usr/bin/env python
# test_gateway_with_local_config.py

"""
Test script to verify the LLM gateway can use the API key from the local configuration.
"""

import os
import sys
import logging
import asyncio
import yaml
from pathlib import Path
from datetime import datetime, timezone

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info(f"Added {project_root} to Python path")

# Import gateway components
from asf.medical.llm_gateway.core.models import (
    GatewayConfig,
    InterventionConfig,
    InterventionContext,
    LLMConfig,
    LLMRequest,
)
from asf.medical.llm_gateway.core.client import LLMGatewayClient
from asf.medical.llm_gateway.core.factory import ProviderFactory

async def test_gateway_with_local_config():
    """Test the LLM gateway with the API key from the local configuration."""
    try:
        # Get configuration directory
        config_dir = os.path.join(project_root, "bo", "backend", "config", "llm")
        base_config_path = os.path.join(config_dir, "llm_gateway_config.yaml")
        local_config_path = os.path.join(config_dir, "llm_gateway_config.local.yaml")
        
        if not os.path.exists(base_config_path):
            logger.error(f"Base configuration file not found at {base_config_path}")
            return False
        
        if not os.path.exists(local_config_path):
            logger.error(f"Local configuration file not found at {local_config_path}")
            return False
        
        logger.info(f"Loading base configuration from {base_config_path}")
        with open(base_config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        logger.info(f"Loading local configuration from {local_config_path}")
        with open(local_config_path, 'r') as f:
            local_config = yaml.safe_load(f)
        
        # Merge configurations
        if local_config:
            # Deep merge additional_config
            if "additional_config" in local_config and "additional_config" in config_dict:
                for key, value in local_config["additional_config"].items():
                    if key in config_dict["additional_config"] and isinstance(value, dict) and isinstance(config_dict["additional_config"][key], dict):
                        # Merge nested dictionaries
                        for subkey, subvalue in value.items():
                            if subkey in config_dict["additional_config"][key] and isinstance(subvalue, dict) and isinstance(config_dict["additional_config"][key][subkey], dict):
                                # Merge nested dictionaries
                                for subsubkey, subsubvalue in subvalue.items():
                                    config_dict["additional_config"][key][subkey][subsubkey] = subsubvalue
                            else:
                                config_dict["additional_config"][key][subkey] = subvalue
                    else:
                        config_dict["additional_config"][key] = value
            
            # Merge other top-level keys
            for key, value in local_config.items():
                if key != "additional_config":
                    config_dict[key] = value
        
        # Extract API key from configuration
        providers = config_dict.get("additional_config", {}).get("providers", {})
        openai_provider = providers.get("openai_gpt4_default", {})
        api_key = openai_provider.get("connection_params", {}).get("api_key")
        
        if api_key:
            # Mask the API key for logging
            if len(api_key) > 8:
                masked_key = f"{api_key[:5]}...{api_key[-4:]}"
            else:
                masked_key = "***MASKED***"
            logger.info(f"Using API key from configuration: {masked_key}")
        else:
            logger.error("API key not found in configuration")
            return False
        
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
        logger.exception(f"Error testing LLM gateway: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing LLM gateway with local configuration...")
    
    # Run the async test
    success = asyncio.run(test_gateway_with_local_config())
    
    if success:
        logger.info("✅ LLM gateway test PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ LLM gateway test FAILED!")
        sys.exit(1)
