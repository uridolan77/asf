#!/usr/bin/env python
# test_gateway_stack.py

"""
Test script to verify the full LLM Gateway stack with OpenAI.
This script tests the entire call stack from LLMGatewayClient through interventions to the OpenAI API.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

async def test_gateway_stack():
    """Test the full LLM Gateway stack with OpenAI."""

    # Check if API key is set or use a test key from command line args
    api_key = os.environ.get("OPENAI_API_KEY")

    # Allow passing API key as command line argument for testing
    if not api_key and len(sys.argv) > 1:
        api_key = sys.argv[1]
        os.environ["OPENAI_API_KEY"] = api_key

    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set. Please set it or pass as command line argument.")
        return False

    logger.info(f"Using OpenAI API key: {api_key[:5]}...{api_key[-4:]}")

    try:
        # 1. Create minimal configs
        provider_config = ProviderConfig(
            provider_id="openai_test",
            provider_type="openai",
            description="Test OpenAI provider",
            models=["gpt-3.5-turbo"],
            connection_params={
                "api_key_env_var": "OPENAI_API_KEY",
                "max_retries": 2,
                "timeout_seconds": 30,
            }
        )

        gateway_config = GatewayConfig(
            gateway_id="test_gateway",
            description="Test Gateway",
            default_provider="openai_test",
            allowed_providers=["openai_test"],
            default_timeout_seconds=30,
            max_retries=2,
            retry_delay_seconds=1,
            providers={"openai_test": provider_config},
            additional_config={
                "enabled_interventions": [],  # No interventions for this test
                "max_concurrent_batch_requests": 5
            }
        )

        # 2. Initialize provider factory and gateway client
        logger.info("Initializing LLM Gateway client...")
        provider_factory = ProviderFactory()
        client = LLMGatewayClient(gateway_config, provider_factory)

        # 3. Create a simple request
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

        # 4. Call generate method through the full stack
        logger.info(f"Sending test request through the full LLM Gateway stack with request ID: {request_id}")
        response = await client.generate(request)

        # 5. Check response
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
            if response.performance_metrics.pre_processing_duration_ms:
                logger.info(f"Pre-processing duration: {response.performance_metrics.pre_processing_duration_ms:.2f}ms")
            if response.performance_metrics.post_processing_duration_ms:
                logger.info(f"Post-processing duration: {response.performance_metrics.post_processing_duration_ms:.2f}ms")

        # 6. Clean up
        await client.close()

        return True

    except Exception as e:
        logger.exception(f"Error testing LLM Gateway stack: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting LLM Gateway stack test...")

    # Run the async test
    success = asyncio.run(test_gateway_stack())

    if success:
        logger.info("✅ LLM Gateway stack test PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ LLM Gateway stack test FAILED!")
        sys.exit(1)
