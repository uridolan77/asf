#!/usr/bin/env python
# test_with_db.py

"""
Test script to verify OpenAI API connection using the database configuration.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Set OpenAI API key for testing
os.environ["OPENAI_API_KEY"] = "sk-your-openai-api-key"

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
    LLMRequest,
    LLMConfig,
    InterventionContext,
    InterventionConfig,
)
from asf.medical.llm_gateway.core.client import LLMGatewayClient
from asf.medical.llm_gateway.db_utils import get_db_session

async def test_with_db():
    """Test OpenAI connection using the database configuration."""

    try:
        # Get database session
        db = get_db_session()

        # Create gateway client with database session
        client = LLMGatewayClient(db=db)

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
    logger.info("Starting OpenAI connection test with database configuration...")

    # Run the async test
    success = asyncio.run(test_with_db())

    if success:
        logger.info("✅ OpenAI connection test with database configuration PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ OpenAI connection test with database configuration FAILED!")
        sys.exit(1)
