#!/usr/bin/env python
# run_test_with_key.py

"""
Script to set the OpenAI API key as an environment variable and run the test.
"""

import os
import sys
import logging
import asyncio
import subprocess
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

# Import gateway components
from asf.medical.llm_gateway.core.models import (
    LLMRequest,
    LLMConfig,
    InterventionContext,
    InterventionConfig,
)
from asf.medical.llm_gateway.core.client import LLMGatewayClient
from asf.medical.core.secrets import SecretManager
from datetime import datetime, timezone

async def test_with_key(api_key):
    """Test OpenAI connection using the provided API key."""
    try:
        # Set the API key as an environment variable
        os.environ["OPENAI_API_KEY"] = api_key
        logger.info(f"Set OPENAI_API_KEY environment variable")
        
        # Set the API key in the Secret Manager
        secret_manager = SecretManager()
        secret_manager._secrets.setdefault("llm", {})
        secret_manager._secrets["llm"]["openai_api_key"] = api_key
        logger.info(f"Set llm:openai_api_key in Secret Manager")
        
        # Create gateway client
        client = LLMGatewayClient()
        
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

def main():
    """Main function."""
    # Get API key from command line
    if len(sys.argv) < 2:
        logger.error("Please provide your OpenAI API key as a command line argument")
        sys.exit(1)
    
    api_key = sys.argv[1]
    
    # Mask the API key for logging
    if len(api_key) > 8:
        masked_key = f"{api_key[:5]}...{api_key[-4:]}"
    else:
        masked_key = "***MASKED***"
    logger.info(f"Using OpenAI API key: {masked_key}")
    
    # Run the async test
    success = asyncio.run(test_with_key(api_key))
    
    if success:
        logger.info("✅ OpenAI connection test PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ OpenAI connection test FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
