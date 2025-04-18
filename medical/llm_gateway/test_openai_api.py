#!/usr/bin/env python
# test_openai_api.py

"""
Simple test script to verify the OpenAI API key is working.
"""

import os
import sys
import logging
import asyncio
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_openai_api():
    """Test the OpenAI API with a simple request."""
    try:
        # Use a real OpenAI API key for testing
        api_key = "sk-your-real-openai-api-key"
        
        # Create OpenAI client
        client = AsyncOpenAI(api_key=api_key)
        
        # Make a simple request
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, can you tell me what time it is?"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        # Print response
        logger.info(f"Response: {response.choices[0].message.content}")
        logger.info(f"Usage: {response.usage.prompt_tokens} prompt tokens, {response.usage.completion_tokens} completion tokens, {response.usage.total_tokens} total tokens")
        
        return True
    
    except Exception as e:
        logger.exception(f"Error testing OpenAI API: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing OpenAI API...")
    
    # Run the async test
    success = asyncio.run(test_openai_api())
    
    if success:
        logger.info("✅ OpenAI API test PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ OpenAI API test FAILED!")
        sys.exit(1)
