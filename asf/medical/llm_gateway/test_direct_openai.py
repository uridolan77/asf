#!/usr/bin/env python
# test_direct_openai.py

"""
Test script to verify direct connection to OpenAI API using the official SDK.
This bypasses the LLM Gateway to test the OpenAI API connection directly.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_direct_openai():
    """Test direct connection to OpenAI API using the official SDK."""
    
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
        # Import OpenAI SDK
        try:
            from openai import AsyncOpenAI
        except ImportError:
            logger.error("OpenAI SDK not installed. Please install it with 'pip install openai'")
            return False
        
        # Initialize client
        logger.info("Initializing OpenAI client directly...")
        client = AsyncOpenAI(api_key=api_key)
        
        # Send a simple request
        logger.info("Sending test request to OpenAI API directly...")
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, can you tell me what time it is?"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        # Check response
        logger.info(f"Response received successfully!")
        logger.info(f"Response ID: {response.id}")
        logger.info(f"Model: {response.model}")
        
        if response.choices:
            logger.info(f"Generated content: {response.choices[0].message.content}")
            logger.info(f"Finish reason: {response.choices[0].finish_reason}")
        
        if response.usage:
            logger.info(f"Usage: {response.usage.prompt_tokens} prompt tokens, "
                       f"{response.usage.completion_tokens} completion tokens, "
                       f"{response.usage.total_tokens} total tokens")
        
        return True
    
    except Exception as e:
        logger.exception(f"Error testing direct OpenAI connection: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting direct OpenAI connection test...")
    
    # Run the async test
    success = asyncio.run(test_direct_openai())
    
    if success:
        logger.info("✅ Direct OpenAI connection test PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ Direct OpenAI connection test FAILED!")
        sys.exit(1)
