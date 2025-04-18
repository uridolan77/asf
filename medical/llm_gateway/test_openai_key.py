#!/usr/bin/env python
# test_openai_key.py

"""
Script to test the OpenAI API key by making a simple request.
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

async def test_openai_key(api_key=None):
    """Test the OpenAI API key by making a simple request."""
    try:
        # Get the API key from the environment or parameter
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            logger.error("OpenAI API key not found")
            return False
        
        # Mask the API key for logging
        if len(api_key) > 8:
            masked_key = f"{api_key[:5]}...{api_key[-4:]}"
        else:
            masked_key = "***MASKED***"
        logger.info(f"Using OpenAI API key: {masked_key}")
        
        # Initialize the OpenAI client
        client = AsyncOpenAI(api_key=api_key)
        
        # Make a simple request
        logger.info("Making a simple request to OpenAI API...")
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, what time is it?"}
            ],
            max_tokens=50
        )
        
        # Check the response
        if response and response.choices and response.choices[0].message.content:
            logger.info(f"Response received: {response.choices[0].message.content}")
            logger.info(f"Usage: {response.usage.prompt_tokens} prompt tokens, {response.usage.completion_tokens} completion tokens, {response.usage.total_tokens} total tokens")
            return True
        else:
            logger.error("No valid response received")
            return False
    
    except Exception as e:
        logger.exception(f"Error testing OpenAI API key: {e}")
        return False

def main():
    """Main function."""
    logger.info("Testing OpenAI API key...")
    
    # Get API key from command line if provided
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run the async test
    success = asyncio.run(test_openai_key(api_key))
    
    if success:
        logger.info("✅ OpenAI API key test PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ OpenAI API key test FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
