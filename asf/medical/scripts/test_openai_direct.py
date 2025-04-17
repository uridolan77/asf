#!/usr/bin/env python
"""
Direct OpenAI API key test script.
This script tests the OpenAI API directly without going through the gateway.
"""

import openai
import os

# Set your API key here
API_KEY = os.environ.get("OPENAI_API_KEY", "PLACEHOLDER_API_KEY")  # Get from environment or use placeholder

# Set the API key directly in the OpenAI client
client = openai.OpenAI(api_key=API_KEY)

# Try a simple API call
try:
    print("Sending test request to OpenAI API...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, can you tell me what time it is?"}
        ],
        max_tokens=100
    )

    print("\nAPI call successful!")
    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens used: {response.usage.total_tokens}")

except Exception as e:
    print(f"\nAPI call failed with error: {e}")

print("\nTest complete.")