"""
Test script to verify the LLM Gateway router fix.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the router
from bo.backend.api.routers.llm_gateway import get_gateway_client

def main():
    """Test the get_gateway_client function."""
    try:
        print("Testing get_gateway_client function...")
        client = get_gateway_client()
        print("Success! get_gateway_client function works.")
        print(f"Client: {client}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
