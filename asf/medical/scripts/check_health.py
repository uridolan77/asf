#!/usr/bin/env python
"""
Script to check the health of the Medical Research Synthesizer API.

This script sends a request to the health endpoint and reports the status.
It can be used as a simple monitoring tool or in CI/CD pipelines.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_health(base_url: str, timeout: int = 5) -> bool:
    """
    Check the health of the API.
    
    Args:
        base_url: Base URL of the API
        timeout: Request timeout in seconds
        
    Returns:
        True if the API is healthy, False otherwise
    """
    health_url = f"{base_url.rstrip('/')}/health"
    logger.info(f"Checking health at {health_url}")
    
    try:
        response = requests.get(health_url, timeout=timeout)
        response.raise_for_status()
        
        health_data = response.json()
        logger.info(f"Health check response: {json.dumps(health_data, indent=2)}")
        
        if health_data.get("status") == "ok":
            logger.info("API is healthy")
            return True
        else:
            logger.error(f"API is unhealthy: {health_data.get('status')}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking health: {str(e)}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check the health of the Medical Research Synthesizer API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--timeout", type=int, default=5, help="Request timeout in seconds")
    args = parser.parse_args()
    
    if check_health(args.url, args.timeout):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
