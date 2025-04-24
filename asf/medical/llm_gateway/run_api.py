"""
Run script for LLM Gateway API.

This script starts the LLM Gateway API.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from asf.medical.llm_gateway.api.main import start

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """
    Start the LLM Gateway API.
    """
    try:
        logger.info("Starting LLM Gateway API...")
        start()
    except Exception as e:
        logger.error(f"Error starting LLM Gateway API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
