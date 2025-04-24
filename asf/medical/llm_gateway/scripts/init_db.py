"""
Database initialization script for LLM Gateway.

This script initializes the database tables for the LLM Gateway.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from asf.medical.llm_gateway.db.session import init_db
from asf.medical.llm_gateway.core.config_loader import ConfigLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """
    Initialize the database tables for the LLM Gateway.
    """
    parser = argparse.ArgumentParser(description="Initialize the database tables for the LLM Gateway.")
    parser.add_argument("--config", help="Path to the configuration file")
    parser.add_argument("--sync", action="store_true", help="Synchronize database with configuration file")
    args = parser.parse_args()
    
    try:
        # Initialize database
        logger.info("Initializing database tables...")
        init_db()
        logger.info("Database tables initialized successfully")
        
        # Synchronize database with configuration file if requested
        if args.sync:
            logger.info("Synchronizing database with configuration file...")
            config_loader = ConfigLoader(config_path=args.config)
            result = config_loader.sync_db_with_yaml()
            if result:
                logger.info("Database synchronized successfully with configuration file")
            else:
                logger.error("Failed to synchronize database with configuration file")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
