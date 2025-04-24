"""
Run script for the LLM Management BO server.

This script provides a convenient way to start the LLM Management BO server
with the specified configuration.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLM Management BO Server")
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=int(os.environ.get("BOLLM_PORT", 8001)),
        help="Port to run the server on (default: 8001 or BOLLM_PORT env var)"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default=os.environ.get("BOLLM_HOST", "0.0.0.0"),
        help="Host to bind the server to (default: 0.0.0.0 or BOLLM_HOST env var)"
    )
    
    parser.add_argument(
        "--config", 
        type=str,
        default=os.environ.get("BOLLM_CONFIG"),
        help="Path to config file (default: BOLLM_CONFIG env var or built-in default)"
    )
    
    parser.add_argument(
        "--cache-dir", 
        type=str,
        default=os.environ.get("BOLLM_CACHE_DIR"),
        help="Directory to store cache files (default: BOLLM_CACHE_DIR env var or ~/.bollm/cache)"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with auto-reload"
    )
    
    return parser.parse_args()

def main():
    """Run the LLM Management BO server."""
    args = parse_args()
    
    # Set environment variables based on arguments
    if args.config:
        os.environ["BOLLM_CONFIG"] = str(args.config)
        
    if args.cache_dir:
        os.environ["BOLLM_CACHE_DIR"] = str(args.cache_dir)
        
    if args.no_cache:
        os.environ["BOLLM_CACHE_ENABLED"] = "false"
    else:
        os.environ["BOLLM_CACHE_ENABLED"] = "true"
    
    # Import uvicorn here to ensure environment variables are set first
    import uvicorn
    
    logger.info(f"Starting LLM Management BO server at {args.host}:{args.port}")
    if args.config:
        logger.info(f"Using config file: {args.config}")
    
    if args.no_cache:
        logger.info("Caching is disabled")
    elif args.cache_dir:
        logger.info(f"Cache directory: {args.cache_dir}")
    
    # Run the server
    uvicorn.run(
        "asf.bollm.backend.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.debug
    )

if __name__ == "__main__":
    main()