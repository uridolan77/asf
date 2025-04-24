"""
Cache Warming Module for LLM Gateway

This module provides functionality to pre-populate the cache with common queries
to improve cache hit rates from the start.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional

from asf.medical.llm_gateway.core.models import LLMRequest, InitialContext, ModelConfig
from asf.medical.llm_gateway.cache.cache_manager import get_cache_manager
from asf.medical.llm_gateway.gateway import LLMGateway

logger = logging.getLogger(__name__)

class CacheWarmer:
    """
    Utility class for warming up the cache with common queries.
    """
    
    def __init__(
        self, 
        gateway: LLMGateway,
        queries_file: Optional[str] = None,
        query_list: Optional[List[Dict[str, Any]]] = None,
        concurrency: int = 2
    ):
        """
        Initialize cache warmer.
        
        Args:
            gateway: LLM Gateway instance
            queries_file: Path to JSON file containing queries (optional)
            query_list: List of query configurations (optional)
            concurrency: Maximum number of concurrent requests
        """
        self.gateway = gateway
        self.queries_file = queries_file
        self.query_list = query_list or []
        self.concurrency = max(1, concurrency)
        self.cache_manager = get_cache_manager()
        
    async def load_queries_from_file(self) -> List[Dict[str, Any]]:
        """
        Load queries from a JSON file.
        
        Returns:
            List of query configurations
        """
        if not self.queries_file or not os.path.exists(self.queries_file):
            return []
            
        try:
            with open(self.queries_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading queries from {self.queries_file}: {str(e)}")
            return []
            
    async def warm_cache(self) -> Dict[str, Any]:
        """
        Warm the cache by executing common queries.
        
        Returns:
            Dictionary with warming results
        """
        # Load queries from file if provided
        if self.queries_file:
            file_queries = await self.load_queries_from_file()
            self.query_list.extend(file_queries)
            
        if not self.query_list:
            logger.warning("No queries provided for cache warming")
            return {
                "status": "warning",
                "message": "No queries provided for cache warming",
                "queries_processed": 0,
                "queries_successful": 0,
                "queries_failed": 0
            }
            
        # Initialize semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.concurrency)
        
        # Start cache warming
        total_queries = len(self.query_list)
        successful = 0
        failed = 0
        
        logger.info(f"Starting cache warming with {total_queries} queries (concurrency: {self.concurrency})")
        
        # Create and execute tasks
        tasks = []
        for query_config in self.query_list:
            task = asyncio.create_task(self._execute_query(semaphore, query_config))
            tasks.append(task)
            
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful and failed queries
        for result in results:
            if isinstance(result, Exception):
                failed += 1
            elif result:
                successful += 1
            else:
                failed += 1
                
        logger.info(f"Cache warming complete: {successful}/{total_queries} queries successful")
        
        return {
            "status": "success" if failed == 0 else "partial",
            "message": f"Cache warming complete: {successful}/{total_queries} queries successful",
            "queries_processed": total_queries,
            "queries_successful": successful,
            "queries_failed": failed
        }
        
    async def _execute_query(self, semaphore: asyncio.Semaphore, query_config: Dict[str, Any]) -> bool:
        """
        Execute a single query for cache warming.
        
        Args:
            semaphore: Semaphore for concurrency control
            query_config: Query configuration
            
        Returns:
            True if successful, False otherwise
        """
        async with semaphore:
            try:
                # Extract query parameters
                prompt = query_config.get("prompt", "")
                if not prompt:
                    logger.warning("Skipping query with empty prompt")
                    return False
                    
                model_identifier = query_config.get("model", "")
                temperature = query_config.get("temperature", 0.0)
                max_tokens = query_config.get("max_tokens", 500)
                system_prompt = query_config.get("system_prompt", "")
                
                # Create request
                request_id = f"warming_{hash(prompt)}"
                
                config = ModelConfig(
                    model_identifier=model_identifier,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt
                )
                
                context = InitialContext(
                    request_id=request_id,
                    conversation_id=None,
                    conversation_history=[]
                )
                
                request = LLMRequest(
                    prompt_content=prompt,
                    config=config,
                    initial_context=context
                )
                
                # Execute query
                logger.info(f"Warming cache with query: '{prompt[:50]}...' (model: {model_identifier})")
                response = await self.gateway.generate(request)
                
                if response and response.generated_content:
                    logger.info(f"Successfully warmed cache with query: '{prompt[:50]}...'")
                    return True
                else:
                    logger.warning(f"Failed to warm cache with query: '{prompt[:50]}...'")
                    return False
                    
            except Exception as e:
                logger.error(f"Error warming cache with query: {str(e)}")
                return False

async def warm_cache_from_file(
    gateway: LLMGateway,
    queries_file: str,
    concurrency: int = 2
) -> Dict[str, Any]:
    """
    Convenience function to warm the cache from a queries file.
    
    Args:
        gateway: LLM Gateway instance
        queries_file: Path to JSON file containing queries
        concurrency: Maximum number of concurrent requests
        
    Returns:
        Dictionary with warming results
    """
    warmer = CacheWarmer(
        gateway=gateway,
        queries_file=queries_file,
        concurrency=concurrency
    )
    
    return await warmer.warm_cache()

async def warm_cache_with_queries(
    gateway: LLMGateway,
    queries: List[Dict[str, Any]],
    concurrency: int = 2
) -> Dict[str, Any]:
    """
    Convenience function to warm the cache with a list of queries.
    
    Args:
        gateway: LLM Gateway instance
        queries: List of query configurations
        concurrency: Maximum number of concurrent requests
        
    Returns:
        Dictionary with warming results
    """
    warmer = CacheWarmer(
        gateway=gateway,
        query_list=queries,
        concurrency=concurrency
    )
    
    return await warmer.warm_cache()

# Example cache warming query list format:
# [{
#     "prompt": "What are the symptoms of COVID-19?",
#     "model": "gpt-4-turbo",
#     "temperature": 0.0,
#     "max_tokens": 500,
#     "system_prompt": "You are a helpful medical assistant."
# }]