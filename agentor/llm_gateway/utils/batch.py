import asyncio
from typing import List, TypeVar, Generic, Callable, Awaitable, Any
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class BatchProcessor(Generic[T, R]):
    """Process items in batches."""
    
    def __init__(
        self,
        processor: Callable[[T], Awaitable[R]],
        batch_size: int = 10,
        max_concurrency: int = 5
    ):
        """Initialize the batch processor.
        
        Args:
            processor: The function to process each item
            batch_size: The maximum number of items to process in a batch
            max_concurrency: The maximum number of concurrent tasks
        """
        self.processor = processor
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
    
    async def process_batch(self, items: List[T]) -> List[R]:
        """Process a batch of items.
        
        Args:
            items: The items to process
            
        Returns:
            The processed items
        """
        # Split the items into batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        results = []
        for batch in batches:
            # Process the batch with limited concurrency
            batch_results = await self._process_with_concurrency(batch)
            results.extend(batch_results)
        
        return results
    
    async def _process_with_concurrency(self, items: List[T]) -> List[R]:
        """Process items with limited concurrency.
        
        Args:
            items: The items to process
            
        Returns:
            The processed items
        """
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def process_with_semaphore(item: T) -> R:
            async with semaphore:
                return await self.processor(item)
        
        # Create tasks for each item
        tasks = [process_with_semaphore(item) for item in items]
        
        # Wait for all tasks to complete
        return await asyncio.gather(*tasks)


class LLMBatchProcessor:
    """Process LLM requests in batches."""
    
    def __init__(self, llm_router: Any, batch_size: int = 10, max_concurrency: int = 5):
        """Initialize the LLM batch processor.
        
        Args:
            llm_router: The LLM router to use
            batch_size: The maximum number of requests to process in a batch
            max_concurrency: The maximum number of concurrent requests
        """
        self.llm_router = llm_router
        self.processor = BatchProcessor(
            processor=self._process_request,
            batch_size=batch_size,
            max_concurrency=max_concurrency
        )
    
    async def _process_request(self, request: Any) -> Any:
        """Process an LLM request.
        
        Args:
            request: The LLM request
            
        Returns:
            The LLM response
        """
        return await self.llm_router.generate(request)
    
    async def process_batch(self, requests: List[Any]) -> List[Any]:
        """Process a batch of LLM requests.
        
        Args:
            requests: The LLM requests
            
        Returns:
            The LLM responses
        """
        return await self.processor.process_batch(requests)
