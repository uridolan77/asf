import pytest
import asyncio
from unittest.mock import AsyncMock
import time

from agentor.llm_gateway.utils.batch import BatchProcessor, LLMBatchProcessor


@pytest.mark.asyncio
async def test_batch_processor():
    """Test the BatchProcessor class."""
    # Create a mock processor function
    async def mock_processor(item):
        await asyncio.sleep(0.1)  # Simulate some processing time
        return item * 2
    
    # Create a batch processor
    processor = BatchProcessor(
        processor=mock_processor,
        batch_size=5,
        max_concurrency=3
    )
    
    # Process a batch of items
    items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    results = await processor.process_batch(items)
    
    # Check the results
    assert results == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


@pytest.mark.asyncio
async def test_batch_processor_concurrency():
    """Test that the BatchProcessor respects the concurrency limit."""
    # Create a counter to track concurrent executions
    concurrent_count = 0
    max_concurrent = 0
    
    async def mock_processor(item):
        nonlocal concurrent_count, max_concurrent
        concurrent_count += 1
        max_concurrent = max(max_concurrent, concurrent_count)
        await asyncio.sleep(0.1)  # Simulate some processing time
        concurrent_count -= 1
        return item
    
    # Create a batch processor with a concurrency limit of 3
    processor = BatchProcessor(
        processor=mock_processor,
        batch_size=10,
        max_concurrency=3
    )
    
    # Process a batch of items
    items = list(range(10))
    await processor.process_batch(items)
    
    # Check that the concurrency limit was respected
    assert max_concurrent <= 3


@pytest.mark.asyncio
async def test_llm_batch_processor():
    """Test the LLMBatchProcessor class."""
    # Create a mock LLM router
    mock_router = AsyncMock()
    mock_router.generate.side_effect = lambda req: asyncio.sleep(0.1).then(lambda: f"Response to {req}")
    
    # Create an LLM batch processor
    processor = LLMBatchProcessor(
        llm_router=mock_router,
        batch_size=5,
        max_concurrency=3
    )
    
    # Process a batch of requests
    requests = ["Request 1", "Request 2", "Request 3", "Request 4", "Request 5"]
    results = await processor.process_batch(requests)
    
    # Check that the router was called for each request
    assert mock_router.generate.call_count == 5
    
    # Check the results
    assert len(results) == 5
    for i, result in enumerate(results):
        assert result == f"Response to {requests[i]}"
