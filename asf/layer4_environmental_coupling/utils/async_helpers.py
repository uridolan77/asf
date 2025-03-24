import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

class AsyncEventQueue:
    """
    Asynchronous queue for coupling events with priority support.
    Provides ordered, priority-based event processing with monitoring capabilities.
    """
    def __init__(self, max_size=0):
        """
        Initialize the async event queue.
        
        Args:
            max_size: Maximum queue size (0 for unlimited)
        """
        self.queue = asyncio.PriorityQueue(maxsize=max_size)
        self.event_count = 0
        self.total_submitted = 0
        self.total_processed = 0
        self.processing_times = []
        self.priority_distribution = defaultdict(int)
        self.logger = logging.getLogger("ASF.Layer4.AsyncEventQueue")
        
    async def put(self, event, priority=None):
        """
        Add an event to the queue with priority support.
        
        Args:
            event: The event to add to the queue
            priority: Optional priority override (0-1, higher is higher priority)
                     If None, uses event.priority if available
        
        Returns:
            True if successful
        """
        # Determine priority
        if priority is None and hasattr(event, 'priority'):
            priority = event.priority
        else:
            priority = 0.5  # Default priority
            
        # Record priority for metrics
        priority_bin = round(priority * 10) / 10  # Round to nearest 0.1
        self.priority_distribution[priority_bin] += 1
        
        # Use counter to maintain FIFO order for same priority
        self.event_count += 1
        self.total_submitted += 1
        
        # Invert priority so lower values are processed first
        inverted_priority = 1.0 - priority
        
        # Put in queue with (priority, count, event) structure
        await self.queue.put((inverted_priority, self.event_count, event))
        return True
        
    async def get(self):
        """
        Get the next event from the queue based on priority.
        
        Returns:
            The next event
        """
        # Get from queue, ignoring priority and count
        _, _, event = await self.queue.get()
        return event
        
    def task_done(self):
        """Mark a task as done."""
        self.queue.task_done()
        self.total_processed += 1
        
    async def get_with_timeout(self, timeout=1.0):
        """
        Get an event with timeout.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Event or None if timeout
        """
        try:
            # Use wait_for with timeout
            _, _, event = await asyncio.wait_for(self.queue.get(), timeout)
            return event
        except asyncio.TimeoutError:
            return None
            
    async def put_batch(self, events, base_priority=0.5):
        """
        Add multiple events to the queue.
        
        Args:
            events: List of events to add
            base_priority: Base priority to use if event has no priority
            
        Returns:
            Number of events added
        """
        for event in events:
            priority = getattr(event, 'priority', base_priority)
            await self.put(event, priority)
        return len(events)
        
    def qsize(self):
        """Get current queue size."""
        return self.queue.qsize()
        
    def empty(self):
        """Check if queue is empty."""
        return self.queue.empty()
        
    def full(self):
        """Check if queue is full."""
        return self.queue.full()
        
    async def drain(self):
        """
        Drain the queue by processing all pending events.
        
        Returns:
            Number of events drained
        """
        drained = 0
        while not self.queue.empty():
            await self.queue.get()
            self.queue.task_done()
            drained += 1
        return drained
        
    def get_metrics(self):
        """
        Get queue metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            'current_size': self.queue.qsize(),
            'total_submitted': self.total_submitted,
            'total_processed': self.total_processed,
            'backlog': self.total_submitted - self.total_processed,
            'priority_distribution': dict(self.priority_distribution)
        }
        
    async def wait_for_completion(self, timeout=None):
        """
        Wait for all current items to be processed.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            True if completed, False if timeout
        """
        try:
            await asyncio.wait_for(self.queue.join(), timeout)
            return True
        except asyncio.TimeoutError:
            return False
