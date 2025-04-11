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
        if priority is None and hasattr(event, 'priority'):
            priority = event.priority
        else:
            priority = 0.5  # Default priority
            
        priority_bin = round(priority * 10) / 10  # Round to nearest 0.1
        self.priority_distribution[priority_bin] += 1
        
        self.event_count += 1
        self.total_submitted += 1
        
        inverted_priority = 1.0 - priority
        
        await self.queue.put((inverted_priority, self.event_count, event))
        return True
        
    async def get(self):
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
        Add multiple events to the queue.
        
        Args:
            events: List of events to add
            base_priority: Base priority to use if event has no priority
            
        Returns:
            Number of events added
        return self.queue.qsize()
        
    def empty(self):
        """Check if queue is empty."""
        return self.queue.empty()
        
    def full(self):
        """Check if queue is full."""
        return self.queue.full()
        
    async def drain(self):
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
        try:
            await asyncio.wait_for(self.queue.join(), timeout)
            return True
        except asyncio.TimeoutError:
            return False
