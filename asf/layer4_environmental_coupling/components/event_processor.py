import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

class EventDrivenProcessor:
    """
    Optimized event processor with multi-stream parallel processing.
    Implements high-throughput event handling through concurrent processing streams.
    """
    def __init__(self, max_concurrency=16, stream_count=8):
        self.max_concurrency = max_concurrency
        self.stream_count = stream_count  # Number of parallel processing streams
        self.processing_semaphore = asyncio.Semaphore(max_concurrency)
        self.event_callback = None
        self.running = False
        self.stream_queues = [asyncio.Queue() for _ in range(stream_count)]  # Multiple streams
        self.stream_workers = []
        self.stream_stats = [{'processed': 0, 'average_time': 0.0} for _ in range(stream_count)]
        self.logger = logging.getLogger("ASF.Layer4.EventDrivenProcessor")
        
    async def initialize(self, callback):
        """Initialize with event processing callback."""
        self.event_callback = callback
        self.running = True
        self.logger.info(f"Initialized event processor with {self.stream_count} parallel streams")
        return True
        
    async def submit_event(self, event):
        """Submit an event for processing using adaptive stream selection."""
        if not self.running:
            return False
            
        # Select optimal stream based on load balancing
        stream_index = await self._select_optimal_stream(event)
        
        # Add event to selected stream queue
        await self.stream_queues[stream_index].put(event)
        return True
        
    async def _select_optimal_stream(self, event):
        """Select the optimal stream for an event based on current load and event type."""
        # Simple strategy: choose the stream with the shortest queue
        queue_sizes = [q.qsize() for q in self.stream_queues]
        
        # Consider event priority for queue selection
        if hasattr(event, 'priority') and event.priority > 0.7:
            # For high priority events, use the least loaded queue
            return queue_sizes.index(min(queue_sizes))
            
        # For regular events, use modulo by entity_id if available for affinity
        if hasattr(event, 'entity_id') and event.entity_id:
            # Consistent hashing for entity affinity (keeps related events on same stream)
            entity_hash = hash(event.entity_id) % self.stream_count
            
            # Only use this stream if it's not overloaded
            if queue_sizes[entity_hash] < 1.5 * min(queue_sizes):
                return entity_hash
                
        # Default to least loaded queue
        return queue_sizes.index(min(queue_sizes))
        
    async def run_processing_loop(self):
        """Start processing loops for all streams."""
        # Create and start a worker for each stream
        self.stream_workers = [
            asyncio.create_task(self._stream_processing_loop(i))
            for i in range(self.stream_count)
        ]
        
        self.logger.info(f"Started {self.stream_count} parallel processing streams")
        return self.stream_workers
        
    async def _stream_processing_loop(self, stream_index):
        """Processing loop for a single stream."""
        queue = self.stream_queues[stream_index]
        stats = self.stream_stats[stream_index]
        
        self.logger.info(f"Started processing loop for stream {stream_index}")
        
        while self.running:
            try:
                # Get next event
                event = await queue.get()
                
                # Process event with semaphore to limit total concurrency
                async with self.processing_semaphore:
                    start_time = time.time()
                    
                    # Process the event
                    if self.event_callback:
                        try:
                            result = await self.event_callback(event)
                            # Record successful processing
                            if hasattr(event, 'processed'):
                                event.processed = True
                            if hasattr(event, 'result'):
                                event.result = result
                        except Exception as e:
                            # Handle errors during event processing
                            self.logger.error(f"Error processing event {getattr(event, 'id', 'unknown')}: {str(e)}")
                            if hasattr(event, 'processed'):
                                event.processed = False
                            if hasattr(event, 'result'):
                                event.result = {'error': str(e), 'processed': False}
                    
                    processing_time = time.time() - start_time
                    
                    # Update statistics with exponential moving average
                    stats['processed'] += 1
                    stats['average_time'] = (stats['average_time'] * 0.95) + (processing_time * 0.05)
                    
                    # Log slow events
                    if processing_time > 1.0:  # More than 1 second
                        self.logger.warning(f"Slow event processing: {getattr(event, 'event_type', 'unknown')} took {processing_time:.2f}s")
                        
                # Mark task as done
                queue.task_done()
                
            except asyncio.CancelledError:
                self.logger.info(f"Stream {stream_index} processing loop cancelled")
                break
                
            except Exception as e:
                self.logger.error(f"Error in stream {stream_index} processing loop: {str(e)}")
                # Continue processing despite errors
                
        self.logger.info(f"Stream {stream_index} processing loop exited")
        
    async def perform_maintenance(self):
        """Perform periodic maintenance tasks."""
        # Calculate overall metrics
        total_processed = sum(stats['processed'] for stats in self.stream_stats)
        avg_times = [stats['average_time'] for stats in self.stream_stats]
        
        # Balance queues if needed
        await self._balance_queues()
        
        return {
            'stream_count': self.stream_count,
            'events_processed': total_processed,
            'stream_load': [q.qsize() for q in self.stream_queues],
            'avg_processing_times': avg_times
        }
        
    async def _balance_queues(self):
        """Balance workload across queues if imbalanced."""
        queue_sizes = [q.qsize() for q in self.stream_queues]
        max_size = max(queue_sizes)
        min_size = min(queue_sizes)
        
        # If imbalance is significant
        if max_size > min_size * 2 and max_size > 10:
            source_idx = queue_sizes.index(max_size)
            target_idx = queue_sizes.index(min_size)
            
            # Move some events (up to half of the difference)
            moves = (max_size - min_size) // 2
            
            for _ in range(moves):
                if not self.stream_queues[source_idx].empty():
                    event = await self.stream_queues[source_idx].get()
                    await self.stream_queues[target_idx].put(event)
                    self.stream_queues[source_idx].task_done()
                    
            self.logger.info(f"Balanced queues by moving {moves} events from stream {source_idx} to {target_idx}")
            
    async def get_metrics(self):
        """Get processor metrics."""
        return {
            'stream_count': self.stream_count,
            'events_processed': [stats['processed'] for stats in self.stream_stats],
            'avg_processing_times': [stats['average_time'] for stats in self.stream_stats],
            'current_queue_sizes': [q.qsize() for q in self.stream_queues]
        }
        
    async def stop(self):
        """Stop processing and shut down workers gracefully."""
        self.running = False
        
        # Cancel all worker tasks
        for worker in self.stream_workers:
            worker.cancel()
            
        # Wait for workers to complete
        if self.stream_workers:
            await asyncio.gather(*self.stream_workers, return_exceptions=True)
            
        self.logger.info("Event processor stopped")
        return True
