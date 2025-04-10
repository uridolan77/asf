import time
from collections import deque
import numpy as np

class TemporalSequence:
    """
    Represents a sequence of temporal events for pattern detection.
    Optimized for performance with efficient data structures and operations.
    """
    def __init__(self, max_length=100):
        self.events = deque(maxlen=max_length)
        self.timestamps = deque(maxlen=max_length)
        self._max_length = max_length
        # Cache for window lookups
        self._last_window_time = 0
        self._last_window_size = 0
        self._last_window_result = []
    
    def add_event(self, event, timestamp=None):
        """Add event to sequence with optional timestamp"""
        if timestamp is None:
            timestamp = time.time()
        
        # Local references for performance
        events = self.events
        timestamps = self.timestamps
        
        events.append(event)
        timestamps.append(timestamp)
        
        # Invalidate cache when new events are added
        self._last_window_time = 0
        
        return len(events)  # Return new size for convenience
    
    def get_events_in_window(self, window_size):
        """Return events within the specified time window from now"""
        current_time = time.time()
        
        # Check if we can use cached result
        if (current_time - self._last_window_time < 0.1 and  # Cache valid for 100ms
            window_size == self._last_window_size and
            self._last_window_result):
            return self._last_window_result
        
        # Local references for performance
        timestamps = self.timestamps
        events = self.events
        
        # Optimize: use binary search to find cutoff index
        cutoff_time = current_time - window_size
        
        # If sequence is empty
        if not timestamps:
            return []
        
        # If all events are within window
        if timestamps[0] >= cutoff_time:
            result = list(events)
        else:
            # Find index using binary search approximation
            left, right = 0, len(timestamps) - 1
            cutoff_idx = 0
            
            while left <= right:
                mid = (left + right) // 2
                if timestamps[mid] < cutoff_time:
                    left = mid + 1
                    cutoff_idx = left  # First index >= cutoff_time
                else:
                    right = mid - 1
            
            # Extract events from cutoff_idx to end
            result = list(events)[cutoff_idx:]
        
        # Cache the result
        self._last_window_time = current_time
        self._last_window_size = window_size
        self._last_window_result = result
        
        return result
    
    def get_latest_events(self, n=1):
        """Get the n most recent events"""
        if not self.events:
            return []
        
        if n >= len(self.events):
            return list(self.events)
        
        return list(self.events)[-n:]
    
    def clear(self):
        """Clear all events and timestamps"""
        self.events.clear()
        self.timestamps.clear()
        self._last_window_time = 0
        self._last_window_result = []
        
    def __len__(self):
        """Return the number of events in sequence"""
        return len(self.events)
    
    def get_statistics(self):
        """Return statistical information about the sequence"""
        if not self.timestamps:
            return {
                "count": 0,
                "oldest": None,
                "newest": None,
                "time_span": 0,
                "avg_interval": 0
            }
        
        ts_array = np.array(self.timestamps)
        oldest = float(ts_array.min())
        newest = float(ts_array.max())
        
        return {
            "count": len(self.timestamps),
            "oldest": oldest,
            "newest": newest,
            "time_span": newest - oldest,
            "avg_interval": (newest - oldest) / max(1, len(self.timestamps) - 1)
        }
