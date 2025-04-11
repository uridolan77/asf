"""
Module description.

This module provides functionality for...
"""
import time
from collections import deque
import numpy as np

class TemporalSequence:
    Represents a sequence of temporal events for pattern detection.
    Optimized for performance with efficient data structures and operations.
    def __init__(self, max_length=100):
        """
        __init__ function.
        
        This function provides functionality for...
        Args:
            max_length: Description of max_length
        """
        self.events = deque(maxlen=max_length)
        self.timestamps = deque(maxlen=max_length)
        self._max_length = max_length
        self._last_window_time = 0
        self._last_window_size = 0
        self._last_window_result = []
    
    def add_event(self, event, timestamp=None):
        """Add event to sequence with optional timestamp

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
        if timestamp is None:
            timestamp = time.time()
        
        events = self.events
        timestamps = self.timestamps
        
        events.append(event)
        timestamps.append(timestamp)
        
        self._last_window_time = 0
        
        return len(events)  # Return new size for convenience
    
    def get_events_in_window(self, window_size):
        """Return events within the specified time window from now

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
        current_time = time.time()
        
        if (current_time - self._last_window_time < 0.1 and  # Cache valid for 100ms
            window_size == self._last_window_size and
            self._last_window_result):
            return self._last_window_result
        
        timestamps = self.timestamps
        events = self.events
        
        cutoff_time = current_time - window_size
        
        if not timestamps:
            return []
        
        if timestamps[0] >= cutoff_time:
            result = list(events)
        else:
            left, right = 0, len(timestamps) - 1
            cutoff_idx = 0
            
            while left <= right:
                mid = (left + right) // 2
                if timestamps[mid] < cutoff_time:
                    left = mid + 1
                    cutoff_idx = left  # First index >= cutoff_time
                else:
                    right = mid - 1
            
            result = list(events)[cutoff_idx:]
        
        self._last_window_time = current_time
        self._last_window_size = window_size
        self._last_window_result = result
        
        return result
    
    def get_latest_events(self, n=1):
        """Get the n most recent events

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
        if not self.events:
            return []
        
        if n >= len(self.events):
            return list(self.events)
        
        return list(self.events)[-n:]
    
    def clear(self):
        """Clear all events and timestamps

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
        self.events.clear()
        self.timestamps.clear()
        self._last_window_time = 0
        self._last_window_result = []
        
    def __len__(self):
        """Return the number of events in sequence

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
        return len(self.events)
    
    def get_statistics(self):
        """Return statistical information about the sequence

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
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
