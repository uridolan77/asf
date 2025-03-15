import time
import math

class AdaptiveTemporalMetadata:
    """
    Manages temporal aspects of knowledge entities with dynamic decay rates.
    Philosophical Influence: Bergson's time-consciousness and Heidegger's Dasein
    """
    def __init__(self, context_type="default"):
        self.creation_time = time.time()
        self.last_access_time = self.creation_time
        self.access_count = 1
        # Half-life constants for different context types (in seconds)
        self.half_lives = {
            "default": 86400,  # 24 hours
            "critical": 604800,  # 1 week
            "ephemeral": 3600  # 1 hour
        }
        self.context_type = context_type
    
    def update_access_time(self):
        """Update last access time and increment access counter"""
        self.last_access_time = time.time()
        self.access_count += 1
    
    def get_temporal_relevance(self) -> float:
        """
        Calculate temporal relevance based on adaptive logarithmic decay.
        Returns a value between 0 and 1 where 1 is most relevant.
        """
        current_time = time.time()
        time_since_creation = current_time - self.creation_time
        time_since_access = current_time - self.last_access_time
        # Adjust half-life based on access frequency
        access_factor = math.log2(max(2, self.access_count))
        adjusted_half_life = self.half_lives[self.context_type] * access_factor
        # Calculate decay using adapted half-life formula
        decay = math.exp(-time_since_access / adjusted_half_life)
        return max(0.01, min(1.0, decay))  # Ensure value between 0.01 and 1
    
    def set_context_type(self, context_type):
        """Update the context type to change temporal relevance calculation"""
        if context_type in self.half_lives:
            self.context_type = context_type
