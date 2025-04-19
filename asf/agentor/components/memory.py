class Memory:
    """Memory system for agents to store and retrieve information."""
    
    def __init__(self, capacity: int = None):
        """
        Initialize a new Memory system.
        
        Args:
            capacity: Optional maximum number of items to remember (None for unlimited)
        """
        self.capacity = capacity
        self.short_term = {}  # For temporary storage
        self.long_term = {}   # For persistent storage
        self.episodic = []    # For sequences of events
        self.semantic = {}    # For knowledge and facts
        
    def store_short_term(self, key: str, value: Any):
        """Store information in short-term memory."""
        self.short_term[key] = {
            'value': value,
            'timestamp': time.time()
        }
        self._enforce_capacity(self.short_term)
    
    def store_long_term(self, key: str, value: Any):
        """Store information in long-term memory."""
        self.long_term[key] = {
            'value': value,
            'timestamp': time.time(),
            'access_count': 0
        }
        self._enforce_capacity(self.long_term)
    
    def add_episode(self, episode: Dict[str, Any]):
        """Add an episodic memory (event)."""
        episode['timestamp'] = time.time()
        self.episodic.append(episode)
        if self.capacity and len(self.episodic) > self.capacity:
            self.episodic.pop(0)  # Remove oldest
    
    def store_knowledge(self, key: str, value: Any, certainty: float = 1.0):
        """Store semantic knowledge with certainty level."""
        self.semantic[key] = {
            'value': value,
            'timestamp': time.time(),
            'certainty': max(0.0, min(1.0, certainty)),  # Clamp between 0 and 1
            'access_count': 0
        }
        self._enforce_capacity(self.semantic)
    
    def retrieve_short_term(self, key: str) -> Any:
        """Retrieve information from short-term memory."""
        if key in self.short_term:
            return self.short_term[key]['value']
        return None
    
    def retrieve_long_term(self, key: str) -> Any:
        """Retrieve information from long-term memory."""
        if key in self.long_term:
            self.long_term[key]['access_count'] += 1
            return self.long_term[key]['value']
        return None
    
    def retrieve_episodes(self, start_time: float = None, end_time: float = None, 
                         filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve episodic memories with optional filters.
        
        Args:
            start_time: Optional start time for filtering episodes
            end_time: Optional end time for filtering episodes
            filters: Optional dictionary of key-value pairs to filter by
            
        Returns:
            List of matching episodes
        """
        results = self.episodic
        
        if start_time:
            results = [ep for ep in results if ep['timestamp'] >= start_time]
            
        if end_time:
            results = [ep for ep in results if ep['timestamp'] <= end_time]
            
        if filters:
            for key, value in filters.items():
                results = [ep for ep in results if ep.get(key) == value]
                
        return results
    
    def retrieve_knowledge(self, key: str) -> Dict[str, Any]:
        """
        Retrieve semantic knowledge.
        
        Returns:
            Dictionary with value and certainty
        """
        if key in self.semantic:
            self.semantic[key]['access_count'] += 1
            return {
                'value': self.semantic[key]['value'],
                'certainty': self.semantic[key]['certainty']
            }
        return None
    
    def _enforce_capacity(self, storage: Dict):
        """
        Enforce memory capacity by removing least accessed items.
        
        Args:
            storage: Dictionary to enforce capacity on
        """
        if not self.capacity or len(storage) <= self.capacity:
            return
            
        # Remove items with lowest access counts
        items_to_remove = len(storage) - self.capacity
        if items_to_remove > 0:
            sorted_items = sorted(
                storage.items(), 
                key=lambda x: x[1].get('access_count', 0)
            )
            for i in range(items_to_remove):
                if i < len(sorted_items):
                    del storage[sorted_items[i][0]]
    
    def decay_short_term(self, decay_time: float = 300):
        """
        Remove short-term memories older than decay_time seconds.
        
        Args:
            decay_time: Time in seconds after which to forget
        """
        current_time = time.time()
        keys_to_remove = []
        
        for key, data in self.short_term.items():
            if current_time - data['timestamp'] > decay_time:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.short_term[key]
    
    def consolidate_to_long_term(self, threshold: int = 3):
        """
        Move frequently accessed short-term memories to long-term.
        
        Args:
            threshold: Number of accesses needed to consolidate
        """
        for key, data in list(self.short_term.items()):
            if data.get('access_count', 0) >= threshold:
                self.store_long_term(key, data['value'])
                del self.short_term[key]