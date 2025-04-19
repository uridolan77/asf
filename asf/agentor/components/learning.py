class LearningModule:
    """Module for implementing learning capabilities in agents."""
    
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.knowledge = {}
    
    def update(self, state, action, reward, new_state):
        """
        Update knowledge based on experience.
        
        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            new_state: State after action
        """
        # Simple Q-learning implementation
        state_key = self._hash_state(state)
        new_state_key = self._hash_state(new_state)
        
        # Initialize if needed
        if state_key not in self.knowledge:
            self.knowledge[state_key] = {}
        if action not in self.knowledge[state_key]:
            self.knowledge[state_key][action] = 0
            
        # Calculate max future value
        future_value = 0
        if new_state_key in self.knowledge:
            future_actions = self.knowledge[new_state_key]
            if future_actions:
                future_value = max(future_actions.values())
        
        # Update Q-value
        old_value = self.knowledge[state_key][action]
        self.knowledge[state_key][action] = old_value + self.learning_rate * (
            reward + future_value - old_value
        )
    
    def best_action(self, state):
        """
        Determine the best action for a given state.
        
        Args:
            state: Current state
            
        Returns:
            Best action based on learned knowledge, or None if no knowledge
        """
        state_key = self._hash_state(state)
        if state_key not in self.knowledge or not self.knowledge[state_key]:
            return None
            
        # Find action with maximum value
        actions = self.knowledge[state_key]
        return max(actions.items(), key=lambda x: x[1])[0]
    
    def _hash_state(self, state):
        """
        Convert a state dictionary to a hashable representation.
        
        Args:
            state: State dictionary
            
        Returns:
            Hashable representation of the state
        """
        import json
        state_str = json.dumps(state, sort_keys=True)
        return hash(state_str)