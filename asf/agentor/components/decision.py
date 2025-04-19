class RuleBasedDecisionEngine:
    """Decision engine based on predefined rules."""
    
    def __init__(self):
        self.rules = []
    
    def add_rule(self, condition_func, action_name, priority=0):
        """
        Add a rule to the decision engine.
        
        Args:
            condition_func: Function that takes the agent and returns True if the rule applies
            action_name: Name of the action to take if the rule applies
            priority: Priority of the rule (higher means more important)
        """
        self.rules.append({
            'condition': condition_func,
            'action': action_name,
            'priority': priority
        })
        
        # Sort rules by priority (descending)
        self.rules.sort(key=lambda x: x['priority'], reverse=True)
    
    def decide(self, agent):
        """
        Make a decision based on the rules.
        
        Args:
            agent: Agent to make a decision for
            
        Returns:
            Name of the action to take, or None if no rules apply
        """
        for rule in self.rules:
            if rule['condition'](agent):
                return rule['action']
        return None


class UtilityBasedDecisionEngine:
    """Decision engine based on utility functions."""
    
    def __init__(self):
        self.actions = {}  # Maps action names to utility functions
    
    def add_action(self, action_name, utility_func):
        """
        Add an action with its utility function.
        
        Args:
            action_name: Name of the action
            utility_func: Function that takes the agent and returns a utility value
        """
        self.actions[action_name] = utility_func
    
    def decide(self, agent):
        """
        Choose the action with the highest utility.
        
        Args:
            agent: Agent to make a decision for
            
        Returns:
            Name of the action with the highest utility, or None if no actions
        """
        if not self.actions:
            return None
            
        # Calculate utility for each action
        utilities = {action: func(agent) for action, func in self.actions.items()}
        
        # Find action with maximum utility
        return max(utilities.items(), key=lambda x: x[1])[0]