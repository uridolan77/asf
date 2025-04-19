from typing import Callable, List, Tuple, Any

class Rule:
    """A rule for a rule-based decision engine."""

    def __init__(self, condition: Callable, action: str, priority: int = 1):
        """Initialize a rule.

        Args:
            condition: A function that takes an agent and returns a boolean
            action: The name of the action to take if the condition is met
            priority: The priority of the rule (higher values = higher priority)
        """
        self.condition = condition
        self.action = action
        self.priority = priority

class RuleBasedDecisionEngine:
    """A decision engine that uses rules to make decisions."""

    def __init__(self):
        """Initialize the decision engine."""
        self.rules: List[Rule] = []

    def add_rule(self, condition: Callable, action: str, priority: int = 1):
        """Add a rule to the decision engine.

        Args:
            condition: A function that takes an agent and returns a boolean
            action: The name of the action to take if the condition is met
            priority: The priority of the rule (higher values = higher priority)
        """
        self.rules.append(Rule(condition, action, priority))

    def decide(self, agent) -> str:
        """Make a decision for the agent.

        Args:
            agent: The agent to make a decision for

        Returns:
            The name of the action to take
        """
        # Find all applicable rules
        applicable_rules = []
        for rule in self.rules:
            if rule.condition(agent):
                applicable_rules.append(rule)

        if not applicable_rules:
            raise ValueError("No applicable rules found")

        # Sort by priority (highest first)
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)

        # Return the action of the highest-priority rule
        return applicable_rules[0].action
