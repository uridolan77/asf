import asyncio
from agentor.agents.base import Agent, AgentInput, AgentOutput
from agentor.components.decision import RuleBasedDecisionEngine

class SimpleRuleAgent(Agent):
    """A simple rule-based agent."""

    def __init__(self, name=None):
        super().__init__(name)
        self.decision_engine = RuleBasedDecisionEngine()
        self._setup_rules()

    def _setup_rules(self):
        """Set up the rules for this agent."""
        # Rule 1: If it's hot, move to shade
        self.decision_engine.add_rule(
            lambda a: a.state.get('last_perception', {}).get('temperature', 0) > 25,
            'move_to_shade',
            priority=2
        )

        # Rule 2: If it's dark, turn on light
        self.decision_engine.add_rule(
            lambda a: a.state.get('last_perception', {}).get('light') == 'dark',
            'turn_on_light',
            priority=3
        )

        # Rule 3: Default action
        self.decision_engine.add_rule(
            lambda a: True,  # Always applies, but lowest priority
            'wait',
            priority=1
        )

    def decide(self):
        """Make a decision using the rule-based engine."""
        return self.decision_engine.decide(self)

    async def preprocess(self, input_data: AgentInput) -> AgentInput:
        """Example of preprocessing with IO-bound operations."""
        print(f"Preprocessing input: {input_data}")
        # Simulate an IO-bound operation
        await asyncio.sleep(0.1)
        return input_data

    async def postprocess(self, output_data: AgentOutput) -> AgentOutput:
        """Example of postprocessing with async operations."""
        print(f"Postprocessing output: {output_data}")
        # Simulate an async operation
        await asyncio.sleep(0.1)
        return output_data


# Usage example
if __name__ == "__main__":
    # Create the agent
    agent = SimpleRuleAgent(name="HomeAgent")

    # Register sensors
    agent.register_sensor('temperature', lambda a: 28)  # Simulate temperature sensor
    agent.register_sensor('light', lambda a: 'bright')  # Simulate light sensor

    # Register actions
    agent.register_action('move_to_shade', lambda a: "Moving to shade")
    agent.register_action('turn_on_light', lambda a: "Turning on the light")
    agent.register_action('wait', lambda a: "Waiting")

    # Run the agent synchronously
    print("\nRunning synchronously:")
    result = agent.run_once()
    print(f"Agent action result: {result}")

    # Run the agent asynchronously
    print("\nRunning asynchronously:")
    async def run_async():
        result = await agent.run("What should I do if it's hot?", {"time": "day"})
        print(f"Agent async result: {result}")

    asyncio.run(run_async())