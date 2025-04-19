import unittest

class AgentTestCase(unittest.TestCase):
    """Base class for testing agents."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a test environment
        self.environment = self.create_test_environment()
        
        # Create the agent
        self.agent = self.create_agent()
    
    def create_test_environment(self):
        """Create a test environment."""
        # Override this method in subclasses
        return {}
    
    def create_agent(self):
        """Create an agent for testing."""
        # Override this method in subclasses
        return None
    
    def test_perception(self):
        """Test that the agent can perceive the environment."""
        perceptions = self.agent.perceive()
        self.assertIsNotNone(perceptions)
    
    def test_decision(self):
        """Test that the agent can make decisions."""
        decision = self.agent.decide()
        self.assertIsNotNone(decision)
    
    def test_action(self):
        """Test that the agent can take actions."""
        # First get a valid action
        action = self.agent.decide()
        if action:
            result = self.agent.act(action)
            self.assertIsNotNone(result)
    
    def test_run_once(self):
        """Test that the agent can run a complete cycle."""
        result = self.agent.run_once()
        # The result might be None depending on the action
        # This test just ensures that run_once doesn't crash


# Example usage
class SimpleAgentTest(AgentTestCase):
    """Test a simple agent implementation."""
    
    def create_test_environment(self):
        """Create a simple test environment."""
        return {
            'temperature': 22,
            'light_level': 'bright',
            'objects': ['chair', 'table', 'book']
        }
    
    def create_agent(self):
        """Create a simple test agent."""
        from simple_agent import SimpleAgent  # Import your agent class
        agent = SimpleAgent(name="TestAgent")
        
        # Add sensors
        agent.register_sensor('temperature', lambda a: self.environment['temperature'])
        agent.register_sensor('light', lambda a: self.environment['light_level'])
        
        # Add actions
        agent.register_action('move', lambda a, direction: f"Moving {direction}")
        agent.register_action('pick_up', lambda a, item: f"Picking up {item}")
        
        return agent
    
    def test_specific_behavior(self):
        """Test a specific behavior of the agent."""
        # Change the environment
        self.environment['temperature'] = 30
        
        # Let the agent perceive and decide
        self.agent.perceive()
        action = self.agent.decide()
        
        # Assert expected behavior
        self.assertEqual(action, 'move')  # Assuming it moves when hot