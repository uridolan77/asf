import pytest
import asyncio
from agentor.agents.base import Agent, AgentInput, AgentOutput


class TestAgent(Agent):
    """A test agent for unit testing."""
    
    def __init__(self, name=None):
        super().__init__(name)
        self.test_result = "Test result"
    
    def decide(self):
        """Return a fixed action for testing."""
        return "test_action"
    
    async def preprocess(self, input_data: AgentInput) -> AgentInput:
        """Add a test field to the context."""
        input_data.context["test_field"] = "test_value"
        return input_data
    
    async def postprocess(self, output_data: AgentOutput) -> AgentOutput:
        """Add a test field to the metadata."""
        output_data.metadata["test_field"] = "test_value"
        return output_data


@pytest.mark.asyncio
async def test_agent_run():
    """Test the Agent.run method."""
    # Create the agent
    agent = TestAgent(name="TestAgent")
    
    # Register a test action
    agent.register_action("test_action", lambda a: a.test_result)
    
    # Run the agent
    result = await agent.run("Test query", {"initial_context": "value"})
    
    # Check the result
    assert result.response == "Test result"
    assert result.metadata["agent_name"] == "TestAgent"
    assert result.metadata["test_field"] == "test_value"
    assert "state" in result.metadata
    assert result.metadata["state"]["current_query"] == "Test query"
    assert result.metadata["state"]["current_context"]["initial_context"] == "value"
    assert result.metadata["state"]["current_context"]["test_field"] == "test_value"


@pytest.mark.asyncio
async def test_agent_preprocess():
    """Test the Agent.preprocess method."""
    # Create the agent
    agent = TestAgent(name="TestAgent")
    
    # Create an input
    input_data = AgentInput(query="Test query", context={"initial_context": "value"})
    
    # Preprocess the input
    result = await agent.preprocess(input_data)
    
    # Check the result
    assert result.query == "Test query"
    assert result.context["initial_context"] == "value"
    assert result.context["test_field"] == "test_value"


@pytest.mark.asyncio
async def test_agent_postprocess():
    """Test the Agent.postprocess method."""
    # Create the agent
    agent = TestAgent(name="TestAgent")
    
    # Create an output
    output_data = AgentOutput(
        response="Test response",
        metadata={"initial_metadata": "value"}
    )
    
    # Postprocess the output
    result = await agent.postprocess(output_data)
    
    # Check the result
    assert result.response == "Test response"
    assert result.metadata["initial_metadata"] == "value"
    assert result.metadata["test_field"] == "test_value"
