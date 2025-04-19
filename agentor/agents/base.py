from typing import Dict, Any, Callable, List, Optional, Tuple
import uuid
import asyncio
from pydantic import BaseModel

from agentor.agents.tools import BaseTool, ToolResult


class AgentInput(BaseModel):
    """Input data for an agent."""
    query: str
    context: Optional[Dict[str, Any]] = None


class AgentOutput(BaseModel):
    """Output data from an agent."""
    response: str
    metadata: Optional[Dict[str, Any]] = None


class Agent:
    """Base class for all agents."""

    def __init__(self, name=None):
        self.name = name or f"Agent-{uuid.uuid4().hex[:8]}"
        self.state = {}
        self.sensors = {}
        self.actions = {}
        self.tools: Dict[str, BaseTool] = {}

    async def preprocess(self, input_data: AgentInput) -> AgentInput:
        """Preprocess the input data before running the agent.

        This is a good place to put IO-bound operations.

        Args:
            input_data: The input data to preprocess

        Returns:
            The preprocessed input data
        """
        return input_data

    async def postprocess(self, output_data: AgentOutput) -> AgentOutput:
        """Postprocess the output data after running the agent.

        This is a good place to put async operations.

        Args:
            output_data: The output data to postprocess

        Returns:
            The postprocessed output data
        """
        return output_data

    def register_sensor(self, name: str, sensor_func: Callable):
        """Register a sensor function.

        Args:
            name: The name of the sensor
            sensor_func: A function that takes the agent as input and returns a sensor reading
        """
        self.sensors[name] = sensor_func

    def register_action(self, name: str, action_func: Callable):
        """Register an action function.

        Args:
            name: The name of the action
            action_func: A function that takes the agent as input and performs an action
        """
        self.actions[name] = action_func

    def register_tool(self, tool: BaseTool):
        """Register a tool.

        Args:
            tool: The tool to register
        """
        self.tools[tool.name] = tool

    async def execute_tools(self, tools_to_run: List[Tuple[BaseTool, Dict[str, Any]]]) -> List[ToolResult]:
        """Execute multiple tools in parallel.

        Args:
            tools_to_run: A list of (tool, params) tuples to run

        Returns:
            A list of tool results
        """
        return await asyncio.gather(
            *(tool.run(**params) for tool, params in tools_to_run)
        )

    def perceive(self):
        """Collect data from all sensors.

        Returns:
            A dictionary of sensor readings
        """
        perception = {}
        for name, sensor in self.sensors.items():
            perception[name] = sensor(self)
        self.state['last_perception'] = perception
        return perception

    def decide(self):
        """Make a decision based on the current state.

        Returns:
            The name of the action to take
        """
        raise NotImplementedError("Subclasses must implement decide()")

    def act(self, action_name):
        """Execute the specified action.

        Args:
            action_name: The name of the action to execute

        Returns:
            The result of the action
        """
        if action_name not in self.actions:
            raise ValueError(f"Unknown action: {action_name}")
        return self.actions[action_name](self)

    def run_once(self):
        """Run one perception-decision-action cycle.

        Returns:
            The result of the action
        """
        self.perceive()
        action = self.decide()
        return self.act(action)

    async def run(self, query: str, context: Dict[str, Any] = None) -> AgentOutput:
        """Run the agent with the given input.

        Args:
            query: The query to process
            context: Additional context for the query

        Returns:
            The agent's response
        """
        input_data = AgentInput(query=query, context=context or {})

        # Preprocess the input
        processed_input = await self.preprocess(input_data)

        # Run the agent's core logic
        self.state['current_query'] = processed_input.query
        self.state['current_context'] = processed_input.context

        result = self.run_once()

        # Create the output
        output_data = AgentOutput(
            response=result,
            metadata={
                "agent_name": self.name,
                "state": self.state
            }
        )

        # Postprocess the output
        processed_output = await self.postprocess(output_data)

        return processed_output
