"""
Comprehensive Python Agent Framework Implementation
Sections 4-6: Base Agent, Memory System, and Decision Engines
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Tuple
import uuid
import time
import logging


# Section 4: Base Agent Implementation
class Agent(ABC):
    """Base Agent class that defines the core functionality of an agent.
    
    This abstract class provides the foundation for all agent implementations,
    including methods for perception, action execution, memory operations,
    decision making, and lifecycle management.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a new Agent.
        
        Args:
            name: Optional name for the agent. If not provided, a UUID will be generated.
        """
        self.agent_id = str(uuid.uuid4())
        self.name = name if name else f"Agent-{self.agent_id[:8]}"
        self.created_at = time.time()
        self.last_action_time = None
        self.state = {}
        self.memory = {}
        self.sensors = {}
        self.actions = {}
        self.logger = self._setup_logger()
        
        self.logger.info(f"Agent {self.name} initialized with ID {self.agent_id}")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for this agent.
        
        Returns:
            Configured logger instance for the agent
        """
        logger = logging.getLogger(f"agent.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def register_sensor(self, name: str, sensor_func: Callable[["Agent"], Any]) -> None:
        """
        Register a sensor function that can perceive the environment.
        
        Args:
            name: Name of the sensor
            sensor_func: Function that takes the agent and returns sensor data
        """
        self.sensors[name] = sensor_func
        self.logger.debug(f"Registered sensor: {name}")
    
    def register_action(self, name: str, action_func: Callable[["Agent", ...], Any]) -> None:
        """
        Register an action function that the agent can perform.
        
        Args:
            name: Name of the action
            action_func: Function that takes the agent and action parameters
        """
        self.actions[name] = action_func
        self.logger.debug(f"Registered action: {name}")
    
    def perceive(self) -> Dict[str, Any]:
        """
        Use all registered sensors to perceive the environment.
        
        Returns:
            Dictionary of sensor name to sensor data
        """
        perceptions = {}
        for sensor_name, sensor_func in self.sensors.items():
            try:
                perceptions[sensor_name] = sensor_func(self)
                self.logger.debug(f"Sensor {sensor_name} returned data")
            except Exception as e:
                self.logger.error(f"Error with sensor {sensor_name}: {str(e)}")
        
        # Update state with new perceptions
        self.state.update({"last_perception": perceptions, "last_perception_time": time.time()})
        return perceptions
    
    def act(self, action_name: str, **kwargs) -> Any:
        """
        Perform a registered action.
        
        Args:
            action_name: Name of the action to perform
            **kwargs: Parameters for the action
            
        Returns:
            Result of the action
        """
        if action_name not in self.actions:
            self.logger.error(f"Action {action_name} not registered")
            return None
        
        try:
            self.logger.info(f"Performing action: {action_name}")
            result = self.actions[action_name](self, **kwargs)
            self.last_action_time = time.time()
            self.state.update({"last_action": action_name, "last_action_time": self.last_action_time})
            return result
        except Exception as e:
            self.logger.error(f"Error performing action {action_name}: {str(e)}")
            return None
    
    def remember(self, key: str, value: Any) -> None:
        """Store information in memory.
        
        Args:
            key: Key to associate with the value
            value: Data to store in memory
        """
        self.memory[key] = value
        self.logger.debug(f"Remembered: {key}")
    
    def recall(self, key: str) -> Any:
        """Retrieve information from memory.
        
        Args:
            key: Key of the information to retrieve
            
        Returns:
            The stored value or None if not found
        """
        return self.memory.get(key)
    
    @abstractmethod
    def decide(self) -> str:
        """
        Make a decision about what action to take.
        
        This method must be implemented by subclasses.
        
        Returns:
            Name of the action to take
        """
        pass
    
    def run_once(self) -> Any:
        """Run a single sense-decide-act cycle.
        
        Returns:
            Result of the executed action, or None if no action was taken
        """
        perceptions = self.perceive()
        action_name = self.decide()
        if action_name:
            return self.act(action_name)
        return None
    
    def run(self, iterations: Optional[int] = None, max_time: Optional[float] = None) -> None:
        """
        Run the agent for a specified number of iterations or time.
        
        Args:
            iterations: Maximum number of iterations to run (None for unlimited)
            max_time: Maximum time to run in seconds (None for unlimited)
        """
        self.logger.info(f"Starting agent {self.name}")
        
        start_time = time.time()
        iteration = 0
        
        while True:
            # Check termination conditions
            if iterations is not None and iteration >= iterations:
                self.logger.info(f"Agent stopped after {iteration} iterations")
                break
            
            if max_time is not None and (time.time() - start_time) > max_time:
                self.logger.info(f"Agent stopped after {time.time() - start_time:.2f} seconds")
                break
            
            self.run_once()
            iteration += 1
