import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Tuple, Optional

# Assuming AutocatalyticNetwork, OperationalClosure, and StructuralCouplingManager from previous responses
# are defined and available.  Import them appropriately.

from asf.layer2_autopoietic_maintanance.structural_coupling_manager import StructuralCouplingManager, MockNonlinearityTracker
from asf.layer2_autopoietic_maintanance.autocatalytic_network import AutocatalyticNetwork
from asf.layer2_autopoietic_maintanance.operational_closure import OperationalClosure, Relationship
from asf.layer2_autopoietic_maintanance.symbol import SymbolElement


class EmbodiedAgent:
    """
    A simple embodied agent with sensors, actuators, and internal state (represented by an autocatalytic network).
    Interacts with an environment and attempts to maintain its internal state (operational closure).
    """

    def __init__(self,
                 initial_position: np.ndarray,
                 target_position: np.ndarray,
                 network: AutocatalyticNetwork,
                 closure_manager: OperationalClosure,
                 coupling_manager: StructuralCouplingManager,
                 sensor_range: float = 2.0,
                 movement_speed: float = 0.1):
        """
        Initializes the EmbodiedAgent.

        Args:
            initial_position: The agent's starting position (numpy array).
            target_position: The target position the agent should move towards (numpy array).
            network: The AutocatalyticNetwork representing the agent's internal state.
            closure_manager:  The OperationalClosure manager.
            coupling_manager: The StructuralCouplingManager.
            sensor_range: The range of the agent's sensors.
            movement_speed: The agent's movement speed.
        """
        self.position = initial_position
        self.target_position = target_position
        self.network = network
        self.closure_manager = closure_manager
        self.coupling_manager = coupling_manager
        self.sensor_range = sensor_range
        self.movement_speed = movement_speed
        self.history = []  # Record agent's trajectory

    def _sense(self) -> np.ndarray:
        """
        Simulates the agent's sensors.

        Returns:
            A numpy array representing the sensor readings (environmental input to the network).
        """
        # Simple sensor:  Measures the *direction* to the target, normalized.
        distance = self.target_position - self.position
        distance_magnitude = np.linalg.norm(distance)

        if distance_magnitude <= self.sensor_range:
            direction = distance / distance_magnitude  # Normalize
            return direction
        else:
            return np.zeros_like(self.position)  # No signal if target is out of range

    def _act(self, action_vector: np.ndarray):
        """
        Simulates the agent's actuators based on the network's output.

        Args:
            action_vector:  The output of the network (numpy array).
        """

        #Simple Action Mapping:  The action vector directly controls movement.
        movement = action_vector * self.movement_speed
        self.position += movement

        # Keep position within bounds (optional, for the simulation)
        self.position = np.clip(self.position, -5, 5)

    def step(self):
        """
        Performs a single step of the agent's interaction loop.
        """
        # 1. Sense the environment
        sensor_input = self._sense()

        # 2. Run the network with the sensor input
        _, _ = self.network.run_to_stability(external_input=sensor_input)

        # 3.  Get network output and act
        network_output = self.network.current_state
        self._act(network_output) #Use the network state *directly* as actions

        # 4. Structural coupling and autopoietic maintenance
        performance, closure = self.coupling_manager.run_interaction()  # Use default number of steps.
        #print(f"Performance = {performance}, and closure = {closure}")

        # 5. Record history
        self.history.append(self.position.copy())

    def run_simulation(self, num_steps: int):
        """
        Runs the simulation for a specified number of steps.

        Args:
            num_steps: The number of steps to run the simulation.
        """
        for _ in range(num_steps):
            self.step()

    def visualize(self):
      """
      Visualizes the agent's trajectory using Matplotlib.
      """
      if not self.history:
        print("No history to visualize. Run the simulation first.")
        return

      history_array = np.array(self.history)
      fig, ax = plt.subplots()
      ax.set_xlim(-5, 5)
      ax.set_ylim(-5, 5)
      ax.set_aspect('equal')
      ax.grid(True)

      # Plot target
      ax.plot(self.target_position[0], self.target_position[1], 'go', markersize=10, label='Target')

      # Plot agent's path
      line, = ax.plot([], [], 'b-', label='Agent Path')

      # Plot agent's current position
      point, = ax.plot([], [], 'ro', markersize=8, label='Agent')

      def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

      def update(frame):
        line.set_data(history_array[:frame, 0], history_array[:frame, 1])
        point.set_data(history_array[frame, 0], history_array[frame, 1])
        return line, point

      ani = FuncAnimation(fig, update, frames=len(self.history), init_func=init, blit=True, repeat=False)
      plt.legend()
      plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Define the network (simplified for this example)
    connectivity_matrix = np.array([
        [0, 0.6, -0.3],
        [-0.4, 0, 0.7],
        [0.5, -0.2, 0]
    ])
    network = AutocatalyticNetwork(connectivity_matrix)

    # 2. Create OperationalClosure and StructuralCouplingManager
    closure_manager = OperationalClosure()
    initial_environment_state = np.array([0.0, 0.0, 0.0])  # Initial environment state
    coupling_manager = StructuralCouplingManager(network, closure_manager, initial_environment_state)

    # 3. Create the EmbodiedAgent
    initial_position = np.array([-3.0, -3.0])
    target_position = np.array([2.0, 2.0])
    agent = EmbodiedAgent(initial_position, target_position, network, closure_manager, coupling_manager)

    # 4. Run the simulation
    agent.run_simulation(100)

    # 5. Visualize the results
    agent.visualize()