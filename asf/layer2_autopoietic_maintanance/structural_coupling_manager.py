import numpy as np
import copy
from typing import List, Dict, Tuple, Optional
# Assuming you have the AutocatalyticNetwork and OperationalClosure classes from previous responses
# Make sure those are defined in your environment.
from asf.layer2_autopoietic_maintanance.operational_closure import OperationalClosure, Relationship
from asf.layer2_autopoietic_maintanance.autocatalytic_network import AutocatalyticNetwork
from asf.layer2_autopoietic_maintanance.symbol import SymbolElement #For typing

class StructuralCouplingManager:
    def __init__(self, system: AutocatalyticNetwork, closure_manager: OperationalClosure,
                 initial_environment_state: np.ndarray, learning_rate: float = 0.1,
                 min_closure_threshold: float = 0.6, closure_adjustment_rate: float = 0.01):
        """
        Manages the structural coupling between a system (AutocatalyticNetwork) and its environment.

        Args:
            system: The AutocatalyticNetwork representing the system.
            closure_manager: An OperationalClosure instance to manage the system's internal organization.
            initial_environment_state: The initial state of the environment.
            learning_rate: The learning rate for adapting to the environment.
            min_closure_threshold: Minimum operational closure for the system.
            closure_adjustment_rate: How much to adjust min_closure based on performance.
        """
        self.system = system
        self.closure_manager = closure_manager
        self.environment_state = initial_environment_state
        self.learning_rate = learning_rate
        self.min_closure_threshold = min_closure_threshold
        self.closure_adjustment_rate = closure_adjustment_rate
        self.interaction_history = []

    def _environment_challenge(self) -> np.ndarray:
        """
        Simulates an environmental challenge/perturbation.

        Returns:
            A numpy array representing the environmental perturbation.
        """
        # Simple example: Randomly perturb some dimensions of the environment state
        perturbation = np.random.randn(*self.environment_state.shape) * 0.2  # 20% random noise
        return self.environment_state + perturbation

    def _evaluate_response(self, initial_system_state: np.ndarray, final_system_state: np.ndarray) -> float:
        """
        Evaluates the system's response to an environmental challenge.

        Args:
            initial_system_state: System state *before* the interaction.
            final_system_state: System state *after* the interaction.

        Returns:
            A performance score (higher is better).
        """
        # Simple metric: How much did the system's state change? (Less change = better)
        state_change = np.sum(np.abs(final_system_state - initial_system_state))

        #Normalize by number of nodes (so it's independent of system size)
        normalized_change = state_change / self.system.num_nodes

        # Invert so higher is better (less change = higher score).  Cap at 1.0
        performance = max(0.0, min(1.0, 1.0 - normalized_change))
        return performance


    def run_interaction(self, steps: int = 10) -> Tuple[float, float]:
        """
        Runs a single interaction between the system and the environment.

        Args:
            steps: Number of simulation steps to run within the interaction.

        Returns:
          A tuple of:
            The system's performance score on this interaction
            The system's operational closure *after* the interaction.
        """
        # 1. Present a challenge from the environment
        challenge = self._environment_challenge()

        # 2. Record initial system state
        initial_system_state = np.copy(self.system.current_state)

        # 3. Run the system for a few steps with the challenge as input
        final_state, _ = self.system.run_to_stability(max_steps=steps, external_input=challenge)

        # 4. Evaluate the system's response
        performance = self._evaluate_response(initial_system_state, final_state)

        # 5. Calculate operational closure *after* the interaction
        # Convert the numeric network state to a set of "active" elements
        # for the OperationalClosure manager.  This is a KEY step in linking
        # the network dynamics to the symbolic level.

        active_elements = {}
        for i in range(self.system.num_nodes):
          if self.system.current_state[i] > 0: #If active
            element_id = str(i) #Simple ID for now.
            symbol = SymbolElement(element_id)
            symbol.add_potential(f"{element_id}_potential", {}) #Add a potential
            active_elements[element_id] = symbol #Add to dictionary.

        #Now, add relationships from the network.
        for i in range(self.system.num_nodes):
          for j in range(self.system.num_nodes):
            if self.system.connectivity_matrix[i,j] > 0:
              #Add a support relationship.
              if str(i) in active_elements and str(j) in active_elements: #Only add relationships for active elements.
                active_elements[str(i)].potentials[f"{str(i)}_potential"].add_association(f"{str(j)}:{str(j)}_potential")
                self.closure_manager.add_internal_relation(f"{str(i)}:{str(i)}_potential", f"{str(j)}:{str(j)}_potential", "supports", self.system.connectivity_matrix[i,j])
            elif self.system.connectivity_matrix[i,j] < 0:
              if str(i) in active_elements and str(j) in active_elements:
                active_elements[str(i)].potentials[f"{str(i)}_potential"].add_association(f"{str(j)}:{str(j)}_potential")
                self.closure_manager.add_internal_relation(f"{str(i)}:{str(i)}_potential", f"{str(j)}:{str(j)}_potential", "contradicts", -self.system.connectivity_matrix[i,j])

        current_closure = self.closure_manager.calculate_closure(list(active_elements.keys()))

        # 6.  Rudimentary Autopoiesis:  Adjust network structure if closure is too low.
        if current_closure < self.min_closure_threshold:
            #For now:  Suggest the *strongest* possible connections between existing nodes
            suggested_relations = self.closure_manager.maintain_closure(active_elements, MockNonlinearityTracker(), min_closure = self.min_closure_threshold) #Pass in a dummy.
            for source, target, rel_type in suggested_relations:
                # Extract node IDs from the potential IDs.
                source_node = int(source.split(":")[0])
                target_node = int(target.split(":")[0])

                # Apply a simple rule.
                if rel_type == "supports":
                    self.system.connectivity_matrix[source_node, target_node] = 1.0
                elif rel_type == "contradicts":
                    self.system.connectivity_matrix[source_node, target_node] = -1.0
                #Ignore other types for now.
        # 7. Adjust the closure threshold dynamically
        self.adjust_closure_threshold(performance)

        # 8. Record the interaction for later analysis
        self.interaction_history.append({
            "timestamp": time.time(),
            "environment_state": self.environment_state.copy().tolist(),
            "initial_system_state": initial_system_state.tolist(),
            "final_system_state": final_state.tolist(),
            "performance": performance,
            "closure": current_closure,
        })
        return performance, current_closure

    def adjust_closure_threshold(self, performance_metric: float):
        """
        Adjusts the `min_closure_threshold` based on performance.
        """
        if performance_metric > 0.8:
            self.min_closure_threshold += self.closure_adjustment_rate
        elif performance_metric < 0.6:
            self.min_closure_threshold -= self.closure_adjustment_rate
        self.min_closure_threshold = max(0.1, min(0.95, self.min_closure_threshold))
#Dummy Nonlinearity Tracker
class MockNonlinearityTracker:
  def __init__(self):
    self.potential_nonlinearity = {}

# --- Example Usage ---
# 1. Create an AutocatalyticNetwork (the system)
connectivity_matrix = np.array([
    [0, 0.8, -0.5],
    [0, 0, 0.9],
    [0.6, 0, 0]
])
system = AutocatalyticNetwork(connectivity_matrix)

# 2. Create an OperationalClosure manager
closure_manager = OperationalClosure()

# 3. Create a StructuralCouplingManager
initial_environment_state = np.array([0.5, 0.2, -0.3])  # Example environment
manager = StructuralCouplingManager(system, closure_manager, initial_environment_state)

# 4. Run several interactions
num_interactions = 20
for i in range(num_interactions):
    performance, closure = manager.run_interaction()
    print(f"Interaction {i+1}: Performance = {performance:.2f}, Closure = {closure:.2f}, Min Closure = {manager.min_closure_threshold:.2f}")

# 5. (Optional) Inspect the interaction history
# print("\nInteraction History:", manager.interaction_history)

# Show final network
print(system.connectivity_matrix)