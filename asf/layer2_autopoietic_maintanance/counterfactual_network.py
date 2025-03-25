import numpy as np
import copy
from typing import List, Dict, Optional, Tuple

class AutocatalyticNetwork:
    """
    An autocatalytic network with counterfactual reasoning capabilities.
    """

    def __init__(self, connectivity_matrix: np.ndarray, activation_threshold: float = 0.5):
        """
        Initializes the network.

        Args:
            connectivity_matrix: A square numpy array representing network connections.
                                 connectivity_matrix[i, j] > 0 if node i activates node j.
                                 connectivity_matrix[i, j] < 0 if node i inhibits node j.
                                 connectivity_matrix[i, j] == 0 for no connection.
            activation_threshold: The threshold for node activation.
        """
        self.connectivity_matrix = connectivity_matrix
        self.activation_threshold = activation_threshold
        self.num_nodes = connectivity_matrix.shape[0]
        self.current_state = np.zeros(self.num_nodes)  # Initial state: all nodes inactive

    def step(self, external_input: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simulates one step of network dynamics.

        Args:
            external_input: Optional external input to the nodes.  Must be a numpy array
                            of the same size as the network.

        Returns:
            The new state of the network after the step.
        """
        if external_input is None:
            external_input = np.zeros(self.num_nodes)

        if external_input.shape != (self.num_nodes,):
            raise ValueError("External input must have the same dimension as the number of nodes.")

        net_input = np.dot(self.connectivity_matrix, self.current_state) + external_input
        self.current_state = np.where(net_input > self.activation_threshold, 1, 0) # Apply threshold
        return self.current_state

    def run_to_stability(self, max_steps: int = 100, external_input: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """
        Runs the network until it reaches a stable state or the maximum number of steps is reached.

        Args:
            max_steps: The maximum number of steps to run.
            external_input:  Optional external input (applied at each step).

        Returns:
            A tuple: (final_state, is_stable).  is_stable is True if the network reached
            a stable state, False if it oscillated or didn't converge within max_steps.
        """
        previous_state = np.copy(self.current_state)
        for _ in range(max_steps):
            current_state = self.step(external_input)
            if np.array_equal(current_state, previous_state):
                return current_state, True  # Stable state reached
            previous_state = np.copy(current_state)
        return self.current_state, False  # Did not converge

    def evaluate_counterfactual(self,
                                modified_connections: Dict[Tuple[int, int], float],
                                external_input: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool, Dict]:
        """
        Evaluates a counterfactual scenario by modifying network connections.

        Args:
            modified_connections: A dictionary where keys are (source, target) node tuples
                                 and values are the new connection strengths.
            external_input: Optional external input for the counterfactual scenario.

        Returns:
            A tuple: (counterfactual_state, counterfactual_stable, impact_summary).
                counterfactual_state: The final state of the network in the counterfactual.
                counterfactual_stable:  True if stable, False otherwise.
                impact_summary: A dictionary describing the impact of the changes:
                    "changed_connections": The connections that were modified.
                    "state_diff":  Nodes that changed state compared to the original.
                    "output_change": A measure of the overall change in network output.

        """
        # 1. Create a *copy* of the network
        counterfactual_network = copy.deepcopy(self)

        # 2. Apply the modifications to the *copy*
        for (source, target), new_strength in modified_connections.items():
            if 0 <= source < self.num_nodes and 0 <= target < self.num_nodes:
                counterfactual_network.connectivity_matrix[source, target] = new_strength
            else:
                raise ValueError(f"Invalid connection indices: ({source}, {target})")

        # 3. Run the *modified* network to stability
        initial_state, initial_stable = self.run_to_stability(external_input=external_input) # Run Original to Stability
        counterfactual_state, counterfactual_stable = counterfactual_network.run_to_stability(external_input=external_input)

        # 4. Compare the original and counterfactual states
        state_diff = np.where(counterfactual_state != initial_state)[0].tolist()
        output_change = np.sum(counterfactual_state) - np.sum(initial_state)

        impact_summary = {
            "changed_connections": modified_connections,
            "state_diff": state_diff,
            "output_change": output_change,
            "initial_state": initial_state.tolist(), #Include Initial State
            "initial_stable": initial_stable, #And whether it was stable.
        }

        return counterfactual_state, counterfactual_stable, impact_summary


    def evaluate_counterfactual_extended(self,
                                        modified_connections: Optional[Dict[Tuple[int, int], float]] = None,
                                        modified_thresholds: Optional[Dict[int, float]] = None,
                                        modified_inputs: Optional[Dict[int, float]] = None
                                        ) -> Tuple[np.ndarray, bool, Dict]:
        """
        Extends counterfactual evaluation to allow changes in thresholds and initial states,
        in addition to connections.

        Args:
            modified_connections: Changes to the connectivity matrix (as before).
            modified_thresholds:  A dictionary where keys are node indices and values
                                  are the new activation thresholds.
            modified_inputs: A dictionary for CONSTANT external inputs. Keys are node indices,
                             values are the input magnitudes.  Applied at *every* step.

        Returns:
            Same as evaluate_counterfactual, but impact_summary now includes:
                "changed_thresholds": (if applicable)
                "changed_inputs": (if applicable)
        """

        counterfactual_network = copy.deepcopy(self)

        # Apply connection modifications
        if modified_connections:
            for (source, target), new_strength in modified_connections.items():
                if 0 <= source < self.num_nodes and 0 <= target < self.num_nodes:
                    counterfactual_network.connectivity_matrix[source, target] = new_strength
                else:
                    raise ValueError(f"Invalid connection indices: ({source}, {target})")

        # Apply threshold modifications
        if modified_thresholds:
            for node, new_threshold in modified_thresholds.items():
                if 0 <= node < self.num_nodes:
                    counterfactual_network.activation_threshold = np.copy(counterfactual_network.activation_threshold)  # Ensure we have a modifiable copy
                    if isinstance(counterfactual_network.activation_threshold, float): #If it's a single value
                        counterfactual_network.activation_threshold = np.full(counterfactual_network.num_nodes, counterfactual_network.activation_threshold) #Convert to array.
                    counterfactual_network.activation_threshold[node] = new_threshold #Modify that array
                else:
                    raise ValueError(f"Invalid node index for threshold modification: {node}")

        # Run to stability (with modified inputs if provided)
        if modified_inputs:
          #Convert the dict to a numpy array.
          external_input_array = np.zeros(self.num_nodes)
          for node,inputValue in modified_inputs.items():
            if 0 <= node < self.num_nodes:
              external_input_array[node] = inputValue
            else:
              raise ValueError(f"Invalid node index for input modification: {node}")
        else:
          external_input_array = None

        initial_state, initial_stable = self.run_to_stability()  # No external input for initial state.
        counterfactual_state, counterfactual_stable = counterfactual_network.run_to_stability(external_input=external_input_array)

        # --- Impact Analysis ---
        state_diff = np.where(counterfactual_state != initial_state)[0].tolist()
        output_change = np.sum(counterfactual_state) - np.sum(initial_state)

        impact_summary = {
            "initial_state": initial_state.tolist(),
            "initial_stable": initial_stable,
            "changed_connections": modified_connections if modified_connections else {},
            "changed_thresholds": modified_thresholds if modified_thresholds else {},
            "changed_inputs": modified_inputs if modified_inputs else {},
            "state_diff": state_diff,
            "output_change": output_change,
        }
        return counterfactual_state, counterfactual_stable, impact_summary
    def identify_critical_connections(self, threshold_change: float = 0.1) -> List[Tuple[Tuple[int, int], float]]:
      """
      Identifies critical connections in the network by systematically perturbing
      each connection and observing the effect on the network's output.

      Args:
          threshold_change: The minimum change in network output to consider a
              connection critical.

      Returns:
          A list of tuples, where each tuple contains:
            - (source, target):  The indices of the connection.
            - impact_score:  The magnitude of the change in network output.
      """
      critical_connections = []
      initial_state, _ = self.run_to_stability()
      initial_output = np.sum(initial_state)

      for i in range(self.num_nodes):
        for j in range(self.num_nodes):
          original_strength = self.connectivity_matrix[i,j]

          #Test Increase
          modified_connections_inc = {(i,j): original_strength + 0.1}
          _, _, impact_summary_inc = self.evaluate_counterfactual(modified_connections_inc)
          impact_inc = abs(impact_summary_inc['output_change'])

          #Test Decrease
          modified_connections_dec = {(i,j): original_strength - 0.1}
          _,_, impact_summary_dec = self.evaluate_counterfactual(modified_connections_dec)
          impact_dec = abs(impact_summary_dec['output_change'])

          impact_score = max(impact_inc, impact_dec) #Take larger impact.

          if impact_score > threshold_change:
            critical_connections.append(((i,j), impact_score))
      return critical_connections
    def suggest_interventions(self, desired_state_change: np.ndarray) -> List[Dict]:
      """
      Suggests interventions to move the network towards a desired state change.

      Args:
        desired_state_change: A numpy array representing the desired change in node states.
                              Positive values indicate desired activation, negative indicate
                              desired deactivation, and 0 indicates no preference.

      Returns:
        A list of suggested interventions, each a dictionary with the following keys:
            "type":  The type of intervention ("connection", "threshold", or "input").
            "target":  The target of the intervention (e.g., a (source, target) tuple for connections).
            "strength": The suggested strength of the intervention (positive for activation, negative for inhibition).
            "confidence":  A confidence score (0.0 to 1.0) for the intervention.
      """
      suggestions = []

      #1. Connection Interventions
      for i in range(self.num_nodes):
        for j in range(self.num_nodes):
          if desired_state_change[j] > 0:  # Node j should be activated
            # Suggest increasing activating or reducing inhibiting connections
            if self.connectivity_matrix[i, j] > 0:
                suggestions.append({
                    "type": "connection",
                    "target": (i, j),
                    "strength": 0.1,  # Suggest increasing existing activation
                    "confidence": 0.6 + 0.2 * self.connectivity_matrix[i,j] #Higher confidence if already activating
                })
            else:
                suggestions.append({
                    "type": "connection",
                    "target": (i, j),
                    "strength": 0.1,  # Suggest adding an activating connection
                    "confidence": 0.4,
                })
          elif desired_state_change[j] < 0:  # Node j should be deactivated
            # Suggest decreasing activating or increasing inhibiting connections
            if self.connectivity_matrix[i, j] < 0:
                suggestions.append({
                    "type": "connection",
                    "target": (i, j),
                    "strength": -0.1,  # Suggest increasing existing inhibition
                    "confidence": 0.6 + 0.2 * abs(self.connectivity_matrix[i,j]) #Confidence
                })
            else:
                suggestions.append({
                  "type": "connection",
                  "target": (i,j),
                  "strength": -0.1, #Suggest Adding an inhibiting connection
                  "confidence": 0.4
                })
      # 2. Threshold Interventions
      for i in range(self.num_nodes):
          if desired_state_change[i] > 0:
            suggestions.append({
                "type": "threshold",
                "target": i,
                "strength": -0.1,  # Lower threshold to activate
                "confidence": 0.5,
            })
          elif desired_state_change[i] < 0:
            suggestions.append({
                "type": "threshold",
                "target": i,
                "strength": 0.1,  # Raise threshold to deactivate
                "confidence": 0.5,
            })

      # 3. Input Interventions
      for i in range(self.num_nodes):
          if desired_state_change[i] > 0:
            suggestions.append({
                "type": "input",
                "target": i,
                "strength": 0.2,  # Apply positive input
                "confidence": 0.7,
            })
          elif desired_state_change[i] < 0:
            suggestions.append({
                "type": "input",
                "target": i,
                "strength": -0.2,  # Apply negative input
                "confidence": 0.7,
            })

      return suggestions

# --- Example Usage ---
# A simple network: A -> B -> C, with C inhibiting A
connectivity = np.array([
    [0, 1, -1],  # A -> B, A -| C
    [0, 0, 1],  # B -> C
    [0, 0, 0]   # C has no outgoing connections
])

network = AutocatalyticNetwork(connectivity)

# --- Counterfactual: What if we break the connection from A to B? ---
modified_connections = {(0, 1): 0}  # A no longer activates B
counterfactual_state, counterfactual_stable, impact_summary = network.evaluate_counterfactual(modified_connections)

print("Counterfactual Scenario: Break A -> B")
print("Original State:", impact_summary['initial_state'], "Stable:", impact_summary['initial_stable'])
print("Counterfactual State:", counterfactual_state, "Stable:", counterfactual_stable)
print("Impact Summary:", impact_summary)

# --- Counterfactual: What if we make C activate A instead of inhibiting it? ---
modified_connections = {(0, 2): 1}  # C now *activates* A
counterfactual_state2, counterfactual_stable2, impact_summary2 = network.evaluate_counterfactual(modified_connections)

print("\nCounterfactual Scenario: C activates A")
print("Original State:", impact_summary2['initial_state'], "Stable:", impact_summary2['initial_stable'])
print("Counterfactual State:", counterfactual_state2, "Stable:", counterfactual_stable2)
print("Impact Summary:", impact_summary2)


# --- Extended Counterfactual: Modify threshold of node B and add external input to A ---
modified_thresholds = {1: 0.3}  # Lower threshold for B
modified_inputs = {0: 0.6}     # Constant input to A

counterfactual_state3, counterfactual_stable3, impact_summary3 = network.evaluate_counterfactual_extended(
    modified_thresholds=modified_thresholds, modified_inputs=modified_inputs
)

print("\nCounterfactual Scenario: Lower B threshold, input to A")
print("Original State:", impact_summary3['initial_state'], "Stable:", impact_summary3['initial_stable'])
print("Counterfactual State:", counterfactual_state3, "Stable:", counterfactual_stable3)
print("Impact Summary:", impact_summary3)


# --- Identify Critical Connections ---
critical_connections = network.identify_critical_connections()
print("\nCritical Connections:")
for (source, target), impact in critical_connections:
    print(f"  Connection ({source}, {target}): Impact = {impact:.2f}")

# --- Suggest Interventions ---
# Example: We want to deactivate node 2 (C) and have no preference for others.
desired_change = np.array([0, 0, -1])
interventions = network.suggest_interventions(desired_change)
print("\nSuggested Interventions:")
for intervention in interventions:
    print(f"  Type: {intervention['type']}, Target: {intervention['target']}, Strength: {intervention['strength']:.2f}, Confidence: {intervention['confidence']:.2f}")
