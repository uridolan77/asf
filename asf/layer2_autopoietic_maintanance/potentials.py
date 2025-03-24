# potentials.py

class SymbolicPotential:
    """
    Base class for symbolic potentials.
    Represents a symbolic entity with potential for activation in predictive processes.
    """

    def __init__(self, id: str, strength: float = 1.0):
        """
        Initialize a symbolic potential.

        Args:
            id (str): Unique identifier for the symbolic potential.
            strength (float): Strength of the potential, default is 1.0.
        """
        self.id = id
        self.strength = strength
        self.activation_history = []  # Track activation events over time

    def activate(self, context: dict) -> float:
        """
        Activate the symbolic potential within a given context.

        Args:
            context (dict): Contextual information influencing activation.

        Returns:
            float: Activation value based on context and strength.
        """
        activation_value = self.strength * self._evaluate_context(context)
        self.activation_history.append((context, activation_value))
        return activation_value

    def _evaluate_context(self, context: dict) -> float:
        """
        Evaluate the influence of the context on activation.

        Args:
            context (dict): Contextual information.

        Returns:
            float: Context evaluation score (default implementation returns 1.0).
        """
        # Placeholder for custom context evaluation logic
        return 1.0

    def get_activation_history(self) -> list:
        """
        Retrieve the history of activations.

        Returns:
            list: List of tuples containing context and activation values.
        """
        return self.activation_history
