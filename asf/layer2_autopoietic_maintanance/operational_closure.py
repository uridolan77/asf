import time
import scipy.sparse as sp
from typing import Dict, List, Tuple, Set, Optional

from asf.layer2_autopoietic_maintanance.enums import NonlinearityOrder
from asf.layer2_autopoietic_maintanance.symbol import SymbolElement


class Relationship:
    """Represents a relationship between two elements."""

    def __init__(self, source_id: str, target_id: str, rel_type: str, strength: float = 1.0, timestamp: Optional[float] = None):
        self.source_id = source_id
        self.target_id = target_id
        self.type = rel_type  # "supports", "contradicts", "part_of", etc.
        self.strength = strength  # Strength of the relationship (0.0 to 1.0)
        self.timestamp = timestamp if timestamp is not None else time.time()

    def __repr__(self):
        return f"Relationship(source={self.source_id}, target={self.target_id}, type={self.type}, strength={self.strength}, time={self.timestamp})"

class OperationalClosure:
    """
    Implements mechanisms for maintaining system coherence through operational closure,
    with richer relationship representation and dynamic threshold adjustment.
    """

    def __init__(self, initial_min_closure=0.7, closure_adjustment_rate=0.01):
        self.boundary_elements: Set[str] = set()
        # Use a dictionary to store relationships
        self._relationships: Dict[str, Relationship] = {} # Key is a unique relationship ID
        self._element_indices: Dict[str, int] = {}
        self._index_elements: Dict[int, str] = {}
        self.closure_metrics: Dict[str, float] = {} #Not used at present, could be incorporated later
        self._need_rebuild: bool = True  # Flag to indicate matrix rebuild
        self.integrity_history = []
        self.last_integrity_check = 0

        # Dynamic closure threshold
        self.min_closure = initial_min_closure
        self.closure_adjustment_rate = closure_adjustment_rate
        self._relation_matrix: Optional[sp.csr_matrix] = None

    def add_boundary_element(self, element_id: str) -> None:
      """Adds a boundary element."""
      self.boundary_elements.add(element_id)
      if element_id not in self._element_indices:
          idx = len(self._element_indices)
          self._element_indices[element_id] = idx
          self._index_elements[idx] = element_id
          self._need_rebuild = True

    def add_internal_relation(self, source_id: str, target_id: str, rel_type: str, strength: float = 1.0) -> str:
        """Adds or updates an internal relationship."""
        for element_id in (source_id, target_id):
            if element_id not in self._element_indices:
                idx = len(self._element_indices)
                self._element_indices[element_id] = idx
                self._index_elements[idx] = element_id

        # Create a unique ID for the relationship
        relation_id = f"{source_id}->{target_id}:{rel_type}"

        # Add or update the relationship
        self._relationships[relation_id] = Relationship(source_id, target_id, rel_type, strength)
        self._need_rebuild = True
        return relation_id

    def remove_relation(self, relation_id:str) -> None:
      """Removes a relationship"""
      if relation_id in self._relationships:
        del self._relationships[relation_id]
        self._need_rebuild = True

    def _rebuild_matrix(self) -> None:
        """Rebuilds the sparse relation matrix from the relationships dictionary."""
        if not self._need_rebuild:
            return

        n_elements = len(self._element_indices)
        if n_elements == 0:
            self._relation_matrix = sp.csr_matrix((0, 0))
            return

        rows, cols, data = [], [], []
        for rel in self._relationships.values():
            if rel.source_id in self._element_indices and rel.target_id in self._element_indices:
                source_idx = self._element_indices[rel.source_id]
                target_idx = self._element_indices[rel.target_id]
                rows.append(source_idx)
                cols.append(target_idx)
                data.append(rel.strength)  # Use relationship strength

        self._relation_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_elements, n_elements))
        self._need_rebuild = False

    def calculate_closure(self, element_ids: List[str]) -> float:
        """
        Calculates the operational closure of a set of elements.
        """
        if not element_ids:
            return 0.0

        #Ensure the matrix is up to date.
        self._rebuild_matrix()

        # Count total relations involving the specified elements (both directions)
        total_relations = 0
        internal_relations = 0

        for rel in self._relationships.values():
          if rel.source_id in element_ids:
            total_relations +=1
            if rel.target_id in element_ids:
              internal_relations += 1
          elif rel.target_id in element_ids:
            total_relations += 1 #Don't double count if both are in element_ids

        if total_relations == 0:
            return 0.0

        # Record system integrity (periodically)
        current_time = time.time()
        if current_time - self.last_integrity_check > 60:  # Check at most once per minute
            closure_score = internal_relations / total_relations
            self.integrity_history.append({
                'timestamp': current_time,
                'closure_score': closure_score,
                'total_relations': total_relations,
                'internal_relations': internal_relations,
                'element_count': len(element_ids)
            })
            self.last_integrity_check = current_time

        return internal_relations / total_relations


    def maintain_closure(self, elements: Dict[str, SymbolElement],
                            nonlinearity_tracker,
                            min_closure: Optional[float] = None) -> List[Tuple[str, str, str]]:
        """
        Maintains operational closure by suggesting new internal relations.
        Prioritizes simpler relationships and considers relationship types.

        Args:
          elements:  A dictionary of SymbolElements.
          nonlinearity_tracker:  A NonlinearityOrderTracker instance.
          min_closure: The minimum acceptable closure.  Uses the instance's dynamic threshold if None.

        Returns:
          A list of suggested relations:  [(source_id, target_id, rel_type)].
        """
        if min_closure is None:
            min_closure = self.min_closure

        current_closure = self.calculate_closure(list(elements.keys()))
        if current_closure >= min_closure:
            return []

        # Extract current relations *including type*
        current_relations = set()
        for symbol_id, symbol in elements.items():
            for potential_id, potential in symbol.potentials.items():
                source_key = f"{symbol_id}:{potential_id}"
                for assoc_id in potential._associations:
                    # Find the relationship type using the stored relationships
                    rel_id = f"{source_key}->{assoc_id}"
                    if rel_id in self._relationships:
                        current_relations.add((source_key, assoc_id, self._relationships[rel_id].type))

        # Find candidate relations
        element_ids = set(elements.keys())
        potential_relations = []

        for symbol_id, symbol in elements.items():
            for potential_id, potential in symbol.potentials.items():
                source_key = f"{symbol_id}:{potential_id}"
                for target_id in element_ids:
                    if target_id != symbol_id:
                        for target_pot_id in elements[target_id].potentials:
                            target_key = f"{target_id}:{target_pot_id}"
                            # Iterate through possible relationship types
                            for rel_type in ["supports", "contradicts", "is_part_of", "is_a", "related"]:
                                rel_id = f"{source_key}->{target_key}:{rel_type}"
                                if (source_key, target_key, rel_type) not in current_relations and rel_id not in self._relationships:
                                    #Calculate Non-linearity
                                    nonlinearity = NonlinearityOrder.LINEAR
                                    if source_key in nonlinearity_tracker.potential_nonlinearity:
                                      source_nl = nonlinearity_tracker.potential_nonlinearity[source_key]
                                    else:
                                      source_nl = NonlinearityOrder.LINEAR

                                    if target_key in nonlinearity_tracker.potential_nonlinearity:
                                        target_nl = nonlinearity_tracker.potential_nonlinearity[target_key]
                                    else:
                                        target_nl = NonlinearityOrder.LINEAR
                                    nonlinearity = NonlinearityOrder(min(NonlinearityOrder.COMPOSITIONAL.value,max(source_nl.value, target_nl.value) + 1))

                                    potential_relations.append((source_key, target_key, rel_type, nonlinearity))

        # Sort potential relations by nonlinearity and type preference
        potential_relations.sort(key=lambda x: (x[3].value, self._relation_type_preference(x[2])))

         # Select top relations to suggest
        needed_relations = int((min_closure - current_closure) * len(current_relations) * 1.5) + 1
        suggested_relations = [(src, tgt, rel_type) for src, tgt, rel_type, _ in potential_relations[:needed_relations]]
        return suggested_relations

    def _relation_type_preference(self, rel_type: str) -> int:
      """Defines a preference order for suggesting relation types."""
      preference_order = {
          "supports": 1,
          "is_a": 2,
          "is_part_of": 3,
          "related": 4,
          "contradicts": 5,  # Generally avoid suggesting contradictions unless necessary
      }
      return preference_order.get(rel_type, 6)  # Default to lowest priority

    def adjust_closure_threshold(self, performance_metric: float):
        """
        Adjusts the `min_closure` threshold based on a performance metric.

        Args:
            performance_metric (float): A metric indicating system performance (higher is better).
        """
        # Simple adjustment: increase threshold if performance is good, decrease if bad.
        if performance_metric > 0.8:  # Example threshold
            self.min_closure += self.closure_adjustment_rate
        elif performance_metric < 0.6:  # Example threshold
            self.min_closure -= self.closure_adjustment_rate

        # Keep the threshold within reasonable bounds
        self.min_closure = max(0.1, min(0.95, self.min_closure))

    def get_relationships(self):
      """Returns a copy of all relationships"""
      return self._relationships.copy()

    def get_relation_matrix(self):
      """Returns the relation matrix"""
      self._rebuild_matrix() #Ensure it is up to date.
      return self._relation_matrix

    def get_element_indices(self):
      """Return the element->index mapping."""
      return self._element_indices.copy()
    def get_index_elements(self):
      """Returns index-> element mapping"""
      return self._index_elements.copy()

# --- Example Usage ---
# Create some dummy SymbolElements (replace with your actual symbols)
symbols = {
    "A": SymbolElement("A"),
    "B": SymbolElement("B"),
    "C": SymbolElement("C"),
    "D": SymbolElement("D"),
}

# Add some potentials (again, replace with your actual potentials and associations)
symbols["A"].add_potential("A_potential_1", {})
symbols["B"].add_potential("B_potential_1", {})
symbols["C"].add_potential("C_potential_1", {})
symbols["A"].potentials["A_potential_1"].add_association("B:B_potential_1")
symbols["C"].potentials["C_potential_1"].add_association("D:D_potential_1") # Create a relationship, but symbol D doesn't have any potentials.
# Create an OperationalClosure instance
closure_manager = OperationalClosure()

# Add some boundary elements
closure_manager.add_boundary_element("A")
closure_manager.add_boundary_element("B")
closure_manager.add_boundary_element("C")

# Add some initial internal relations
closure_manager.add_internal_relation("A:A_potential_1", "B:B_potential_1", "supports", 0.9)


# Calculate initial closure
initial_closure = closure_manager.calculate_closure(list(symbols.keys()))
print(f"Initial Closure: {initial_closure}")

# Maintain closure (suggest new relations)

#Dummy Nonlinearity Tracker
class MockNonlinearityTracker:
  def __init__(self):
    self.potential_nonlinearity = {}
suggested_relations = closure_manager.maintain_closure(symbols, MockNonlinearityTracker())
print(f"Suggested Relations: {suggested_relations}")

# Add the suggested relations (in a real system, this would involve more complex logic)
for src, tgt, rel_type in suggested_relations:
    closure_manager.add_internal_relation(src, tgt, rel_type, 0.7)  # Example strength


# Calculate closure again
new_closure = closure_manager.calculate_closure(list(symbols.keys()))
print(f"New Closure: {new_closure}")

#Adjust the closure theshold
closure_manager.adjust_closure_threshold(0.9) #good performance.
print(f"Adjusted min closure: {closure_manager.min_closure}")

# --- Show the Relationships and Matrix ---
print("Relationships:", closure_manager.get_relationships())
print("Relation Matrix:\n", closure_manager.get_relation_matrix().todense()) # Convert to dense for display
print("Element Indices", closure_manager.get_element_indices())

# Example of removing a relationship.
closure_manager.remove_relation("A:A_potential_1->B:B_potential_1:supports")
print("Relation Matrix after removal:\n", closure_manager.get_relation_matrix().todense()) # Convert to dense for display