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
        Maintains operational closure by suggesting new internal relations.
        Prioritizes simpler relationships and considers relationship types.

        Args:
          elements:  A dictionary of SymbolElements.
          nonlinearity_tracker:  A NonlinearityOrderTracker instance.
          min_closure: The minimum acceptable closure.  Uses the instance's dynamic threshold if None.

        Returns:
          A list of suggested relations:  [(source_id, target_id, rel_type)].
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
        if performance_metric > 0.8:  # Example threshold
            self.min_closure += self.closure_adjustment_rate
        elif performance_metric < 0.6:  # Example threshold
            self.min_closure -= self.closure_adjustment_rate

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

symbols = {
    "A": SymbolElement("A"),
    "B": SymbolElement("B"),
    "C": SymbolElement("C"),
    "D": SymbolElement("D"),
}

symbols["A"].add_potential("A_potential_1", {})
symbols["B"].add_potential("B_potential_1", {})
symbols["C"].add_potential("C_potential_1", {})
symbols["A"].potentials["A_potential_1"].add_association("B:B_potential_1")
symbols["C"].potentials["C_potential_1"].add_association("D:D_potential_1") # Create a relationship, but symbol D doesn't have any potentials.
closure_manager = OperationalClosure()

closure_manager.add_boundary_element("A")
closure_manager.add_boundary_element("B")
closure_manager.add_boundary_element("C")

closure_manager.add_internal_relation("A:A_potential_1", "B:B_potential_1", "supports", 0.9)


initial_closure = closure_manager.calculate_closure(list(symbols.keys()))
print(f"Initial Closure: {initial_closure}")


class MockNonlinearityTracker:
  def __init__(self):
    self.potential_nonlinearity = {}
suggested_relations = closure_manager.maintain_closure(symbols, MockNonlinearityTracker())
print(f"Suggested Relations: {suggested_relations}")

for src, tgt, rel_type in suggested_relations:
    closure_manager.add_internal_relation(src, tgt, rel_type, 0.7)  # Example strength


new_closure = closure_manager.calculate_closure(list(symbols.keys()))
print(f"New Closure: {new_closure}")

closure_manager.adjust_closure_threshold(0.9) #good performance.
print(f"Adjusted min closure: {closure_manager.min_closure}")

print("Relationships:", closure_manager.get_relationships())
print("Relation Matrix:\n", closure_manager.get_relation_matrix().todense()) # Convert to dense for display
print("Element Indices", closure_manager.get_element_indices())

closure_manager.remove_relation("A:A_potential_1->B:B_potential_1:supports")
print("Relation Matrix after removal:\n", closure_manager.get_relation_matrix().todense()) # Convert to dense for display