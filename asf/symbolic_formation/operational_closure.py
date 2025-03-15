import time
import scipy.sparse as sp
from typing import Dict, List, Tuple, Set, Optional

from asf.symbolic_formation.enums import NonlinearityOrder
from asf.symbolic_formation.symbol import SymbolElement

class OperationalClosure:
    """
    Implements mechanisms for maintaining system coherence through
    operational closure as per Maturana and Varela.
    Optimized with sparse matrix representation for efficiency.
    """
    def __init__(self):
        self.boundary_elements: Set[str] = set()
        # Use sparse matrix for internal relations
        self._relation_matrix: Optional[sp.csr_matrix] = None
        self._element_indices: Dict[str, int] = {}
        self._index_elements: Dict[int, str] = {}
        self.closure_metrics: Dict[str, float] = {}
        self._need_rebuild: bool = True
        
        # Phase 2 enhancement: system integrity tracking
        self.integrity_history = []
        self.last_integrity_check = 0
    
    def add_boundary_element(self, element_id: str) -> None:
        """Add an element to the system boundary."""
        self.boundary_elements.add(element_id)
        # Ensure element is in index mapping
        if element_id not in self._element_indices:
            idx = len(self._element_indices)
            self._element_indices[element_id] = idx
            self._index_elements[idx] = element_id
            self._need_rebuild = True
    
    def add_internal_relation(self, source_id: str, target_id: str) -> None:
        """Add an internal relation between elements."""
        # Ensure elements are in index mapping
        for element_id in (source_id, target_id):
            if element_id not in self._element_indices:
                idx = len(self._element_indices)
                self._element_indices[element_id] = idx
                self._index_elements[idx] = element_id
                
        self._need_rebuild = True
        
    def _rebuild_matrix(self, relations: List[Tuple[str, str]]) -> None:
        """Rebuild the sparse relation matrix."""
        if not self._need_rebuild and self._relation_matrix is not None:
            return
            
        n_elements = len(self._element_indices)
        if n_elements == 0:
            self._relation_matrix = sp.csr_matrix((0, 0))
            return
            
        # Build sparse matrix
        rows, cols, data = [], [], []
        for source_id, target_id in relations:
            if source_id in self._element_indices and target_id in self._element_indices:
                source_idx = self._element_indices[source_id]
                target_idx = self._element_indices[target_id]
                rows.append(source_idx)
                cols.append(target_idx)
                data.append(1.0)
                
        self._relation_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_elements, n_elements))
            
        self._need_rebuild = False
        
    def calculate_closure(self, elements: Dict[str, SymbolElement]) -> float:
        """
        Calculate degree of operational closure using efficient sparse operations.
        1.0 means perfect closure, 0.0 means completely open.
        """
        if not elements:
            return 0.0
            
        # Extract relations from elements
        relations = []
        for symbol_id, symbol in elements.items():
            # Add symbol to indices if needed
            if symbol_id not in self._element_indices:
                idx = len(self._element_indices)
                self._element_indices[symbol_id] = idx
                self._index_elements[idx] = symbol_id
                
            # Extract relations from potentials
            for potential_id, potential in symbol.potentials.items():
                source_key = f"{symbol_id}:{potential_id}"
                
                # Add relation source to indices
                if source_key not in self._element_indices:
                    idx = len(self._element_indices)
                    self._element_indices[source_key] = idx
                    self._index_elements[idx] = source_key
                    
                # Add relations to associations
                for assoc_id in potential._associations:
                    relations.append((source_key, assoc_id))
                    
        # Rebuild relation matrix
        self._rebuild_matrix(relations)
        
        # Calculate closure using matrix operations
        if self._relation_matrix.shape[0] == 0:
            return 0.0
            
        # Count total relations
        total_relations = self._relation_matrix.count_nonzero()
        if total_relations == 0:
            return 0.0
            
        # Count relations between elements in the system
        element_indices = [self._element_indices[e_id] for e_id in elements
                         if e_id in self._element_indices]
        if not element_indices:
            return 0.0
            
        # Extract submatrix for system elements
        system_matrix = self._relation_matrix[element_indices, :][:, element_indices]
        internal_relations = system_matrix.count_nonzero()
        
        # Record system integrity
        current_time = time.time()
        if current_time - self.last_integrity_check > 60:  # Check at most once per minute
            closure_score = internal_relations / total_relations
            self.integrity_history.append({
                'timestamp': current_time,
                'closure_score': closure_score,
                'total_relations': total_relations,
                'internal_relations': internal_relations,
                'element_count': len(elements)
            })
            self.last_integrity_check = current_time
            
        return internal_relations / total_relations
        
    def maintain_closure(self, elements: Dict[str, SymbolElement],
                      nonlinearity_tracker,
                      min_closure: float = 0.7) -> List[Tuple[str, str]]:
        """
        Maintain operational closure by suggesting new internal relations
        if closure falls below threshold. Prioritizes simpler relationships.
        """
        current_closure = self.calculate_closure(elements)
        if current_closure >= min_closure:
            return []
            
        # Find potential new relations to increase closure
        suggested_relations = []
        
        # Extract current relations
        current_relations = set()
        for symbol_id, symbol in elements.items():
            for potential_id, potential in symbol.potentials.items():
                source_key = f"{symbol_id}:{potential_id}"
                for assoc_id in potential._associations:
                    current_relations.add((source_key, assoc_id))
                    
        # Find candidate relations between system elements
        element_ids = set(elements.keys())
        potential_relations = []
        
        for symbol_id, symbol in elements.items():
            for potential_id, potential in symbol.potentials.items():
                source_key = f"{symbol_id}:{potential_id}"
                
                for target_id in element_ids:
                    if target_id != symbol_id:
                        for target_pot_id in elements[target_id].potentials:
                            target_key = f"{target_id}:{target_pot_id}"
                            
                            # Check if relation already exists
                            if (source_key, target_key) not in current_relations:
                                # Calculate potential relationship nonlinearity
                                nonlinearity = NonlinearityOrder.LINEAR
                                
                                if source_key in nonlinearity_tracker.potential_nonlinearity:
                                    source_nl = nonlinearity_tracker.potential_nonlinearity[source_key]
                                else:
                                    source_nl = NonlinearityOrder.LINEAR
                                    
                                if target_key in nonlinearity_tracker.potential_nonlinearity:
                                    target_nl = nonlinearity_tracker.potential_nonlinearity[target_key]
                                else:
                                    target_nl = NonlinearityOrder.LINEAR
                                
                                # Combine nonlinearities
                                nonlinearity = NonlinearityOrder(
                                    min(NonlinearityOrder.COMPOSITIONAL.value,
                                       max(source_nl.value, target_nl.value) + 1))
                                        
                                # Add as candidate with nonlinearity as score
                                potential_relations.append(
                                    (source_key, target_key, nonlinearity))
                                    
        # Sort potential relations by nonlinearity (simpler first)
        potential_relations.sort(key=lambda x: x[2].value)
        
        # Select top relations to suggest
        needed_relations = int((min_closure - current_closure) * 
                             len(current_relations) * 1.5) + 1
                              
        suggested_relations = [(src, tgt) for src, tgt, _ in 
                              potential_relations[:needed_relations]]
                               
        return suggested_relations
