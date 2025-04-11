import asyncio
import time
import logging
from typing import Dict, List, Optional, Set
from collections import defaultdict

from asf.layer4_environmental_coupling.models import EnvironmentalCoupling
from asf.layer4_environmental_coupling.enums import CouplingState

class SparseCouplingRegistry:
    """
    Efficiently manages environmental couplings with sparse representation.
    Optimized for quick lookup by both internal and environmental entity IDs.
    """
    
    def __init__(self, initial_capacity=10000):
        self.couplings: Dict[str, EnvironmentalCoupling] = {}
        self.internal_entity_map: Dict[str, Set[str]] = defaultdict(set)
        self.environmental_entity_map: Dict[str, Set[str]] = defaultdict(set)
        self.lock = asyncio.Lock()
        self.capacity = initial_capacity
        self.logger = logging.getLogger("ASF.Layer4.SparseCouplingRegistry")
        
    async def initialize(self):
        async with self.lock:
            if coupling.id in self.couplings:
                self.logger.warning(f"Coupling {coupling.id} already exists")
                return False
                
            self.couplings[coupling.id] = coupling
            self.internal_entity_map[coupling.internal_entity_id].add(coupling.id)
            self.environmental_entity_map[coupling.environmental_entity_id].add(coupling.id)
            
            self.logger.debug(f"Added coupling {coupling.id} between {coupling.internal_entity_id} and {coupling.environmental_entity_id}")
            return True
    
    async def get_coupling(self, coupling_id: str) -> Optional[EnvironmentalCoupling]:
        async with self.lock:
            if coupling.id not in self.couplings:
                self.logger.warning(f"Cannot update: Coupling {coupling.id} does not exist")
                return False
                
            existing = self.couplings[coupling.id]
            if (existing.internal_entity_id != coupling.internal_entity_id or 
                existing.environmental_entity_id != coupling.environmental_entity_id):
                self.internal_entity_map[existing.internal_entity_id].discard(coupling.id)
                self.environmental_entity_map[existing.environmental_entity_id].discard(coupling.id)
                
                self.internal_entity_map[coupling.internal_entity_id].add(coupling.id)
                self.environmental_entity_map[coupling.environmental_entity_id].add(coupling.id)
            
            self.couplings[coupling.id] = coupling
            self.logger.debug(f"Updated coupling {coupling.id}")
            return True
    
    async def delete_coupling(self, coupling_id: str) -> bool:
        coupling_ids = self.internal_entity_map.get(entity_id, set())
        return [self.couplings[cid] for cid in coupling_ids if cid in self.couplings]
    
    async def get_couplings_by_environmental_entity(self, entity_id: str) -> List[EnvironmentalCoupling]:
        return [c for c in self.couplings.values() if c.coupling_state == CouplingState.ACTIVE]
    
    async def get_statistics(self) -> Dict: