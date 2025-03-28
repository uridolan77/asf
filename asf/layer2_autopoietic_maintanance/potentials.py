import uuid
from typing import Dict, Any, Optional
from .utils import DatabaseDriver


class SymbolicPotential:
    def __init__(self, activation_value: float,
                 meaning_potential: Dict[str, Any],
                 contextual_relevance: float,
                 potential_id: Optional[str] = None, # Allow ID
                 db_driver: Optional[DatabaseDriver] = None):
        self.potential_id = potential_id or str(uuid.uuid4())
        self.activation_value = activation_value
        self.meaning_potential = meaning_potential
        self.contextual_relevance = contextual_relevance
        self.db_driver = db_driver


    async def save(self, db_driver: Optional[DatabaseDriver] = None):
        """Saves the potential to the database."""
        db_driver = db_driver or self.db_driver
        if not db_driver:
            raise ValueError("Database driver not set.")

        potential_data = {
            "potential_id": self.potential_id,
            "activation_value": self.activation_value,
            "meaning_potential": self.meaning_potential,  #  Neo4j handles dictionaries
            "contextual_relevance": self.contextual_relevance,
        }
        await db_driver.add_node("Potential", potential_data)


    @classmethod
    async def load(cls, potential_id: str, db_driver: DatabaseDriver) -> "SymbolicPotential":
        """Loads a potential from the database."""
        potential_data = await db_driver.get_node("Potential", "potential_id", potential_id)
        if not potential_data:
            return None

        return cls(
            activation_value=potential_data.get("activation_value"),
            meaning_potential=potential_data.get("meaning_potential"),
            contextual_relevance=potential_data.get("contextual_relevance"),
            potential_id=potential_data.get("potential_id"),
            db_driver=db_driver,
        )