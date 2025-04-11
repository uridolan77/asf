import uuid
import time
import numpy as np
from scipy.stats import beta
import datetime
from typing import Dict, List, Optional, Any

class TemporalKnowledgeManager:
    def __init__(self, domain_decay_rates: Dict[str, float]):
        """
        Manages knowledge with temporal decay, using Beta distributions for confidence.

        Args:
            domain_decay_rates (dict): Dictionary mapping domain (str) to decay rate (float).
        """
        self.knowledge: Dict[str, Dict] = {}  # {knowledge_id: {domain: str, added_at: datetime, alpha: float, beta: float, justifications: list}}
        self.domain_decay_rates = domain_decay_rates
        self.decay_type = "linear" # Can be set to "linear" or "exponential"

    def add_temporal_knowledge(self, domain: str, knowledge_content: str, initial_confidence: float = 0.9, justifications: Optional[List[str]] = None) -> str:
        """
        Adds a new piece of temporal knowledge.

        Args:
            domain (str): The domain of the knowledge.
            knowledge_content (str): The knowledge itself.
            initial_confidence (float): Initial confidence (0-1).
            justifications (list): List of reasons/sources for this knowledge.
        """
        knowledge_id = str(uuid.uuid4())  # Use UUIDs

        if domain not in self.domain_decay_rates:
            raise ValueError(f"Domain '{domain}' not found in decay rates.")
        if not 0 <= initial_confidence <= 1:
            raise ValueError("Initial confidence must be between 0 and 1.")

        alpha = initial_confidence * 10 + 1
        beta = (1 - initial_confidence) * 10 + 1

        self.knowledge[knowledge_id] = {
            'domain': domain,
            'added_at': datetime.datetime.now(),
            'alpha': alpha,
            'beta': beta,
            'content': knowledge_content,
            'justifications': justifications if justifications else []
        }
        return knowledge_id

    def get_knowledge(self, knowledge_id: str, current_time: Optional[datetime.datetime] = None) -> Optional[Dict]:
        """
        Retrieves knowledge and updates confidence based on time decay.

        Args:
            knowledge_id (str): The ID of the knowledge to retrieve.
            current_time (datetime): The current time (defaults to now).

        Returns:
            dict or None: Knowledge information (including updated confidence) or None if not found.
        """
        if knowledge_id not in self.knowledge:
            return None

        knowledge_data = self.knowledge[knowledge_id]
        if current_time is None:
            current_time = datetime.datetime.now()

        time_elapsed = (current_time - knowledge_data['added_at']).total_seconds()
        decay_rate = self.domain_decay_rates.get(knowledge_data['domain'], 0.0001)

        if self.decay_type == "linear":
            knowledge_data['beta'] += time_elapsed * decay_rate
        elif self.decay_type == "exponential":
            knowledge_data['beta'] += knowledge_data['beta'] * time_elapsed * decay_rate
        else:
             raise ValueError("decay_type must be one of 'linear' or 'exponential'")

        current_confidence = knowledge_data['alpha'] / (knowledge_data['alpha'] + knowledge_data['beta'])

        return {
            'knowledge_id': knowledge_id,
            'domain': knowledge_data['domain'],
            'content': knowledge_data['content'],
            'confidence': current_confidence,
            'alpha': knowledge_data['alpha'],
            'beta': knowledge_data['beta'],
            'justifications': knowledge_data['justifications']
        }

    def update_confidence(self, knowledge_id: str, success: bool, observation_strength: float = 1.0):
        """
        Updates the confidence (alpha and beta) based on usage (success/failure).

        Args:
            knowledge_id (str): ID of the knowledge to update.
            success (bool): Whether the knowledge was successfully used.
            observation_strength (float):  How strongly this observation should influence the belief.
        """
        if knowledge_id not in self.knowledge:
            raise KeyError(f"Knowledge ID '{knowledge_id}' not found.")
        if not 0 <= observation_strength <= 1:
          raise ValueError("observation strength must be between zero and one.")

        knowledge_data = self.knowledge[knowledge_id]

        if success:
            knowledge_data['alpha'] += observation_strength
        else:
            knowledge_data['beta'] += observation_strength

        total = knowledge_data['alpha'] + knowledge_data['beta']
        knowledge_data['alpha'] = knowledge_data['alpha'] / total * 100
        knowledge_data['beta'] = knowledge_data['beta'] / total * 100



    def get_all_knowledge(self, current_time:Optional[datetime.datetime] = None) -> Dict[str, Dict]:
        """
        Retrieves all knowledge, updating confidences based on time.
        Args:
          current_time(Optional[datetime]): The current time, for calculating temporal decay.
        Returns:
            Dict[str, Dict]: A dictionary where keys are knowledge IDs and values are knowledge dictionaries
        """
        all_knowledge = {}
        for knowledge_id in self.knowledge:
            all_knowledge[knowledge_id] = self.get_knowledge(knowledge_id, current_time)
        return all_knowledge


    def save_to_file(self, filename: str):
        """Saves the knowledge to a JSON file (for simple persistence)."""
        import json
        data_to_save = {
            'knowledge': self.knowledge,
            'domain_decay_rates': self.domain_decay_rates,
             # Don't save next_id; it's re-initialized
        }
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=4, default=str)  # Use default=str for datetime

    def load_from_file(self, filename: str):
        """Loads knowledge from a JSON file."""
        import json
        with open(filename, 'r') as f:
            loaded_data = json.load(f)
            self.knowledge = loaded_data['knowledge']
            self.domain_decay_rates = loaded_data['domain_decay_rates']


    def integrate_with_substrate(self, substrate):
        """Example of how to integrate with the KnowledgeSubstrateLayer."""
        # This is a simplified example, showing how the substrate could use this manager.
        substrate.temporal_knowledge_manager = self  # Add this manager to the substrate

        # Example: Add temporal knowledge during process_input
        original_process_input = substrate.process_input

        async def new_process_input(input_data, input_type, context=None):
            entity_id = await original_process_input(input_data, input_type, context)

            # Add some temporal knowledge (example)
            if input_type == PerceptualInputType.TEXT:
                knowledge_id = self.add_temporal_knowledge(
                    domain="text_analysis",
                    knowledge_content=f"Entity {entity_id} was created from text input.",
                    initial_confidence=0.8,
                    justifications=["process_input method"]
                )
                print(f"Added temporal knowledge: {knowledge_id}")

            return entity_id

        substrate.process_input = new_process_input # Monkey-patch for demonstration.  A cleaner way is to use composition or inheritance.

# --- Example usage
domain_decay_rates = {
    "physics": 0.00001,
    "history": 0.000005,
    "politics": 0.0001,
    "text_analysis": 0.00005,
}
manager = TemporalKnowledgeManager(domain_decay_rates)
#add some knowledge
knowledge_id1 = manager.add_temporal_knowledge("physics", "Gravity exists.", 0.95, ["Newton's observations", "Einstein's theory"])
knowledge_id2 = manager.add_temporal_knowledge("history", "The Roman Empire fell.", 0.99, ["Historical records", "Archaeological evidence"])
knowledge_id3 = manager.add_temporal_knowledge("politics", "Current president is X.", 0.8, ["News reports"])

# Save to JSON
manager.save_to_file("knowledge.json")

# Load from JSON
new_manager = TemporalKnowledgeManager({})  # Initialize with empty rates
new_manager.load_from_file("knowledge.json")
print(new_manager.get_all_knowledge())