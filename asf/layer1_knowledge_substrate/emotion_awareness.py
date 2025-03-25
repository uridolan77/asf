import numpy as np
from typing import List, Dict, Tuple, Optional

class EmotionAwareKnowledgeBase:
    """
    A knowledge base that incorporates emotional context to influence knowledge
    access and updates.
    """

    def __init__(self, initial_valence: float = 0.0, initial_arousal: float = 0.0):
        """
        Initializes the EmotionAwareKnowledgeBase.

        Args:
            initial_valence: Initial valence (positive/negative feeling).
            initial_arousal: Initial arousal (intensity of feeling).
        """
        self.knowledge: Dict[str, Dict] = {}  # {knowledge_id: {content, valence, arousal, confidence, ...}}
        self.valence = initial_valence
        self.arousal = initial_arousal
        self.valence_threshold_factor = 0.2  # How much valence influences access
        self.arousal_threshold_factor = 0.3 # How much arousal influences access

    def add_knowledge(self, knowledge_id: str, content: str, valence: float, arousal: float,
                      confidence: float = 1.0, justifications: Optional[List[str]] = None,
                      context: Optional[Dict] = None, timestamp: Optional[float] = None):
        """
        Adds knowledge to the knowledge base.

        Args:
            knowledge_id: Unique identifier for the knowledge.
            content: The knowledge content.
            valence: Emotional valence associated with the knowledge.
            arousal: Emotional arousal associated with the knowledge.
            confidence: Confidence in the knowledge.
            justifications: List of justifications for the knowledge.
            context: Contextual information.
            timestamp: Timestamp of knowledge creation.
        """
        if knowledge_id in self.knowledge:
            raise ValueError(f"Knowledge with ID '{knowledge_id}' already exists.")

        self.knowledge[knowledge_id] = {
            "content": content,
            "valence": valence,
            "arousal": arousal,
            "confidence": confidence,
            "justifications": justifications if justifications else [],
            "context": context if context else {},
            "timestamp": timestamp,
        }
        self._update_emotional_state(valence, arousal)  # Update KB's overall emotional state

    def get_knowledge(self, knowledge_id: str) -> Optional[Dict]:
        """
        Retrieves knowledge by ID.

        Args:
            knowledge_id: The ID of the knowledge to retrieve.

        Returns:
            The knowledge item (dictionary), or None if not found.
        """
        return self.knowledge.get(knowledge_id)

    def query_knowledge(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Queries the knowledge base, considering both content relevance and emotional context.

        Args:
            query: The query string.
            top_k: The maximum number of results to return.

        Returns:
            A list of (knowledge_id, score) tuples, sorted by score (highest first).
        """
        results = []
        for knowledge_id, knowledge_item in self.knowledge.items():
            content_relevance = self._calculate_content_relevance(query, knowledge_item["content"])
            emotional_relevance = self._calculate_emotional_relevance(knowledge_item["valence"], knowledge_item["arousal"])
            combined_score = (0.7 * content_relevance) + (0.3 * emotional_relevance)  # Weighted combination
            results.append((knowledge_id, combined_score))

        # Sort by combined score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def update_knowledge(self, knowledge_id: str, updated_content: Optional[str] = None,
                         updated_valence: Optional[float] = None, updated_arousal: Optional[float] = None,
                         updated_confidence: Optional[float] = None, add_justifications: Optional[List[str]] = None,
                         update_context: Optional[Dict] = None):
      """
      Updates existing knowledge in the knowledge base.
      Allows targeted updates of specific fields, which is better for efficiency and tracking.

      Args:
          knowledge_id (str): The ID of the knowledge item to update.
          updated_content (Optional[str]): New content for the knowledge item.
          updated_valence (Optional[float]): New valence value.
          updated_arousal (Optional[float]): New arousal value.
          updated_confidence (Optional[float]): New confidence value.
          add_justifications (Optional[List[str]]): List of justifications to *add*.
          update_context (Optional[Dict]): Dictionary to *update* the existing context.
      """
      if knowledge_id not in self.knowledge:
        raise ValueError(f"Knowledge with ID '{knowledge_id}' not found.")

      knowledge_item = self.knowledge[knowledge_id]

      if updated_content is not None:
          knowledge_item['content'] = updated_content
      if updated_valence is not None:
          knowledge_item['valence'] = updated_valence
          self._update_emotional_state(updated_valence, 0)  # Update KB's state based on change
      if updated_arousal is not None:
          knowledge_item['arousal'] = updated_arousal
          self._update_emotional_state(0, updated_arousal)
      if updated_confidence is not None:
          knowledge_item['confidence'] = updated_confidence
      if add_justifications is not None:
          knowledge_item['justifications'].extend(add_justifications)  # *Append* justifications
      if update_context is not None:
          knowledge_item['context'].update(update_context)  # *Update* the context dictionary

      knowledge_item['timestamp'] = time.time() #Update the timestamp


    def _calculate_content_relevance(self, query: str, content: str) -> float:
        """
        Calculates the relevance of the query to the knowledge content.

        Args:
            query: The query string.
            content: The knowledge content.

        Returns:
            A relevance score (0.0 to 1.0).
        """
        # Simple example: Check for keyword overlap (case-insensitive)
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        common_words = query_words.intersection(content_words)
        return len(common_words) / len(query_words) if query_words else 0.0

    def _calculate_emotional_relevance(self, knowledge_valence: float, knowledge_arousal: float) -> float:
        """
        Calculates the emotional relevance of the knowledge to the current emotional state.

        Args:
            knowledge_valence: Valence of the knowledge.
            knowledge_arousal: Arousal of the knowledge.

        Returns:
            An emotional relevance score (0.0 to 1.0).
        """
        # Simple example: Higher relevance if valence and arousal are close to current state
        valence_diff = abs(self.valence - knowledge_valence)
        arousal_diff = abs(self.arousal - knowledge_arousal)

        # Normalize the differences (assuming valence and arousal are in range [-1, 1])
        normalized_valence_diff = valence_diff / 2.0
        normalized_arousal_diff = arousal_diff / 2.0

        # Calculate relevance (inversely proportional to the differences)
        valence_relevance = 1.0 - (normalized_valence_diff * self.valence_threshold_factor)
        arousal_relevance = 1.0 - (normalized_arousal_diff * self.arousal_threshold_factor)
        # Combine valence and arousal relevance, cap at 1.
        combined_relevance = min(1.0, (valence_relevance + arousal_relevance) / 2.0)

        return max(0.0, combined_relevance) #Ensure 0 or above.

    def _update_emotional_state(self, valence_change: float, arousal_change: float):
        """
        Updates the overall emotional state of the knowledge base.

        Args:
            valence_change: Change in valence.
            arousal_change: Change in arousal.
        """
        # Simple update rule: Weighted average with current state
        self.valence = 0.8 * self.valence + 0.2 * valence_change
        self.arousal = 0.8 * self.arousal + 0.2 * arousal_change

        # Keep valence and arousal within range [-1, 1]
        self.valence = max(-1, min(1, self.valence))
        self.arousal = max(-1, min(1, self.arousal))

    def get_emotional_state(self) -> Tuple[float, float]:
      """Returns the current emotional state (valence, arousal)"""
      return self.valence, self.arousal

    def set_emotional_state(self, valence:float, arousal:float):
      """Sets a specific emotional state"""
      self.valence = max(-1, min(1, valence)) #Clamp
      self.arousal = max(-1, min(1, arousal))

# --- Example Usage ---
import time #Import time to use timestamps.

kb = EmotionAwareKnowledgeBase()

kb.add_knowledge("fact_1", "The sky is blue.", 0.8, 0.2)
kb.add_knowledge("fact_2", "Rain is wet.", 0.1, 0.1)
kb.add_knowledge("fact_3", "Fire is hot.", -0.5, 0.9, justifications=["observation", "experiment"])

print("Initial emotional state:", kb.get_emotional_state())

# Query for "sky"
results = kb.query_knowledge("sky")
print("\nQuery results for 'sky':", results)
for result_id, score in results:
    print(f"  - {result_id}: {kb.get_knowledge(result_id)}, Score: {score:.2f}")

# Update knowledge
kb.update_knowledge("fact_1", updated_content = "The sky is often blue.", updated_valence=0.9, updated_arousal = 0.3)
print("\nEmotional state after updating fact_1:", kb.get_emotional_state())

#Query again
results = kb.query_knowledge("sky")
print("\nQuery results for 'sky' after update:", results)
for result_id, score in results:
    print(f"  - {result_id}: {kb.get_knowledge(result_id)}, Score: {score:.2f}")

#Set a specific state.
kb.set_emotional_state(-0.9, 0.8)
print("\nManually set emotional State:", kb.get_emotional_state())

#Query in this new state
results = kb.query_knowledge("fire")
print("\nQuery results for 'fire' after setting emotional state:", results)
for result_id, score in results:
    print(f"  - {result_id}: {kb.get_knowledge(result_id)}, Score: {score:.2f}")