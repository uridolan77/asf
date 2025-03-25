import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class ContradictionManager:
    def __init__(self, knowledge_base):
        """
        Manages contradictions within a knowledge base.

        Args:
            knowledge_base:  A reference to the KnowledgeBase instance.
        """
        self.knowledge_base = knowledge_base  # Reference to the KnowledgeBase
        self.contradiction_thresholds = {
            "direct": 0.7,
            "contextual": 0.5,
            "temporal": 0.4,
            "perspective": 0.6 # New threshold for perspective contradictions
        }
        self.resolution_strategies = [
            "synthesis",
            "contextualization",
            "temporal_sequencing",
            "perspective_framing",
            "rejection",  # Add a rejection strategy
        ]

        # Sentence Transformer for semantic similarity (download a model, e.g., 'all-MiniLM-L6-v2')
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')


    def _calculate_contradiction_score(self, knowledge1, knowledge2):
        """
        Calculates a contradiction score between two knowledge entities.

        Args:
            knowledge1 (dict): First knowledge entity.
            knowledge2 (dict): Second knowledge entity.

        Returns:
            float: Contradiction score (0.0 = no contradiction, 1.0 = strong contradiction).
        """

        # 1. Direct Value Contradiction (for comparable attributes)
        score = 0.0
        num_comparisons = 0

        for key in knowledge1['content']:
            if key in knowledge2['content'] and isinstance(knowledge1['content'][key], (int, float, str)) and isinstance(knowledge2['content'][key], (int,float, str)):
                num_comparisons +=1
                if isinstance(knowledge1['content'][key], (int, float)) and isinstance(knowledge2['content'][key], (int, float)):
                    #Numeric Comparison
                    val1 = knowledge1['content'][key]
                    val2 = knowledge2['content'][key]
                    # Normalize the difference (assuming a reasonable range, e.g., 0-100)
                    # Adjust the range (100) as needed for your data.
                    normalized_diff = abs(val1 - val2) / 100.0
                    score += min(1.0, normalized_diff) # Cap at 1.0

                elif isinstance(knowledge1['content'][key], str) and isinstance(knowledge2['content'][key], str):
                    #String Comparison
                    emb1 = self.sentence_transformer.encode(knowledge1['content'][key], convert_to_tensor=True)
                    emb2 = self.sentence_transformer.encode(knowledge2['content'][key], convert_to_tensor=True)
                    similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
                    score += 1.0 - similarity #Higher similarity, lower contradiction.


        if num_comparisons > 0:
          score /= num_comparisons #average score.

        # 2. Confidence-Weighted Score
        #   Lower confidence in either piece of knowledge reduces the contradiction score.
        combined_confidence = knowledge1['confidence'] * knowledge2['confidence']
        score *= combined_confidence

        return score

    def _is_synthesis_viable(self, knowledge1, knowledge2):
        """
        Checks if synthesis (creating a new concept) is a viable resolution strategy.
        Placeholder for more complex logic that might examine the semantic content of the knowledge.

        """
        #Basic Check: if domains are similar, maybe.
        if knowledge1['domain'] == knowledge2['domain']:
          return True

        return False


    def _identify_differentiating_contexts(self, knowledge1, knowledge2):
        """
        Identifies contextual factors that might differentiate the two knowledge entities.
        """
        # Placeholder for context analysis (e.g., using NLP techniques or examining metadata)
        #This is very hard without having access to all the context in the knowledge graph
        #Here is a *VERY* simplified example.

        context1 = knowledge1.get('context', {}) #get context, or empty dict.
        context2 = knowledge2.get('context', {})

        differentiating_contexts = []

        #Example: Check 'source'
        if 'source' in context1 and 'source' in context2 and context1['source'] != context2['source']:
          differentiating_contexts.append('source')

        #Example: Check Publication year:
        if 'publication_year' in context1 and 'publication_year' in context2 and context1['publication_year'] != context2['publication_year']:
          differentiating_contexts.append('publication_year')
        return differentiating_contexts


    def _create_contextual_boundary(self, knowledge1, knowledge2, contexts):
        """
        Creates a contextual boundary to resolve a contradiction.
        """
        #Placeholder
        boundary_description = f"Knowledge '{knowledge1['content']}' is valid in contexts: {', '.join(contexts)} " \
                              f"while knowledge '{knowledge2['content']}' is valid in other contexts."

        return boundary_description

    def _create_synthesis(self, knowledge1, knowledge2):
        """
        Creates a synthesis of two knowledge items.
        Placeholder.
        """
        #Very Simple Example:
        return f"Synthesis of {knowledge1['content']} and {knowledge2['content']}"

    def _suggest_resolutions(self, new_knowledge, existing_knowledge):
      """Suggests possible resolutions for a detected contradiction."""
      resolutions = []
      contradiction_score = self._calculate_contradiction_score(new_knowledge, existing_knowledge)

      #Direct contradiction
      if contradiction_score > self.contradiction_thresholds["direct"]:
        #Synthesis:
        if self._is_synthesis_viable(new_knowledge, existing_knowledge):
          resolutions.append({
            "strategy": "synthesis",
            "confidence": 0.6, #Confidence in this approach
            "implementation": self._create_synthesis(new_knowledge, existing_knowledge)
          })
        #Contextualization:
        contexts = self._identify_differentiating_contexts(new_knowledge, existing_knowledge)
        if contexts:
            resolutions.append({
                "strategy": "contextualization",
                "confidence": 0.8,
                "implementation": self._create_contextual_boundary(new_knowledge, existing_knowledge, contexts)
            })
        #Rejection (of new knowledge)
        resolutions.append({
            "strategy": "rejection",
            "confidence": 0.7, #Could depend on relative confidences.
            "implementation": f"Reject new knowledge: {new_knowledge['content']}"
        })
      #Temporal
      if 'added_at' in new_knowledge and 'added_at' in existing_knowledge:
        if new_knowledge['added_at'] > existing_knowledge['added_at']:
          resolutions.append({
            "strategy": "temporal_sequencing",
            "confidence": 0.9,
            "implementation": f"Knowledge '{existing_knowledge['content']}' precedes '{new_knowledge['content']}'"
          })
        elif new_knowledge['added_at'] < existing_knowledge['added_at']:
          resolutions.append({
             "strategy": "temporal_sequencing",
             "confidence": 0.9,
             "implementation": f"Knowledge '{new_knowledge['content']}' precedes '{existing_knowledge['content']}'"
          })

      #Perspective (Example: If sources differ significantly)
      if 'source' in new_knowledge and 'source' in existing_knowledge and new_knowledge['source'] != existing_knowledge['source']:
        #Check if sources are significantly different using embeddings
        source_emb1 = self.sentence_transformer.encode(new_knowledge['source'], convert_to_tensor=True)
        source_emb2 = self.sentence_transformer.encode(existing_knowledge['source'], convert_to_tensor=True)
        source_similarity = torch.nn.functional.cosine_similarity(source_emb1, source_emb2, dim=0).item()

        if source_similarity < self.contradiction_thresholds["perspective"]:
          resolutions.append({
            "strategy": "perspective_framing",
            "confidence": 0.7,
            "implementation": f"Knowledge '{new_knowledge['content']}' from perspective {new_knowledge.get('source','N/A')}, '{existing_knowledge['content']}' from perspective {existing_knowledge.get('source', 'N/A')}"
          })
      return resolutions

    def detect_contradictions(self, new_knowledge_id):
        """
        Detects contradictions between new knowledge and existing knowledge in the base.

        Args:
            new_knowledge_id (int): ID of the newly added knowledge entity.

        Returns:
            list: List of detected contradictions (dictionaries).
        """
        contradictions = []
        new_knowledge = self.knowledge_base.get_knowledge(new_knowledge_id)
        if new_knowledge is None:
            return []  # Or raise an exception

        for existing_id, existing_knowledge in self.knowledge_base.get_all_knowledge().items():
            if existing_id != new_knowledge_id:  # Don't compare to itself
                resolutions = self._suggest_resolutions(new_knowledge, existing_knowledge)
                if resolutions:
                    contradictions.append({
                        "new_knowledge_id": new_knowledge_id,
                        "existing_knowledge_id": existing_id,
                        "resolutions": resolutions,
                    })
        return contradictions

    def resolve_contradiction(self, new_knowledge_id, existing_knowledge_id, chosen_strategy):
      """Resolves a chosen contradiction."""
      new_knowledge = self.knowledge_base.get_knowledge(new_knowledge_id)
      existing_knowledge = self.knowledge_base.get_knowledge(existing_knowledge_id)

      if chosen_strategy == "synthesis":
        #Implement your logic.  Need to create a new knowledge entry.
        synthesis_result = self._create_synthesis(new_knowledge, existing_knowledge)
        new_id = self.knowledge_base.add_knowledge("synthesized", synthesis_result) #Need domain
        return f"Synthesized new knowledge (ID: {new_id}): {synthesis_result}"
      elif chosen_strategy == "contextualization":
        #Need to update metadata of *both* knowledge items.
        contexts = self._identify_differentiating_contexts(new_knowledge, existing_knowledge)
        boundary = self._create_contextual_boundary(new_knowledge, existing_knowledge, contexts)

        #Update the context of the new knowledge:
        if "context" not in new_knowledge:
          new_knowledge['context'] = {}
        new_knowledge['context']['validity'] = f"Valid in contexts: {', '.join(contexts)}"
        self.knowledge_base.update_knowledge(new_knowledge_id, new_knowledge)

        if "context" not in existing_knowledge:
          existing_knowledge['context'] = {}
        existing_knowledge['context']['validity'] = f"Valid in contexts other than: {', '.join(contexts)}"
        self.knowledge_base.update_knowledge(existing_knowledge_id, existing_knowledge)

        return boundary

      elif chosen_strategy == "temporal_sequencing":
        #Update temporal metadata
        if new_knowledge['added_at'] > existing_knowledge['added_at']:
          result_string = f"Knowledge '{existing_knowledge['content']}' precedes '{new_knowledge['content']}'"
          existing_knowledge['temporal_status'] = 'superseded'
          new_knowledge['temporal_status'] = 'current'
        else:
          result_string = f"Knowledge '{new_knowledge['content']}' precedes '{existing_knowledge['content']}'"
          new_knowledge['temporal_status'] = 'superseded'
          existing_knowledge['temporal_status'] = 'current'
        self.knowledge_base.update_knowledge(new_knowledge_id, new_knowledge)
        self.knowledge_base.update_knowledge(existing_knowledge_id, existing_knowledge)
        return result_string

      elif chosen_strategy == "perspective_framing":
        new_knowledge['context']['perspective'] = new_knowledge.get('source', "Unknown")
        existing_knowledge['context']['perspective'] = existing_knowledge.get('source', "Unknown")
        self.knowledge_base.update_knowledge(new_knowledge_id, new_knowledge)
        self.knowledge_base.update_knowledge(existing_knowledge_id, existing_knowledge)
        return f"Framed knowledge within perspectives: {new_knowledge.get('source', 'N/A')} and {existing_knowledge.get('source', 'N/A')}"

      elif chosen_strategy == "rejection":
        #Don't add the new knowledge, or mark it as rejected.
        return f"Rejected new knowledge: {new_knowledge['content']}"
      else:
        return "Invalid Resolution Strategy"