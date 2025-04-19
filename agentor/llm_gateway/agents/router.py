import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import openai

logger = logging.getLogger(__name__)


class SemanticRouter:
    """A router that uses semantic similarity to route queries to the appropriate agent."""
    
    def __init__(self, embedding_model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """Initialize the semantic router.
        
        Args:
            embedding_model: The name of the embedding model to use
            api_key: The OpenAI API key (optional, can be set in the environment)
        """
        self.embedding_model = embedding_model
        self.route_embeddings: Dict[str, List[float]] = {}
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get an embedding for a text.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding
        """
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate the cosine similarity between two vectors.
        
        Args:
            a: The first vector
            b: The second vector
            
        Returns:
            The cosine similarity
        """
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
    
    async def add_route(self, intent: str, description: str):
        """Add a route to the router.
        
        Args:
            intent: The intent name
            description: A description of the intent
        """
        embedding = await self._get_embedding(description)
        self.route_embeddings[intent] = embedding
        logger.info(f"Added route for intent: {intent}")
    
    async def route(self, query: str) -> str:
        """Route a query to the appropriate intent.
        
        Args:
            query: The query to route
            
        Returns:
            The intent name
        """
        if not self.route_embeddings:
            raise ValueError("No routes have been added")
        
        query_embed = await self._get_embedding(query)
        
        similarities = {
            intent: self._cosine_similarity(query_embed, embed)
            for intent, embed in self.route_embeddings.items()
        }
        
        # Get the intent with the highest similarity
        best_intent = max(similarities.items(), key=lambda x: x[1])
        logger.info(f"Routed query to intent: {best_intent[0]} (similarity: {best_intent[1]:.4f})")
        
        return best_intent[0]
