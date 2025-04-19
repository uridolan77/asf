import logging
from typing import Dict, Any, Optional, List, Tuple
import re
import random

logger = logging.getLogger(__name__)


class ModelSelector:
    """Select the best model for a query."""
    
    def __init__(self, llm_router):
        """Initialize the model selector.
        
        Args:
            llm_router: The LLM router to use
        """
        self.llm_router = llm_router
        self.model_capabilities = {
            "gpt-4": {
                "reasoning": 0.9,
                "creativity": 0.8,
                "knowledge": 0.85,
                "coding": 0.9,
                "cost": 0.1  # Higher means more expensive
            },
            "gpt-3.5-turbo": {
                "reasoning": 0.7,
                "creativity": 0.75,
                "knowledge": 0.7,
                "coding": 0.75,
                "cost": 0.8  # Lower means less expensive
            },
            "claude-3-opus": {
                "reasoning": 0.9,
                "creativity": 0.85,
                "knowledge": 0.9,
                "coding": 0.85,
                "cost": 0.2
            },
            "claude-3-sonnet": {
                "reasoning": 0.8,
                "creativity": 0.8,
                "knowledge": 0.8,
                "coding": 0.8,
                "cost": 0.5
            },
            "claude-3-haiku": {
                "reasoning": 0.7,
                "creativity": 0.7,
                "knowledge": 0.7,
                "coding": 0.7,
                "cost": 0.9
            }
        }
    
    async def select_model(
        self,
        query: str,
        user_preferences: Optional[Dict[str, float]] = None
    ) -> str:
        """Select the best model for a query.
        
        Args:
            query: The query to select a model for
            user_preferences: User preferences for different capabilities
            
        Returns:
            The name of the selected model
        """
        # Use default preferences if none are provided
        if user_preferences is None:
            user_preferences = {
                "reasoning": 0.5,
                "creativity": 0.5,
                "knowledge": 0.5,
                "coding": 0.5,
                "cost": 0.5
            }
        
        # Analyze the query to determine required capabilities
        required_capabilities = await self._analyze_query(query)
        
        # Combine required capabilities with user preferences
        weighted_capabilities = {}
        for capability, weight in required_capabilities.items():
            user_weight = user_preferences.get(capability, 0.5)
            weighted_capabilities[capability] = weight * user_weight
        
        # Score each model
        model_scores = {}
        for model, capabilities in self.model_capabilities.items():
            score = 0
            for capability, weight in weighted_capabilities.items():
                score += capabilities.get(capability, 0) * weight
            model_scores[model] = score
        
        # Return the highest scoring model
        return max(model_scores.items(), key=lambda x: x[1])[0]
    
    async def _analyze_query(self, query: str) -> Dict[str, float]:
        """Analyze a query to determine the required capabilities.
        
        Args:
            query: The query to analyze
            
        Returns:
            A dictionary of capability weights
        """
        # This is a simple heuristic-based approach
        # In a real implementation, you would use a classifier
        
        # Default capabilities
        capabilities = {
            "reasoning": 0.5,
            "creativity": 0.5,
            "knowledge": 0.5,
            "coding": 0.5,
            "cost": 0.5
        }
        
        # Check for coding-related keywords
        if re.search(r'\b(code|function|program|algorithm|class|method|variable|api|json|xml|html|css|javascript|python|java|c\+\+|sql)\b', query.lower()):
            capabilities["coding"] = 0.9
            capabilities["reasoning"] = 0.7
        
        # Check for reasoning-related keywords
        if re.search(r'\b(explain|why|how|reason|analyze|compare|contrast|evaluate|assess|solve|problem)\b', query.lower()):
            capabilities["reasoning"] = 0.8
            capabilities["knowledge"] = 0.7
        
        # Check for creativity-related keywords
        if re.search(r'\b(create|generate|write|story|poem|song|creative|imagine|design|invent|novel|unique)\b', query.lower()):
            capabilities["creativity"] = 0.8
            capabilities["reasoning"] = 0.6
        
        # Check for knowledge-related keywords
        if re.search(r'\b(what is|who is|when|where|history|science|facts|information|details|explain|describe)\b', query.lower()):
            capabilities["knowledge"] = 0.8
            capabilities["reasoning"] = 0.6
        
        return capabilities


class ModelRotator:
    """Rotate between models to distribute load."""
    
    def __init__(self, models: List[str], weights: Optional[List[float]] = None):
        """Initialize the model rotator.
        
        Args:
            models: The models to rotate between
            weights: The weights for each model (optional)
        """
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.current_index = 0
    
    def get_next_model(self) -> str:
        """Get the next model in the rotation.
        
        Returns:
            The name of the next model
        """
        # Use weighted random selection
        model = random.choices(self.models, weights=self.weights, k=1)[0]
        return model
