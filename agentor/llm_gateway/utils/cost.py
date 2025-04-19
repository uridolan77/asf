import logging
import re
from typing import Dict, Any, Optional, List, Tuple
import json

logger = logging.getLogger(__name__)


class CostTracker:
    """Track the cost of LLM requests."""
    
    def __init__(self):
        """Initialize the cost tracker."""
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
            "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
            "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
        }
        
        self.total_cost = 0.0
        self.total_tokens = 0
        self.requests_by_model: Dict[str, int] = {}
        self.tokens_by_model: Dict[str, Dict[str, int]] = {}
        self.cost_by_model: Dict[str, float] = {}
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, float]:
        """Calculate the cost of a request.
        
        Args:
            model: The model used
            input_tokens: The number of input tokens
            output_tokens: The number of output tokens
            
        Returns:
            A dictionary with the cost breakdown
        """
        # Get the pricing for the model
        model_pricing = self.pricing.get(model, {"input": 0.01, "output": 0.01})
        
        # Calculate the cost
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        # Update the totals
        self.total_cost += total_cost
        self.total_tokens += input_tokens + output_tokens
        
        # Update the model-specific stats
        self.requests_by_model[model] = self.requests_by_model.get(model, 0) + 1
        
        if model not in self.tokens_by_model:
            self.tokens_by_model[model] = {"input": 0, "output": 0}
        
        self.tokens_by_model[model]["input"] += input_tokens
        self.tokens_by_model[model]["output"] += output_tokens
        
        self.cost_by_model[model] = self.cost_by_model.get(model, 0.0) + total_cost
        
        # Return the cost breakdown
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get the current cost statistics.
        
        Returns:
            A dictionary with the cost statistics
        """
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "requests_by_model": self.requests_by_model,
            "tokens_by_model": self.tokens_by_model,
            "cost_by_model": self.cost_by_model
        }


class TokenCounter:
    """Count tokens in a text."""
    
    def __init__(self):
        """Initialize the token counter."""
        # This is a simple approximation for demonstration purposes
        # In a real implementation, you would use a proper tokenizer
        # like tiktoken or the tokenizer from the model
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text.
        
        This is a simple approximation. In a real implementation,
        you would use a proper tokenizer like tiktoken.
        
        Args:
            text: The text to count tokens in
            
        Returns:
            The number of tokens
        """
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return len(tokens)


class CostAwareLLM:
    """A wrapper around an LLM that tracks cost."""
    
    def __init__(self, llm, cost_tracker: CostTracker, token_counter: TokenCounter):
        """Initialize the cost-aware LLM.
        
        Args:
            llm: The LLM to wrap
            cost_tracker: The cost tracker to use
            token_counter: The token counter to use
        """
        self.llm = llm
        self.cost_tracker = cost_tracker
        self.token_counter = token_counter
    
    async def generate(self, request):
        """Generate a response, tracking the cost.
        
        Args:
            request: The LLM request
            
        Returns:
            The LLM response
        """
        # Count the input tokens
        input_tokens = self.token_counter.count_tokens(request.prompt)
        
        # Generate the response
        response = await self.llm.generate(request)
        
        # Count the output tokens
        output_tokens = self.token_counter.count_tokens(response.text)
        
        # Calculate the cost
        cost = self.cost_tracker.calculate_cost(
            model=request.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        
        # Add the cost to the response metadata
        if response.metadata is None:
            response.metadata = {}
        
        response.metadata["cost"] = cost
        
        return response
