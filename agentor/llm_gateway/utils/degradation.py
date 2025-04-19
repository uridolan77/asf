import logging
from typing import Dict, Any, Optional, List, Set
import time
import asyncio
import threading

logger = logging.getLogger(__name__)


class DegradationManager:
    """Manage service degradation levels."""
    
    def __init__(self):
        """Initialize the degradation manager."""
        self.degradation_levels = {
            "normal": {
                "models": ["gpt-4", "claude-3-opus"],
                "max_tokens": 4096,
                "tools_enabled": True,
                "cache_ttl": 3600,
                "rate_limit": 100
            },
            "moderate": {
                "models": ["gpt-3.5-turbo", "claude-3-haiku"],
                "max_tokens": 2048,
                "tools_enabled": True,
                "cache_ttl": 7200,
                "rate_limit": 50
            },
            "severe": {
                "models": ["gpt-3.5-turbo", "text-babbage-002"],
                "max_tokens": 1024,
                "tools_enabled": False,
                "cache_ttl": 14400,
                "rate_limit": 20
            }
        }
        
        self.current_level = "normal"
        self.lock = threading.RLock()
        self.last_change_time = time.time()
        self.metrics: Dict[str, Any] = {
            "error_rate": 0.0,
            "latency": 0.0,
            "requests_per_second": 0.0
        }
    
    def set_degradation_level(self, level: str):
        """Set the degradation level.
        
        Args:
            level: The degradation level to set
        """
        with self.lock:
            if level in self.degradation_levels:
                if level != self.current_level:
                    logger.warning(f"Changing degradation level from {self.current_level} to {level}")
                    self.current_level = level
                    self.last_change_time = time.time()
    
    def get_settings(self) -> Dict[str, Any]:
        """Get the settings for the current degradation level.
        
        Returns:
            The settings for the current degradation level
        """
        with self.lock:
            return self.degradation_levels[self.current_level].copy()
    
    def should_enable_tool(self, tool_name: str) -> bool:
        """Check if a tool should be enabled.
        
        Args:
            tool_name: The name of the tool
            
        Returns:
            True if the tool should be enabled, False otherwise
        """
        with self.lock:
            return self.degradation_levels[self.current_level]["tools_enabled"]
    
    def update_metrics(self, error_rate: float, latency: float, requests_per_second: float):
        """Update the metrics used to determine the degradation level.
        
        Args:
            error_rate: The error rate (0.0 to 1.0)
            latency: The average latency in seconds
            requests_per_second: The number of requests per second
        """
        with self.lock:
            self.metrics["error_rate"] = error_rate
            self.metrics["latency"] = latency
            self.metrics["requests_per_second"] = requests_per_second
            
            # Automatically adjust the degradation level based on metrics
            self._adjust_degradation_level()
    
    def _adjust_degradation_level(self):
        """Adjust the degradation level based on the current metrics."""
        # Only adjust if it's been at least 5 minutes since the last change
        if time.time() - self.last_change_time < 300:
            return
        
        # Determine the appropriate level based on metrics
        if self.metrics["error_rate"] > 0.1 or self.metrics["latency"] > 5.0:
            # High error rate or latency, use severe degradation
            self.set_degradation_level("severe")
        elif self.metrics["error_rate"] > 0.05 or self.metrics["latency"] > 2.0:
            # Moderate error rate or latency, use moderate degradation
            self.set_degradation_level("moderate")
        else:
            # Low error rate and latency, use normal operation
            self.set_degradation_level("normal")


class DegradationAwareLLM:
    """A wrapper around an LLM that respects degradation settings."""
    
    def __init__(self, llm, degradation_manager: DegradationManager):
        """Initialize the degradation-aware LLM.
        
        Args:
            llm: The LLM to wrap
            degradation_manager: The degradation manager to use
        """
        self.llm = llm
        self.degradation_manager = degradation_manager
    
    async def generate(self, request):
        """Generate a response, respecting degradation settings.
        
        Args:
            request: The LLM request
            
        Returns:
            The LLM response
        """
        # Get the current settings
        settings = self.degradation_manager.get_settings()
        
        # Check if the requested model is allowed
        if request.model not in settings["models"]:
            # Use the first allowed model instead
            original_model = request.model
            request.model = settings["models"][0]
            logger.warning(f"Model {original_model} not allowed in current degradation level, using {request.model} instead")
        
        # Limit the max tokens
        if request.max_tokens is None or request.max_tokens > settings["max_tokens"]:
            request.max_tokens = settings["max_tokens"]
        
        # Generate the response
        return await self.llm.generate(request)
