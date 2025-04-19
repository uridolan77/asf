from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TypeVar, Generic
from pydantic import BaseModel


class ToolResult(BaseModel):
    """Result from a tool execution."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BaseTool(ABC):
    """Base class for all tools."""
    
    def __init__(self, name: str, description: str):
        """Initialize the tool.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    async def run(self, **kwargs) -> ToolResult:
        """Run the tool with the given parameters.
        
        Args:
            **kwargs: The parameters for the tool
            
        Returns:
            The result of running the tool
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the schema for the tool parameters.
        
        Returns:
            A dictionary describing the parameters for the tool
        """
        # This would typically be implemented by subclasses
        # to provide a JSON schema for the parameters
        return {}
