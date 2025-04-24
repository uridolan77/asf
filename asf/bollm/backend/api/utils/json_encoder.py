"""
Custom JSON encoder for serializing complex objects.
"""

import json
from datetime import datetime
from typing import Any

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles serialization of complex objects.
    """
    
    def default(self, obj: Any) -> Any:
        """
        Convert objects to JSON serializable types.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON serializable representation of the object
        """
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle objects with to_dict method
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()
        
        # Handle objects with __dict__ attribute
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        
        # Let the base class handle it or raise TypeError
        return super().default(obj)
