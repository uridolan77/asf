import json
import re
from typing import Dict, Optional, Any, List, Union
from enum import Enum
from datetime import datetime
import uuid


class MessageType(Enum):
    """Types of MCP messages for environmental coupling"""
    RESOURCE_REQUEST = "Resource.request"      # Request for data
    RESOURCE = "Resource"                      # Data from environment
    RESOURCE_UPDATE = "Resource.update"        # Update to environmental data
    RESOURCE_COLLECTION = "Resource.collection" # Collection of resources
    TOOL_INVOKE = "Tool.invoke"                # Command/action to execute
    TOOL_RESULT = "Tool.result"                # Result of tool execution
    EVENT = "Event"                            # Event notification
    ERROR = "Error"                            # Error message
    HANDSHAKE = "Handshake"                    # Protocol negotiation


class EventPriority(Enum):
    """Priority levels for events"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class MCPMessageSchema:
    """Schema definition and validation for MCP messages"""
    
    # Schema version
    VERSION = "1.0"
    
    # Required fields for each message type
    REQUIRED_FIELDS = {
        MessageType.RESOURCE_REQUEST: ["id", "resourceType"],
        MessageType.RESOURCE: ["id", "content"],
        MessageType.RESOURCE_UPDATE: ["id", "content"],
        MessageType.RESOURCE_COLLECTION: ["id", "items"],
        MessageType.TOOL_INVOKE: ["id", "tool", "parameters"],
        MessageType.TOOL_RESULT: ["id", "result"],
        MessageType.EVENT: ["id", "eventType", "content"],
        MessageType.ERROR: ["id", "code", "message"],
        MessageType.HANDSHAKE: ["version", "capabilities"]
    }
    
    @staticmethod
    def create_message(
        message_type: MessageType, 
        content: Any = None, 
        **kwargs
    ) -> Dict:
        """Create a new MCP message with the specified type and content"""
        message = {
            "type": message_type.value,
            "id": kwargs.get("id", str(uuid.uuid4())),
            "timestamp": kwargs.get("timestamp", datetime.utcnow().isoformat()),
            "version": MCPMessageSchema.VERSION
        }
        
        if "metadata" in kwargs:
            message["metadata"] = kwargs["metadata"]
        
        if message_type == MessageType.RESOURCE_REQUEST:
            message["resourceType"] = kwargs.get("resourceType")
            message["parameters"] = kwargs.get("parameters", {})
            
        elif message_type in [MessageType.RESOURCE, MessageType.RESOURCE_UPDATE]:
            message["content"] = content
            if "contentType" in kwargs:
                message["contentType"] = kwargs["contentType"]
                
        elif message_type == MessageType.RESOURCE_COLLECTION:
            message["items"] = content if isinstance(content, list) else [content]
            message["count"] = len(message["items"])
            
        elif message_type == MessageType.TOOL_INVOKE:
            message["tool"] = kwargs.get("tool")
            message["parameters"] = kwargs.get("parameters", {})
            
        elif message_type == MessageType.TOOL_RESULT:
            message["result"] = content
            if "toolId" in kwargs:
                message["toolId"] = kwargs["toolId"]
                
        elif message_type == MessageType.EVENT:
            message["eventType"] = kwargs.get("eventType")
            message["content"] = content
            message["priority"] = kwargs.get("priority", EventPriority.NORMAL.value)
            
        elif message_type == MessageType.ERROR:
            message["code"] = kwargs.get("code", "unknown_error")
            message["message"] = kwargs.get("message", str(content))
            if "details" in kwargs:
                message["details"] = kwargs["details"]
                
        elif message_type == MessageType.HANDSHAKE:
            message["version"] = kwargs.get("version", MCPMessageSchema.VERSION)
            message["capabilities"] = kwargs.get("capabilities", [])
        
        if "source" in kwargs:
            if "metadata" not in message:
                message["metadata"] = {}
            message["metadata"]["source"] = kwargs["source"]
            
        if "correlationId" in kwargs:
            message["correlationId"] = kwargs["correlationId"]
        
        return message
    
    @staticmethod
    def validate_message(message: Dict) -> bool:
        """Validate that a message conforms to the MCP schema"""
        # Check if message has a type
        if "type" not in message:
            return False
            
        # Check if it's a valid message type
        try:
            msg_type = next(t for t in MessageType if t.value == message["type"])
        except StopIteration:
            return False
            
        # Check required fields for this message type
        required_fields = MCPMessageSchema.REQUIRED_FIELDS.get(msg_type, [])
        for field in required_fields:
            if field not in message:
                return False
                
        # Validate version format if present
        if "version" in message:
            version_pattern = re.compile(r"^\d+\.\d+$")
            if not version_pattern.match(message["version"]):
                return False
        
        return True
    
    @staticmethod
    def message_to_json(message: Dict) -> str:
        """Convert message dict to JSON string"""
        return json.dumps(message)
        
    @staticmethod
    def json_to_message(json_str: str) -> Dict:
        """Convert JSON string to message dict"""
        return json.loads(json_str)


class ContextMetadata:
    """Helper for creating consistent context metadata for MCP messages"""
    
    @staticmethod
    def create_basic_metadata(
        source: str, 
        timestamp: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> Dict:
        result = metadata.copy()
        result["location"] = location
        return result
    
    @staticmethod
    def with_semantic_tags(
        metadata: Dict, 
        tags: List[str]
    ) -> Dict:
        result = metadata.copy()
        result["provenance"] = provenance
        return result


def create_example_messages():
    resource_request = MCPMessageSchema.create_message(
        MessageType.RESOURCE_REQUEST,
        resourceType="sensor_data",
        parameters={"sensor_id": "temp_sensor_1", "time_range": "1h"}
    )
    
    event_metadata = ContextMetadata.create_basic_metadata(source="motion_sensor")
    event_metadata = ContextMetadata.with_location(
        event_metadata, 
        {"lat": 37.7749, "lng": -122.4194, "area": "zone_3"}
    )
    
    event_message = MCPMessageSchema.create_message(
        MessageType.EVENT,
        {"detected": True, "confidence": 0.92},
        eventType="motion_detected",
        priority=EventPriority.HIGH.value,
        metadata=event_metadata
    )
    
    tool_invoke = MCPMessageSchema.create_message(
        MessageType.TOOL_INVOKE,
        tool="camera_control",
        parameters={"action": "pan", "angle": 45, "speed": 0.5}
    )
    
    return {
        "resource_request": resource_request,
        "event": event_message,
        "tool_invoke": tool_invoke
    }


if __name__ == "__main__":
    examples = create_example_messages()
    for name, message in examples.items():
        print(f"\n--- {name} ---")
        print(json.dumps(message, indent=2))
        print(f"Valid: {MCPMessageSchema.validate_message(message)}")