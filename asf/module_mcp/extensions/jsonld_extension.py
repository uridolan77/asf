import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

class MCPJSONLDMessage:
    def __init__(self, message_type, content, contexts=None):
        self.message = {
            "@context": self._build_context(contexts),
            "type": message_type,
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "content": content
        }
    
    def _build_context(self, custom_contexts=None):
        # Base context with core MCP vocabulary
        context = {
            "@vocab": "https://mcp.yourdomain.com/vocab/",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "schema": "https://schema.org/",
            
            # Define core terms with specific types
            "timestamp": {"@type": "xsd:dateTime"},
            "location": {"@type": "schema:Place"},
            "confidence": {"@type": "xsd:float"},
            "priority": {"@type": "xsd:string"}
        }
        
        # Add custom contexts if provided
        if custom_contexts:
            context.update(custom_contexts)
            
        return context

# Environmental sensors context
SENSOR_CONTEXT = {
    "sensors": "https://yourdomain.com/vocab/sensors/",
    "reading": {"@id": "sensors:reading"},
    "unit": {"@id": "sensors:unit"},
    "accuracy": {"@id": "sensors:accuracy", "@type": "xsd:float"},
    "sampleRate": {"@id": "sensors:sampleRate"}
}

# Spatial context
SPATIAL_CONTEXT = {
    "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "coordinates": {"@id": "geo:Point"},
    "latitude": {"@id": "geo:lat", "@type": "xsd:float"},
    "longitude": {"@id": "geo:long", "@type": "xsd:float"},
    "elevation": {"@id": "geo:alt", "@type": "xsd:float"}
}

class MCPSerializer:
    """Abstract serializer for MCP messages"""
    
    def serialize(self, message):
        """Convert message to wire format"""
        pass
        
    def deserialize(self, data):
        """Convert wire format back to message object"""
        pass


class JSONLDSerializer(MCPSerializer):
    """JSON-LD implementation of MCP serializer"""
    
    def serialize(self, message):
        return json.dumps(message)
        
    def deserialize(self, data):
        return json.loads(data)


# To be implemented later
class ProtobufSerializer(MCPSerializer):
    """Protocol Buffers implementation of MCP serializer"""
    
    def serialize(self, message):
        # Will convert message to protobuf and serialize
        pass
        
    def deserialize(self, data):
        # Will parse protobuf bytes and convert to message dict
        pass

class MessageFormatManager:
    def __init__(self):
        self.formats = {}
        self.default_format = "jsonld"
        
    def register_format(self, name, serializer, deserializer):
        self.formats[name] = {
            "serialize": serializer,
            "deserialize": deserializer
        }
        
    def set_default_format(self, format_name):
        if format_name in self.formats:
            self.default_format = format_name


class SensorReading:
    """Domain model independent of serialization format"""
    def __init__(self, sensor_id, value, unit=None):
        self.sensor_id = sensor_id
        self.value = value
        self.unit = unit
        
    def to_dict(self):
        """Convert to format-neutral dictionary"""
        return {
            "sensorId": self.sensor_id,
            "value": self.value,
            "unit": self.unit
        }

class MCPContextRegistry:
    """Registry of JSON-LD contexts for different domains"""
    
    # Core MCP context with basic vocabulary
    CORE_CONTEXT = {
        "mcp": "https://modelcontextprotocol.io/vocab#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "schema": "https://schema.org/",
        
        # Core MCP terms
        "id": "@id",
        "type": "@type",
        "timestamp": {"@id": "mcp:timestamp", "@type": "xsd:dateTime"},
        "source": {"@id": "mcp:source"},
        "confidence": {"@id": "mcp:confidence", "@type": "xsd:float"},
        "correlationId": {"@id": "mcp:correlationId"},
        "priority": {"@id": "mcp:priority"}
    }
    
    # Sensor-related context
    SENSOR_CONTEXT = {
        "sensors": "https://modelcontextprotocol.io/vocab/sensors#",
        "reading": {"@id": "sensors:reading"},
        "unit": {"@id": "sensors:unit"},
        "accuracy": {"@id": "sensors:accuracy", "@type": "xsd:float"},
        "sampleRate": {"@id": "sensors:sampleRate", "@type": "xsd:float"}
    }
    
    # Spatial context using standard WGS84 ontology
    SPATIAL_CONTEXT = {
        "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
        "location": {"@id": "schema:location", "@type": "@id"},
        "coordinates": {"@id": "geo:Point"},
        "latitude": {"@id": "geo:lat", "@type": "xsd:float"},
        "longitude": {"@id": "geo:long", "@type": "xsd:float"},
        "elevation": {"@id": "geo:alt", "@type": "xsd:float"},
        "zone": {"@id": "mcp:zone"}
    }
    
    # Event-related context
    EVENT_CONTEXT = {
        "event": "https://modelcontextprotocol.io/vocab/events#",
        "eventType": {"@id": "event:type"},
        "occurred": {"@id": "event:occurred", "@type": "xsd:dateTime"},
        "detected": {"@id": "event:detected", "@type": "xsd:dateTime"},
        "category": {"@id": "event:category"}
    }
    
    # Tool/command-related context
    TOOL_CONTEXT = {
        "tool": "https://modelcontextprotocol.io/vocab/tools#",
        "action": {"@id": "tool:action"},
        "parameters": {"@id": "tool:parameters"},
        "result": {"@id": "tool:result"},
        "status": {"@id": "tool:status"}
    }
    
    @classmethod
    def get_context(cls, *domains):
        """
        Get combined context for specified domains
        
        Args:
            *domains: Variable list of domain names ('core', 'sensor', 'spatial', etc.)
            
        Returns:
            Combined context dictionary
        """
        context = {}
        
        # Always include core context
        context.update(cls.CORE_CONTEXT)
        
        # Add requested domain contexts
        for domain in domains:
            if domain == 'sensor' and hasattr(cls, 'SENSOR_CONTEXT'):
                context.update(cls.SENSOR_CONTEXT)
            elif domain == 'spatial' and hasattr(cls, 'SPATIAL_CONTEXT'):
                context.update(cls.SPATIAL_CONTEXT)
            elif domain == 'event' and hasattr(cls, 'EVENT_CONTEXT'):
                context.update(cls.EVENT_CONTEXT)
            elif domain == 'tool' and hasattr(cls, 'TOOL_CONTEXT'):
                context.update(cls.TOOL_CONTEXT)
                
        return context


class MCPJSONLDMessage:
    """
    Base class for MCP messages using JSON-LD for semantic context
    """
    
    def __init__(
        self, 
        message_type: str,
        context_domains: List[str] = None,
        custom_context: Dict = None
    ):
        """
        Initialize a new MCP message with JSON-LD context
        
        Args:
            message_type: Type of MCP message
            context_domains: List of domain contexts to include
            custom_context: Additional custom context definitions
        """
        # Build the context
        domains = context_domains or ['core']
        if 'core' not in domains:
            domains.append('core')  # Always include core
            
        self.context = MCPContextRegistry.get_context(*domains)
        
        # Add any custom context
        if custom_context:
            self.context.update(custom_context)
            
        # Create the base message
        self.message = {
            "@context": self.context,
            "type": message_type,
            "id": f"mcp:{str(uuid.uuid4())}",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def add_content(self, content: Any) -> None:
        """Add content to the message"""
        self.message["content"] = content
        
    def add_metadata(self, metadata: Dict) -> None:
        """Add metadata to the message"""
        self.message["metadata"] = metadata
        
    def add_location(self, latitude: float, longitude: float, elevation: Optional[float] = None, zone: Optional[str] = None) -> None:
        """
        Add geospatial location to the message
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            elevation: Optional elevation in meters
            zone: Optional named zone identifier
        """
        location = {
            "type": "geo:Point",
            "latitude": latitude,
            "longitude": longitude
        }
        
        if elevation is not None:
            location["elevation"] = elevation
            
        if zone:
            location["zone"] = zone
            
        if "metadata" not in self.message:
            self.message["metadata"] = {}
            
        self.message["metadata"]["location"] = location
        
    def set_source(self, source: str) -> None:
        """Set the source of the message"""
        if "metadata" not in self.message:
            self.message["metadata"] = {}
            
        self.message["metadata"]["source"] = source
        
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for related messages"""
        self.message["correlationId"] = correlation_id
        
    def set_confidence(self, confidence: float) -> None:
        """Set confidence value (0.0-1.0)"""
        if "metadata" not in self.message:
            self.message["metadata"] = {}
            
        self.message["metadata"]["confidence"] = max(0.0, min(1.0, confidence))
        
    def to_dict(self) -> Dict:
        """Get the message as a dictionary"""
        return self.message
        
    def to_json(self, pretty: bool = False) -> str:
        """
        Serialize message to JSON
        
        Args:
            pretty: Whether to format with indentation
            
        Returns:
            JSON string representation of the message
        """
        if pretty:
            return json.dumps(self.message, indent=2)
        return json.dumps(self.message)
        
    @classmethod
    def from_json(cls, json_str: str) -> Dict:
        """
        Deserialize JSON to a message dictionary
        
        Args:
            json_str: JSON string to parse
            
        Returns:
            Parsed message dictionary
        """
        return json.loads(json_str)


class SensorReadingMessage(MCPJSONLDMessage):
    """
    Message representing a sensor reading with rich context
    """
    
    def __init__(
        self,
        sensor_id: str,
        value: Union[float, int, str, bool],
        unit: Optional[str] = None,
        accuracy: Optional[float] = None,
        sample_rate: Optional[float] = None
    ):
        """
        Create a new sensor reading message
        
        Args:
            sensor_id: Identifier of the sensor
            value: Reading value
            unit: Unit of measurement
            accuracy: Accuracy of reading (0.0-1.0)
            sample_rate: Sample rate in Hz
        """
        super().__init__(
            message_type="mcp:SensorReading",
            context_domains=['core', 'sensor']
        )
        
        reading = {
            "sensorId": sensor_id,
            "value": value
        }
        
        if unit:
            reading["unit"] = unit
            
        if accuracy is not None:
            reading["accuracy"] = accuracy
            
        if sample_rate is not None:
            reading["sampleRate"] = sample_rate
            
        self.add_content(reading)
        self.set_source(f"sensor:{sensor_id}")


class EnvironmentalEventMessage(MCPJSONLDMessage):
    """
    Message representing an environmental event
    """
    
    def __init__(
        self,
        event_type: str,
        event_data: Dict,
        category: Optional[str] = None,
        occurred_at: Optional[str] = None,
        detected_at: Optional[str] = None,
        confidence: Optional[float] = None
    ):
        """
        Create a new environmental event message
        
        Args:
            event_type: Type of event
            event_data: Event details
            category: Category of event
            occurred_at: When event occurred (ISO timestamp)
            detected_at: When event was detected (ISO timestamp)
            confidence: Confidence in event detection (0.0-1.0)
        """
        super().__init__(
            message_type="mcp:EnvironmentalEvent",
            context_domains=['core', 'event']
        )
        
        event_content = {
            "eventType": event_type,
            **event_data
        }
        
        if category:
            event_content["category"] = category
            
        if occurred_at:
            event_content["occurred"] = occurred_at
        
        if detected_at:
            event_content["detected"] = detected_at
        else:
            event_content["detected"] = datetime.utcnow().isoformat()
            
        self.add_content(event_content)
        
        if confidence is not None:
            self.set_confidence(confidence)


class ToolInvocationMessage(MCPJSONLDMessage):
    """
    Message representing a tool/action invocation
    """
    
    def __init__(
        self,
        tool_id: str,
        action: str,
        parameters: Dict = None
    ):
        """
        Create a new tool invocation message
        
        Args:
            tool_id: Identifier of the tool
            action: Action to perform
            parameters: Parameters for the action
        """
        super().__init__(
            message_type="mcp:ToolInvocation",
            context_domains=['core', 'tool']
        )
        
        tool_content = {
            "toolId": tool_id,
            "action": action
        }
        
        if parameters:
            tool_content["parameters"] = parameters
            
        self.add_content(tool_content)


# Example usage
def create_examples():
    # Example 1: Temperature sensor reading
    temp_reading = SensorReadingMessage(
        sensor_id="temp_sensor_1",
        value=22.5,
        unit="celsius",
        accuracy=0.95
    )
    temp_reading.add_location(37.7749, -122.4194, zone="building_1_floor_2")
    
    # Example 2: Motion detection event
    motion_event = EnvironmentalEventMessage(
        event_type="motion.detected",
        event_data={"object": "person", "count": 2},
        category="security",
        confidence=0.87
    )
    motion_event.add_location(37.7750, -122.4195, zone="entrance")
    motion_event.set_source("camera_system")
    
    # Example 3: Tool invocation to control a camera
    camera_control = ToolInvocationMessage(
        tool_id="camera_system",
        action="pan",
        parameters={"angle": 45, "speed": 0.5}
    )
    
    return {
        "temperature_reading": temp_reading.to_json(pretty=True),
        "motion_event": motion_event.to_json(pretty=True),
        "camera_control": camera_control.to_json(pretty=True)
    }


if __name__ == "__main__":
    examples = create_examples()
    
    for name, json_data in examples.items():
        print(f"\n--- {name} ---")
        print(json_data)