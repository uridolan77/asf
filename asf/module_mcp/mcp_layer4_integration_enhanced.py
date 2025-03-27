import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
import uuid

# Import the components we've defined in previous artifacts
# In a real implementation, you would use proper imports


class MCPLayer4Manager:
    """
    Main coordinator for MCP integration in Layer 4.
    Manages all external communication channels and internal routing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("MCPLayer4Manager")
        self.config = config
        
        # Initialize components
        self.event_hub = MCPEventHub()
        self.security_handler = MCPSecurityHandler(config.get("security", {}))
        self.auth_manager = MCPAuthorizationManager()
        self.source_manager = ExternalSourceManager(self.event_hub)
        self.adapter_factory = AdapterFactory()
        
        # Initialize connection state
        self.is_running = False
        self.initialized_sources = set()
        
        # Set up schema validator
        self.schema = MCPMessageSchema()
        
        # Set up internal subscription for Layer 4 -> Layer 5/6 communication
        self.cross_layer_handlers = {}
        
    async def initialize(self):
        """Initialize the MCP Layer 4 manager and all components"""
        self.logger.info("Initializing MCP Layer 4 Manager")
        
        # Start event hub
        await self.event_hub.start()
        
        # Pre-configure roles and permissions if specified in config
        self._configure_security()
        
        # Connect to pre-configured external sources
        await self._connect_configured_sources()
        
        # Set up internal message handling
        self._setup_internal_handlers()
        
        self.is_running = True
        self.logger.info("MCP Layer 4 Manager initialized")
        
    async def shutdown(self):
        """Shutdown all MCP connections and components"""
        self.logger.info("Shutting down MCP Layer 4 Manager")
        
        self.is_running = False
        
        # Disconnect all external sources
        await self.source_manager.shutdown()
        
        # Stop event hub
        await self.event_hub.stop()
        
        self.logger.info("MCP Layer 4 Manager shutdown complete")
        
    def _configure_security(self):
        """Configure security roles and permissions"""
        security_config = self.config.get("security", {})
        
        # Set up roles and permissions
        for role_config in security_config.get("roles", []):
            role_name = role_config.get("name")
            permissions = role_config.get("permissions", [])
            
            if role_name:
                self.auth_manager.add_role(role_name, permissions)
                
        # Assign roles to sources
        for source_role in security_config.get("source_roles", []):
            source_id = source_role.get("source_id")
            role_name = source_role.get("role")
            
            if source_id and role_name:
                self.auth_manager.assign_role_to_source(source_id, role_name)
                
    async def _connect_configured_sources(self):
        """Connect to all sources defined in the config"""
        sources_config = self.config.get("sources", {})
        
        for source_id, source_config in sources_config.items():
            # Skip disabled sources
            if not source_config.get("enabled", True):
                continue
                
            try:
                success = await self.connect_external_source(source_id, source_config)
                if success:
                    self.initialized_sources.add(source_id)
                    self.logger.info(f"Connected to pre-configured source: {source_id}")
                else:
                    self.logger.warning(f"Failed to connect to pre-configured source: {source_id}")
            except Exception as e:
                self.logger.error(f"Error connecting to source {source_id}: {str(e)}")
                
    def _setup_internal_handlers(self):
        """Set up handlers for internal message routing"""
        # Subscribe to all incoming messages to perform security validation
        self.event_hub.subscribe_pattern(
            {}, 
            self._validate_incoming_messages
        )
        
    async def _validate_incoming_messages(self, message: Dict):
        """Validate all incoming messages for security and authorization"""
        # Skip internal messages
        if message.get("metadata", {}).get("internal", False):
            return
            
        # Get the source from message metadata
        source_id = message.get("metadata", {}).get("source")
        if not source_id:
            self.logger.warning("Received message without source identifier")
            return
            
        # Validate message security
        is_valid, error = self.security_handler.validate_message(message, source_id)
        if not is_valid:
            self.logger.warning(f"Rejected message from {source_id}: {error}")
            # Could publish a security violation event here
            return
            
        # Authorize the message
        if not self.auth_manager.authorize_message(
            message, 
            source_id, 
            default_permission_mapper  # Use the default permission mapper
        ):
            self.logger.warning(f"Unauthorized message from {source_id}")
            # Could publish an authorization violation event here
            return
            
        # If we get here, the message passed security and authorization checks
        # No need to do anything else, as subscribers are already notified
        
    async def connect_external_source(
        self, 
        source_id: str, 
        source_config: Dict
    ) -> bool:
        """
        Connect to an external source using MCP
        """
        try:
            success = await self.source_manager.connect_source(
                source_id,
                source_config,
                self.adapter_factory.create_adapter
            )
            
            if success:
                self.logger.info(f"Connected to external source: {source_id}")
                
                # If this source has pre-defined roles, assign them
                if "roles" in source_config:
                    for role in source_config["roles"]:
                        self.auth_manager.assign_role_to_source(source_id, role)
                        
                return True
            else:
                self.logger.error(f"Failed to connect to external source: {source_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to source {source_id}: {str(e)}")
            return False
            
    async def disconnect_external_source(self, source_id: str) -> bool:
        """
        Disconnect from an external source
        """
        success = await self.source_manager.disconnect_source(source_id)
        
        if success:
            self.logger.info(f"Disconnected from external source: {source_id}")
            if source_id in self.initialized_sources:
                self.initialized_sources.remove(source_id)
        else:
            self.logger.warning(f"Failed to disconnect from external source: {source_id}")
            
        return success
        
    async def send_to_external_source(
        self, 
        source_id: str, 
        message: Dict
    ) -> Optional[Dict]:
        """
        Send a message to an external source and return the response
        """
        if not self.is_running:
            self.logger.error("Cannot send message: MCP Layer 4 Manager not running")
            return None
            
        # Validate message format
        if not self.schema.validate_message(message):
            self.logger.error("Invalid message format")
            return None
            
        # Add security information
        secured_message = self.security_handler.secure_message(message, source_id)
        
        # Send via source manager
        response = await self.source_manager.send_message(source_id, secured_message)
        return response
        
    def register_cross_layer_handler(
        self, 
        layer_id: str, 
        message_pattern: Dict, 
        handler: Callable[[Dict], Awaitable[Dict]]
    ):
        """
        Register a handler for messages from Layer 4 to other layers
        """
        handler_id = str(uuid.uuid4())
        
        # Store the handler
        self.cross_layer_handlers[handler_id] = {
            "layer": layer_id,
            "pattern": message_pattern,
            "handler": handler
        }
        
        # Subscribe to matching messages
        unsubscribe = self.event_hub.subscribe_pattern(
            message_pattern,
            lambda message: self._handle_cross_layer_message(handler_id, message)
        )
        
        return handler_id, unsubscribe
        
    async def _handle_cross_layer_message(self, handler_id: str, message: Dict):
        """Handle messages that need to be passed to other layers"""
        handler_info = self.cross_layer_handlers.get(handler_id)
        if not handler_info:
            return
            
        try:
            # Call the registered handler
            response = await handler_info["handler"](message)
            
            # If the handler returned a response, publish it
            if response:
                # Mark as an internal message
                if "metadata" not in response:
                    response["metadata"] = {}
                response["metadata"]["internal"] = True
                
                # Add correlation ID if the original message had one
                if "correlationId" in message:
                    response["correlationId"] = message["correlationId"]
                    
                # Publish the response
                await self.event_hub.publish(response)
                
        except Exception as e:
            self.logger.error(f"Error in cross-layer handler: {str(e)}")
            
    async def request_resource(
        self, 
        resource_type: str, 
        parameters: Dict = None, 
        source_id: Optional[str] = None
    ) -> Dict:
        """
        Request a resource from an external source or the appropriate handler
        """
        # Create a resource request message
        request = MCPMessageSchema.create_message(
            MessageType.RESOURCE_REQUEST,
            resourceType=resource_type,
            parameters=parameters or {}
        )
        
        if source_id:
            # Directed request to a specific source
            return await self.send_to_external_source(source_id, request)
        else:
            # Broadcast request to any handler that can fulfill it
            # Create a future to receive the response
            future = asyncio.Future()
            
            # Create a correlation ID for this request
            correlation_id = request["id"]
            
            # Define a handler for the response
            async def response_handler(message):
                if message.get("correlationId") == correlation_id:
                    if not future.done():
                        future.set_result(message)
            
            # Subscribe to responses with this correlation ID
            unsubscribe = self.event_hub.subscribe_pattern(
                {"correlationId": correlation_id},
                response_handler
            )
            
            # Publish the request
            await self.event_hub.publish(request)
            
            try:
                # Wait for the response with timeout
                response = await asyncio.wait_for(future, 30.0)  # 30 second timeout
                return response
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout waiting for response to resource request: {resource_type}")
                return {
                    "type": "Error",
                    "code": "timeout",
                    "message": f"Timeout waiting for resource: {resource_type}"
                }
            finally:
                # Always unsubscribe to prevent memory leaks
                unsubscribe()
                
    async def invoke_tool(
        self, 
        tool: str, 
        parameters: Dict,
        source_id: Optional[str] = None
    ) -> Dict:
        """
        Invoke a tool on an external source or the appropriate handler
        """
        # Create a tool invocation message
        request = MCPMessageSchema.create_message(
            MessageType.TOOL_INVOKE,
            tool=tool,
            parameters=parameters
        )
        
        # Similar pattern to request_resource
        if source_id:
            return await self.send_to_external_source(source_id, request)
        else:
            future = asyncio.Future()
            correlation_id = request["id"]
            
            async def response_handler(message):
                if message.get("correlationId") == correlation_id:
                    if not future.done():
                        future.set_result(message)
            
            unsubscribe = self.event_hub.subscribe_pattern(
                {"correlationId": correlation_id},
                response_handler
            )
            
            await self.event_hub.publish(request)
            
            try:
                response = await asyncio.wait_for(future, 60.0)  # 60 second timeout for tools
                return response
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout waiting for tool invocation result: {tool}")
                return {
                    "type": "Error",
                    "code": "timeout",
                    "message": f"Timeout invoking tool: {tool}"
                }
            finally:
                unsubscribe()
                
    async def publish_event(
        self, 
        event_type: str, 
        content: Any,
        priority: Union[str, EventPriority] = EventPriority.NORMAL,
        metadata: Dict = None
    ):
        """
        Publish an event to the event hub
        """
        if isinstance(priority, EventPriority):
            priority = priority.value
            
        event = MCPMessageSchema.create_message(
            MessageType.EVENT,
            content,
            eventType=event_type,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Mark as internal event
        if "metadata" not in event:
            event["metadata"] = {}
        event["metadata"]["internal"] = True
        
        await self.event_hub.publish(event)
        
    def get_connected_sources(self) -> Dict[str, Dict]:
        """
        Get information about all connected sources
        """
        return self.event_hub.get_all_external_sources()
        
    def get_message_stats(self) -> Dict[str, Any]:
        """
        Get message statistics
        """
        return self.event_hub.get_stats()


# Example of initialization and usage
async def example_main():
    # Configuration
    config = {
        "security": {
            "shared_secrets": {
                "camera_system": "secret1",
                "weather_api": "secret2"
            },
            "jwt_secret": "jwt_secret_key",
            "roles": [
                {"name": "sensor", "permissions": ["resources.read.*", "events.publish.sensor.*"]},
                {"name": "actuator", "permissions": ["tools.invoke.*"]}
            ]
        },
        "sources": {
            "camera_system": {
                "type": "websocket",
                "ws_url": "wss://camera-system.example.com/mcp",
                "roles": ["sensor"]
            },
            "weather_api": {
                "type": "rest",
                "base_url": "https://weather-api.example.com/v1",
                "roles": ["sensor"]
            }
        }
    }
    
    # Create and initialize the manager
    manager = MCPLayer4Manager(config)
    await manager.initialize()
    
    # Register a cross-layer handler for weather events
    async def handle_weather_update(message):
        print(f"Layer 5 received weather update: {message}")
        # Process the data and return a response if needed
        return {
            "type": "Event",
            "eventType": "weather.processed",
            "content": {"processed": True, "original_id": message["id"]}
        }
    
    manager.register_cross_layer_handler(
        "layer5",
        {"type": "Event", "eventType": "weather.update"},
        handle_weather_update
    )
    
    # Request weather data
    response = await manager.request_resource(
        "weather",
        {"location": "San Francisco", "units": "metric"},
        "weather_api"
    )
    
    print(f"Weather data: {response}")
    
    # Publish an internal event
    await manager.publish_event(
        "system.status",
        {"status": "operational", "uptime": 3600},
        EventPriority.NORMAL
    )
    
    # Wait for a while to allow async processes to run
    await asyncio.sleep(10)
    
    # Shutdown
    await manager.shutdown()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(example_main())