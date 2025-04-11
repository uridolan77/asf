import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
import uuid



class MCPLayer4Manager:
    """
    Main coordinator for MCP integration in Layer 4.
    Manages all external communication channels and internal routing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("MCPLayer4Manager")
        self.config = config
        
        self.event_hub = MCPEventHub()
        self.security_handler = MCPSecurityHandler(config.get("security", {}))
        self.auth_manager = MCPAuthorizationManager()
        self.source_manager = ExternalSourceManager(self.event_hub)
        self.adapter_factory = AdapterFactory()
        
        self.is_running = False
        self.initialized_sources = set()
        
        self.schema = MCPMessageSchema()
        
        self.cross_layer_handlers = {}
        
    async def initialize(self):
        self.logger.info("Shutting down MCP Layer 4 Manager")
        
        self.is_running = False
        
        await self.source_manager.shutdown()
        
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
        if message.get("metadata", {}).get("internal", False):
            return
            
        source_id = message.get("metadata", {}).get("source")
        if not source_id:
            self.logger.warning("Received message without source identifier")
            return
            
        is_valid, error = self.security_handler.validate_message(message, source_id)
        if not is_valid:
            self.logger.warning(f"Rejected message from {source_id}: {error}")
            return
            
        if not self.auth_manager.authorize_message(
            message, 
            source_id, 
            default_permission_mapper  # Use the default permission mapper
        ):
            self.logger.warning(f"Unauthorized message from {source_id}")
            return
            
        
    async def connect_external_source(
        self, 
        source_id: str, 
        source_config: Dict
    ) -> bool:
        try:
            success = await self.source_manager.connect_source(
                source_id,
                source_config,
                self.adapter_factory.create_adapter
            )
            
            if success:
                self.logger.info(f"Connected to external source: {source_id}")
                
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
        if not self.is_running:
            self.logger.error("Cannot send message: MCP Layer 4 Manager not running")
            return None
            
        if not self.schema.validate_message(message):
            self.logger.error("Invalid message format")
            return None
            
        secured_message = self.security_handler.secure_message(message, source_id)
        
        response = await self.source_manager.send_message(source_id, secured_message)
        return response
        
    def register_cross_layer_handler(
        self, 
        layer_id: str, 
        message_pattern: Dict, 
        handler: Callable[[Dict], Awaitable[Dict]]
    ):
        handler_id = str(uuid.uuid4())
        
        self.cross_layer_handlers[handler_id] = {
            "layer": layer_id,
            "pattern": message_pattern,
            "handler": handler
        }
        
        unsubscribe = self.event_hub.subscribe_pattern(
            message_pattern,
            lambda message: self._handle_cross_layer_message(handler_id, message)
        )
        
        return handler_id, unsubscribe
        
    async def _handle_cross_layer_message(self, handler_id: str, message: Dict):
        Request a resource from an external source or the appropriate handler
        Invoke a tool on an external source or the appropriate handler
        Publish an event to the event hub
        Get information about all connected sources
        Get message statistics