import asyncio
import json
import logging
from typing import Dict, List, Any, Callable, Awaitable, Optional, Pattern, Union
import re
import time
from collections import defaultdict


class MCPEventHub:
    """
    Central hub for managing MCP message routing between internal and external components.
    Implements the content-based publish/subscribe pattern for Layer 4.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("MCPEventHub")
        self.subscriptions = defaultdict(list)  # message_type -> list of subscribers
        self.pattern_subscriptions = []  # list of (pattern, callback) tuples
        self.message_stats = defaultdict(int)  # message_type -> count
        self.external_sources = {}  # source_id -> source_info
        self.running = False
        self.processing_tasks = set()
        
    async def start(self):
        """Start the event hub"""
        self.running = True
        self.logger.info("MCP Event Hub started")
        
    async def stop(self):
        """Stop the event hub and cleanup"""
        self.running = False
        
        # Wait for all processing tasks to complete
        if self.processing_tasks:
            self.logger.info(f"Waiting for {len(self.processing_tasks)} tasks to complete")
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
            
        self.logger.info("MCP Event Hub stopped")
        
    async def publish(self, message: Dict) -> None:
        """
        Publish a message to all interested subscribers
        """
        if not self.running:
            self.logger.warning("Attempted to publish message while hub is stopped")
            return
            
        message_type = message.get("type")
        if not message_type:
            self.logger.error("Cannot publish message without type")
            return
            
        # Track message stats
        self.message_stats[message_type] += 1
        
        # Create task for message processing to avoid blocking
        task = asyncio.create_task(self._process_message(message))
        self.processing_tasks.add(task)
        task.add_done_callback(self.processing_tasks.remove)
        
    async def _process_message(self, message: Dict) -> None:
        """Process a single message, notifying all subscribers"""
        message_type = message.get("type")
        
        # Notify type-based subscribers
        subscribers = self.subscriptions.get(message_type, [])
        subscriber_tasks = []
        
        for callback in subscribers:
            try:
                subscriber_tasks.append(asyncio.create_task(callback(message)))
            except Exception as e:
                self.logger.error(f"Error notifying subscriber: {str(e)}")
                
        # Notify pattern-based subscribers
        for pattern, callback in self.pattern_subscriptions:
            if self._message_matches_pattern(message, pattern):
                try:
                    subscriber_tasks.append(asyncio.create_task(callback(message)))
                except Exception as e:
                    self.logger.error(f"Error notifying pattern subscriber: {str(e)}")
        
        # Wait for all subscriber notifications to complete
        if subscriber_tasks:
            await asyncio.gather(*subscriber_tasks, return_exceptions=True)
            
    def subscribe(
        self, 
        message_type: str, 
        callback: Callable[[Dict], Awaitable[None]]
    ) -> Callable[[], None]:
        """
        Subscribe to messages of a specific type
        Returns a function to unsubscribe
        """
        self.subscriptions[message_type].append(callback)
        
        def unsubscribe():
            if callback in self.subscriptions[message_type]:
                self.subscriptions[message_type].remove(callback)
                
        return unsubscribe
        
    def subscribe_pattern(
        self, 
        pattern: Dict[str, Any], 
        callback: Callable[[Dict], Awaitable[None]]
    ) -> Callable[[], None]:
        """
        Subscribe to messages matching a pattern of attributes
        Pattern is a dict where keys are attribute paths (dot notation)
        and values are the expected values or regex patterns
        
        Example: {"eventType": "sensor.temperature", "metadata.source": re.compile("zone_1.*")}
        
        Returns a function to unsubscribe
        """
        subscription = (pattern, callback)
        self.pattern_subscriptions.append(subscription)
        
        def unsubscribe():
            if subscription in self.pattern_subscriptions:
                self.pattern_subscriptions.remove(subscription)
                
        return unsubscribe
        
    def _message_matches_pattern(self, message: Dict, pattern: Dict) -> bool:
        """Check if a message matches a subscription pattern"""
        for attr_path, expected_value in pattern.items():
            # Navigate through nested attributes using dot notation
            parts = attr_path.split('.')
            value = message
            
            try:
                for part in parts:
                    if not isinstance(value, dict) or part not in value:
                        return False
                    value = value[part]
                    
                # Check if value matches expected value
                if isinstance(expected_value, Pattern):  # regex pattern
                    if not isinstance(value, str) or not expected_value.search(value):
                        return False
                elif value != expected_value:  # exact match
                    return False
            except (KeyError, TypeError):
                return False
                
        return True
        
    def register_external_source(
        self, 
        source_id: str, 
        source_info: Dict
    ) -> None:
        """Register information about an external source"""
        self.external_sources[source_id] = {
            **source_info,
            "registered_at": time.time()
        }
        self.logger.info(f"Registered external source: {source_id}")
        
    def unregister_external_source(self, source_id: str) -> None:
        """Unregister an external source"""
        if source_id in self.external_sources:
            del self.external_sources[source_id]
            self.logger.info(f"Unregistered external source: {source_id}")
            
    def get_external_source(self, source_id: str) -> Optional[Dict]:
        """Get information about an external source"""
        return self.external_sources.get(source_id)
        
    def get_all_external_sources(self) -> Dict[str, Dict]:
        """Get information about all registered external sources"""
        return self.external_sources.copy()
        
    def get_stats(self) -> Dict:
        """Get message statistics"""
        return {
            "message_counts": dict(self.message_stats),
            "subscriber_counts": {
                msg_type: len(subscribers) 
                for msg_type, subscribers in self.subscriptions.items()
            },
            "pattern_subscribers": len(self.pattern_subscriptions),
            "external_sources": len(self.external_sources)
        }


class ExternalSourceManager:
    """
    Manages connections to external MCP sources and handles automatic reconnection
    and message forwarding to the event hub
    """
    
    def __init__(self, event_hub: MCPEventHub):
        self.event_hub = event_hub
        self.logger = logging.getLogger("ExternalSourceManager")
        self.adapters = {}  # source_id -> adapter instance
        self.connection_tasks = {}  # source_id -> asyncio task
        self.reconnect_attempts = defaultdict(int)  # source_id -> attempt count
        self.max_reconnect_attempts = 5
        
    async def connect_source(
        self, 
        source_id: str, 
        source_config: Dict,
        adapter_factory: Callable
    ) -> bool:
        """
        Connect to an external source using the appropriate adapter
        """
        # Create adapter for this source
        try:
            adapter_type = source_config.get("type", "rest")
            adapter = adapter_factory(adapter_type, source_config)
            
            # Initialize connection
            if hasattr(adapter, "connect"):
                await adapter.connect()
            elif hasattr(adapter, "initialize"):
                await adapter.initialize()
                
            # Store adapter
            self.adapters[source_id] = adapter
            
            # Register with event hub
            self.event_hub.register_external_source(source_id, {
                "type": adapter_type,
                "config": source_config,
                "status": "connected"
            })
            
            # If it's a streaming source, start message receiver
            if hasattr(adapter, "receive_message"):
                self._start_message_receiver(source_id, adapter)
                
            self.logger.info(f"Connected to external source: {source_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to source {source_id}: {str(e)}")
            return False
            
    def _start_message_receiver(self, source_id: str, adapter):
        """Start a background task to receive messages from a streaming source"""
        task = asyncio.create_task(self._receive_messages(source_id, adapter))
        self.connection_tasks[source_id] = task
        
    async def _receive_messages(self, source_id: str, adapter):
        """Continuously receive messages from a streaming source"""
        self.reconnect_attempts[source_id] = 0
        
        while True:
            try:
                # Receive message from the adapter
                message = await adapter.receive_message()
                
                # Add source identifier to message metadata if not present
                if "metadata" not in message:
                    message["metadata"] = {}
                if "source" not in message["metadata"]:
                    message["metadata"]["source"] = source_id
                    
                # Publish to the event hub
                await self.event_hub.publish(message)
                
                # Reset reconnect counter on successful message
                self.reconnect_attempts[source_id] = 0
                
            except ConnectionError:
                # Handle connection loss
                await self._handle_connection_loss(source_id, adapter)
                
            except Exception as e:
                self.logger.error(f"Error receiving message from {source_id}: {str(e)}")
                
    async def _handle_connection_loss(self, source_id: str, adapter):
        """Handle connection loss with exponential backoff retry"""
        self.reconnect_attempts[source_id] += 1
        attempt = self.reconnect_attempts[source_id]
        
        if attempt > self.max_reconnect_attempts:
            self.logger.error(f"Max reconnection attempts reached for {source_id}, giving up")
            return
            
        # Exponential backoff: 1s, 2s, 4s, 8s, 16s
        backoff = min(2 ** (attempt - 1), 16)
        self.logger.info(f"Connection lost to {source_id}, retry in {backoff}s (attempt {attempt})")
        
        await asyncio.sleep(backoff)
        
        try:
            if hasattr(adapter, "connect"):
                await adapter.connect()
                self.logger.info(f"Reconnected to {source_id}")
        except Exception as e:
            self.logger.error(f"Failed to reconnect to {source_id}: {str(e)}")
            
    async def disconnect_source(self, source_id: str) -> bool:
        """Disconnect from an external source"""
        if source_id not in self.adapters:
            return False
            
        # Cancel any running tasks
        if source_id in self.connection_tasks:
            self.connection_tasks[source_id].cancel()
            del self.connection_tasks[source_id]
            
        # Close adapter connection
        adapter = self.adapters[source_id]
        try:
            if hasattr(adapter, "close"):
                await adapter.close()
        except Exception as e:
            self.logger.error(f"Error closing adapter for {source_id}: {str(e)}")
            
        # Remove adapter
        del self.adapters[source_id]
        
        # Unregister from event hub
        self.event_hub.unregister_external_source(source_id)
        
        self.logger.info(f"Disconnected from external source: {source_id}")
        return True
        
    async def send_message(self, source_id: str, message: Dict) -> Optional[Dict]:
        """Send a message to an external source and return the response"""
        if source_id not in self.adapters:
            self.logger.error(f"Unknown source: {source_id}")
            return None
            
        adapter = self.adapters[source_id]
        
        try:
            if hasattr(adapter, "execute_request"):
                # For request/response adapters like REST
                return await adapter.execute_request(message)
            elif hasattr(adapter, "send_message"):
                # For streaming adapters like WebSocket
                await adapter.send_message(message)
                return {"status": "sent"}
            else:
                self.logger.error(f"Adapter for {source_id} doesn't support sending messages")
                return None
        except Exception as e:
            self.logger.error(f"Error sending message to {source_id}: {str(e)}")
            return None
            
    async def shutdown(self):
        """Shutdown all connections"""
        for source_id in list(self.adapters.keys()):
            await self.disconnect_source(source_id)