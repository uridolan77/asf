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
        self.running = False
        
        if self.processing_tasks:
            self.logger.info(f"Waiting for {len(self.processing_tasks)} tasks to complete")
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
            
        self.logger.info("MCP Event Hub stopped")
        
    async def publish(self, message: Dict) -> None:
        if not self.running:
            self.logger.warning("Attempted to publish message while hub is stopped")
            return
            
        message_type = message.get("type")
        if not message_type:
            self.logger.error("Cannot publish message without type")
            return
            
        self.message_stats[message_type] += 1
        
        task = asyncio.create_task(self._process_message(message))
        self.processing_tasks.add(task)
        task.add_done_callback(self.processing_tasks.remove)
        
    async def _process_message(self, message: Dict) -> None:
        Subscribe to messages of a specific type
        Returns a function to unsubscribe
        Subscribe to messages matching a pattern of attributes
        Pattern is a dict where keys are attribute paths (dot notation)
        and values are the expected values or regex patterns
        
        Example: {"eventType": "sensor.temperature", "metadata.source": re.compile("zone_1.*")}
        
        Returns a function to unsubscribe
        for attr_path, expected_value in pattern.items():
            parts = attr_path.split('.')
            value = message
            
            try:
                for part in parts:
                    if not isinstance(value, dict) or part not in value:
                        return False
                    value = value[part]
                    
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
        Connect to an external source using the appropriate adapter
        task = asyncio.create_task(self._receive_messages(source_id, adapter))
        self.connection_tasks[source_id] = task
        
    async def _receive_messages(self, source_id: str, adapter):
        self.reconnect_attempts[source_id] += 1
        attempt = self.reconnect_attempts[source_id]
        
        if attempt > self.max_reconnect_attempts:
            self.logger.error(f"Max reconnection attempts reached for {source_id}, giving up")
            return
            
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
        if source_id not in self.adapters:
            self.logger.error(f"Unknown source: {source_id}")
            return None
            
        adapter = self.adapters[source_id]
        
        try:
            if hasattr(adapter, "execute_request"):
                return await adapter.execute_request(message)
            elif hasattr(adapter, "send_message"):
                await adapter.send_message(message)
                return {"status": "sent"}
            else:
                self.logger.error(f"Adapter for {source_id} doesn't support sending messages")
                return None
        except Exception as e:
            self.logger.error(f"Error sending message to {source_id}: {str(e)}")
            return None
            
    async def shutdown(self):