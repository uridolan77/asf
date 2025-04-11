import json
import asyncio
from abc import ABC, abstractmethod
import aiohttp
from typing import Dict, Any, Optional


class MCPProtocolAdapter(ABC):
    """Base adapter for translating between MCP and external protocols"""
    
    @abstractmethod
    async def to_mcp_message(self, external_data: Any) -> Dict:
        """Convert external protocol data to MCP message format"""
        pass
        
    @abstractmethod
    async def from_mcp_message(self, mcp_message: Dict) -> Any:
    
    def __init__(self, config: Dict):
        self.base_url = config.get('base_url')
        self.headers = config.get('headers', {})
        self.session = None
        
    async def initialize(self):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def to_mcp_message(self, response_data: Dict) -> Dict:
        if mcp_message["type"] == "Tool.invoke":
            method = mcp_message.get("method", "POST")
            endpoint = mcp_message.get("endpoint", "")
            body = mcp_message.get("parameters", {})
            
            return {
                "method": method,
                "url": f"{self.base_url}/{endpoint}",
                "json": body
            }
        else:
            query_params = mcp_message.get("parameters", {})
            endpoint = mcp_message.get("endpoint", "")
            
            return {
                "method": "GET",
                "url": f"{self.base_url}/{endpoint}",
                "params": query_params
            }
    
    async def execute_request(self, mcp_message: Dict) -> Dict:
        from datetime import datetime
        return datetime.utcnow().isoformat()


class WebSocketAdapter(MCPProtocolAdapter):
    """Adapter for WebSocket integration with MCP"""
    
    def __init__(self, config: Dict):
        self.ws_url = config.get('ws_url')
        self.headers = config.get('headers', {})
        self.ws_connection = None
        self.message_queue = asyncio.Queue()
        
    async def connect(self):
        """Establish WebSocket connection"""
        if not self.ws_connection:
            self.ws_connection = await aiohttp.ClientSession().ws_connect(
                self.ws_url, headers=self.headers
            )
            asyncio.create_task(self._message_receiver())
            
    async def _message_receiver(self):
        event_type = ws_data.get("event", "update")
        
        if event_type == "update":
            mcp_type = "Resource.update"
        elif event_type == "error":
            mcp_type = "Error"
        else:
            mcp_type = "Event"
            
        return {
            "type": mcp_type,
            "content": ws_data.get("data", {}),
            "metadata": {
                "source": "websocket",
                "event": event_type,
                "timestamp": self._get_current_timestamp()
            }
        }
    
    async def from_mcp_message(self, mcp_message: Dict) -> Dict:
        if not self.ws_connection:
            await self.connect()
            
        ws_message = await self.from_mcp_message(mcp_message)
        await self.ws_connection.send_json(ws_message)
    
    async def receive_message(self) -> Dict:
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
    
    def _map_mcp_to_ws_action(self, mcp_type: str) -> str:
        """Map MCP message type to WebSocket action"""
        mapping = {
            "Tool.invoke": "call",
            "Resource.request": "get",
            "Resource.update": "update",
            "Event": "event"
        }
        return mapping.get(mcp_type, "message")
    
    def _get_current_timestamp(self):
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.utcnow().isoformat()


class DatabaseAdapter(MCPProtocolAdapter):
    """Adapter for database integration with MCP"""
    
    def __init__(self, config: Dict):
        self.db_config = config
        self.connection = None
        
    async def initialize(self):
        """Initialize database connection"""
        pass
        
    async def to_mcp_message(self, db_result: Any) -> Dict:
        operation = None
        
        if mcp_message["type"] == "Resource.request":
            operation = {
                "type": "query",
                "collection": mcp_message.get("collection", ""),
                "query": mcp_message.get("parameters", {})
            }
        elif mcp_message["type"] == "Tool.invoke":
            method = mcp_message.get("method", "")
            if method == "insert":
                operation = {
                    "type": "insert",
                    "collection": mcp_message.get("collection", ""),
                    "data": mcp_message.get("parameters", {})
                }
            elif method == "update":
                operation = {
                    "type": "update",
                    "collection": mcp_message.get("collection", ""),
                    "query": mcp_message.get("query", {}),
                    "update": mcp_message.get("parameters", {})
                }
            elif method == "delete":
                operation = {
                    "type": "delete",
                    "collection": mcp_message.get("collection", ""),
                    "query": mcp_message.get("parameters", {})
                }
                
        return operation
    
    def _get_current_timestamp(self):
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.utcnow().isoformat()


# Factory to create appropriate adapter
class AdapterFactory:
    """Factory for creating protocol adapters"""
    
    @staticmethod
    def create_adapter(adapter_type: str, config: Dict) -> MCPProtocolAdapter:
        """Create and return adapter instance based on type"""
        if adapter_type == "rest":
            return RESTApiAdapter(config)
        elif adapter_type == "websocket":
            return WebSocketAdapter(config)
        elif adapter_type == "database":
            return DatabaseAdapter(config)
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")