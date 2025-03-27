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
        """Convert MCP message to external protocol format"""
        pass


class RESTApiAdapter(MCPProtocolAdapter):
    """Adapter for REST API integration with MCP"""
    
    def __init__(self, config: Dict):
        self.base_url = config.get('base_url')
        self.headers = config.get('headers', {})
        self.session = None
        
    async def initialize(self):
        """Initialize HTTP session"""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)
            
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def to_mcp_message(self, response_data: Dict) -> Dict:
        """Convert REST API response to MCP Resource message"""
        return {
            "type": "Resource",
            "content": response_data,
            "metadata": {
                "source": "rest_api",
                "endpoint": self.base_url,
                "timestamp": self._get_current_timestamp()
            }
        }
    
    async def from_mcp_message(self, mcp_message: Dict) -> Dict:
        """Convert MCP request to REST API format"""
        # Extract query parameters or body from MCP message
        if mcp_message["type"] == "Tool.invoke":
            # Handle tool invocation (POST, PUT, etc.)
            method = mcp_message.get("method", "POST")
            endpoint = mcp_message.get("endpoint", "")
            body = mcp_message.get("parameters", {})
            
            return {
                "method": method,
                "url": f"{self.base_url}/{endpoint}",
                "json": body
            }
        else:
            # Handle resource request (GET)
            query_params = mcp_message.get("parameters", {})
            endpoint = mcp_message.get("endpoint", "")
            
            return {
                "method": "GET",
                "url": f"{self.base_url}/{endpoint}",
                "params": query_params
            }
    
    async def execute_request(self, mcp_message: Dict) -> Dict:
        """Execute a REST request based on MCP message and return MCP response"""
        await self.initialize()
        
        # Convert MCP to REST format
        request_data = await self.from_mcp_message(mcp_message)
        
        # Execute the request
        method = request_data["method"].lower()
        request_func = getattr(self.session, method)
        
        async with request_func(
            request_data["url"],
            params=request_data.get("params"),
            json=request_data.get("json")
        ) as response:
            response_data = await response.json()
            
            # Convert response to MCP format
            return await self.to_mcp_message(response_data)
    
    def _get_current_timestamp(self):
        """Get current timestamp in ISO format"""
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
            # Start background task to receive messages
            asyncio.create_task(self._message_receiver())
            
    async def _message_receiver(self):
        """Background task to receive WebSocket messages"""
        try:
            async for msg in self.ws_connection:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    mcp_message = await self.to_mcp_message(data)
                    await self.message_queue.put(mcp_message)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        except Exception as e:
            print(f"WebSocket error: {str(e)}")
            # Put error in queue
            await self.message_queue.put({"type": "Error", "content": str(e)})
    
    async def to_mcp_message(self, ws_data: Dict) -> Dict:
        """Convert WebSocket message to MCP format"""
        # Determine MCP message type based on WebSocket event
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
        """Convert MCP message to WebSocket format"""
        # Convert to format expected by WebSocket server
        ws_message = {
            "action": self._map_mcp_to_ws_action(mcp_message["type"]),
            "data": mcp_message.get("content", {})
        }
        
        # Add any additional metadata needed by WebSocket server
        if "id" in mcp_message:
            ws_message["id"] = mcp_message["id"]
            
        return ws_message
    
    async def send_message(self, mcp_message: Dict) -> None:
        """Send MCP message through WebSocket"""
        if not self.ws_connection:
            await self.connect()
            
        ws_message = await self.from_mcp_message(mcp_message)
        await self.ws_connection.send_json(ws_message)
    
    async def receive_message(self) -> Dict:
        """Receive next MCP message from WebSocket"""
        return await self.message_queue.get()
    
    async def close(self):
        """Close WebSocket connection"""
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
        # This is a placeholder - would use appropriate async DB driver
        # like asyncpg for PostgreSQL or motor for MongoDB
        pass
        
    async def to_mcp_message(self, db_result: Any) -> Dict:
        """Convert database query result to MCP message"""
        if isinstance(db_result, list):
            # Collection of items
            return {
                "type": "Resource.collection",
                "content": db_result,
                "metadata": {
                    "source": "database",
                    "count": len(db_result),
                    "timestamp": self._get_current_timestamp()
                }
            }
        else:
            # Single item
            return {
                "type": "Resource",
                "content": db_result,
                "metadata": {
                    "source": "database",
                    "timestamp": self._get_current_timestamp()
                }
            }
    
    async def from_mcp_message(self, mcp_message: Dict) -> Dict:
        """Convert MCP message to database query"""
        operation = None
        
        if mcp_message["type"] == "Resource.request":
            # Read operation
            operation = {
                "type": "query",
                "collection": mcp_message.get("collection", ""),
                "query": mcp_message.get("parameters", {})
            }
        elif mcp_message["type"] == "Tool.invoke":
            # Write operation
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