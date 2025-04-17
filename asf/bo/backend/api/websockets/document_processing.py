"""
WebSocket endpoints for document processing.
"""

import json
import asyncio
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import logging
from uuid import uuid4

from api.websockets.connection_manager import connection_manager

logger = logging.getLogger(__name__)

async def handle_document_processing_updates(websocket: WebSocket):
    """
    Handle WebSocket connections for document processing updates.
    
    Args:
        websocket: WebSocket connection
    """
    client_id = str(uuid4())
    await connection_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Wait for messages from the client
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "subscribe":
                    # Subscribe to a task
                    task_id = message.get("task_id")
                    if task_id:
                        await connection_manager.subscribe_to_task(client_id, task_id)
                        await websocket.send_json({
                            "type": "subscribed",
                            "task_id": task_id
                        })
                
                elif message_type == "unsubscribe":
                    # Unsubscribe from a task
                    task_id = message.get("task_id")
                    if task_id:
                        await connection_manager.unsubscribe_from_task(client_id, task_id)
                        await websocket.send_json({
                            "type": "unsubscribed",
                            "task_id": task_id
                        })
                
                elif message_type == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong"
                    })
                
                else:
                    # Unknown message type
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    })
            
            except json.JSONDecodeError:
                # Invalid JSON
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
            
            except Exception as e:
                # Other errors
                logger.error(f"Error processing message: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                })
    
    except WebSocketDisconnect:
        # Client disconnected
        await connection_manager.disconnect(client_id)
    
    except Exception as e:
        # Other errors
        logger.error(f"Error in WebSocket connection: {str(e)}")
        await connection_manager.disconnect(client_id)
