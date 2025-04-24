"""
WebSocket endpoints for the BO backend.
"""

import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from api.websockets.document_processing import handle_document_processing_updates
from api.websockets.mcp import handle_mcp_websocket
from api.auth.dependencies import get_current_user_ws

router = APIRouter(tags=["websockets"])

@router.websocket("/ws/document-processing")
async def websocket_document_processing(websocket: WebSocket):
    """
    WebSocket endpoint for document processing updates.

    This endpoint allows clients to subscribe to real-time document processing updates.

    Args:
        websocket: WebSocket connection
    """
    await handle_document_processing_updates(websocket)

@router.websocket("/ws/mcp/{client_id}")
async def mcp_websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for MCP provider status updates.

    Args:
        websocket: The WebSocket connection
        client_id: Unique client identifier
    """
    await handle_mcp_websocket(websocket, client_id)

@router.websocket("/ws/mcp")
async def mcp_websocket_endpoint_auto_id(websocket: WebSocket):
    """
    WebSocket endpoint for MCP provider status updates with auto-generated client ID.

    Args:
        websocket: The WebSocket connection
    """
    client_id = str(uuid.uuid4())
    await handle_mcp_websocket(websocket, client_id)
