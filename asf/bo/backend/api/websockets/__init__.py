"""
WebSocket endpoints for the BO backend.
"""

from fastapi import APIRouter, WebSocket

from api.websockets.document_processing import handle_document_processing_updates

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
