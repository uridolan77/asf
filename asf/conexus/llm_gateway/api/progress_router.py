"""
REST API Router for progress tracking in the Conexus LLM Gateway.

This module provides FastAPI routes for tracking the progress of long-running operations.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends

from asf.conexus.llm_gateway.progress.tracker import get_progress_tracker, ProgressStatus
from asf.conexus.llm_gateway.core.client import get_client

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/llm/gateway/operations", tags=["llm-gateway-operations"])


@router.get("/", response_model=List[Dict[str, Any]])
async def get_operations(
    type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """
    Get a list of operations, filtered by type and status.
    
    Args:
        type: Filter by operation type
        status: Filter by operation status
        limit: Maximum number of results
    """
    tracker = get_progress_tracker()
    
    if status == "active":
        operations = await tracker.get_active_operations(
            operation_type=type,
            limit=limit
        )
    elif status == "completed":
        operations = await tracker.get_recent_completed_operations(
            operation_type=type, 
            limit=limit
        )
    else:
        # Get both active and completed
        active = await tracker.get_active_operations(
            operation_type=type,
            limit=limit
        )
        
        remaining = max(0, limit - len(active))
        completed = []
        if remaining > 0:
            completed = await tracker.get_recent_completed_operations(
                operation_type=type,
                limit=remaining
            )
            
        operations = active + completed
        
    return operations


@router.get("/{operation_id}", response_model=Dict[str, Any])
async def get_operation_status(operation_id: str):
    """
    Get the status of a specific operation.
    
    Args:
        operation_id: ID of the operation
    """
    tracker = get_progress_tracker()
    status = await tracker.get_operation_status(operation_id)
    
    if not status:
        raise HTTPException(status_code=404, detail=f"Operation {operation_id} not found")
        
    return status


@router.post("/{operation_id}/cancel", response_model=Dict[str, Any])
async def cancel_operation(operation_id: str):
    """
    Cancel an in-progress operation.
    
    Args:
        operation_id: ID of the operation
    """
    tracker = get_progress_tracker()
    
    # Check if operation exists
    status = await tracker.get_operation_status(operation_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Operation {operation_id} not found")
    
    # Check if operation can be canceled
    if status["status"] in [ProgressStatus.COMPLETED.value, 
                           ProgressStatus.FAILED.value, 
                           ProgressStatus.CANCELED.value]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot cancel operation in state {status['status']}"
        )
    
    # Cancel the operation
    success = await tracker.cancel_operation(operation_id)
    
    if not success:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to cancel operation {operation_id}"
        )
    
    # Get the updated status
    updated_status = await tracker.get_operation_status(operation_id)
    return updated_status


@router.websocket("/ws/{operation_id}")
async def operation_websocket(websocket: WebSocket, operation_id: str):
    """
    WebSocket endpoint for real-time updates on operation progress.
    
    Args:
        websocket: The WebSocket connection
        operation_id: ID of the operation to monitor
    """
    await websocket.accept()
    
    tracker = get_progress_tracker()
    
    # Get the current status first
    status = await tracker.get_operation_status(operation_id)
    
    if not status:
        await websocket.send_json({
            "type": "error",
            "message": f"Operation {operation_id} not found"
        })
        await websocket.close()
        return
        
    # Send the initial status
    await websocket.send_json({
        "type": "status",
        "data": status
    })
    
    # Set up a callback for updates
    update_event = asyncio.Event()
    latest_status = status
    
    async def update_callback(op_id, new_status):
        nonlocal latest_status
        latest_status = new_status
        update_event.set()
    
    # Subscribe to updates
    await tracker.subscribe_to_updates(operation_id, update_callback)
    
    try:
        while True:
            # Wait for updates or client messages
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(update_event.wait()),
                    asyncio.create_task(websocket.receive_text())
                ],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            for task in done:
                try:
                    result = task.result()
                    
                    # If this is an update event
                    if task == asyncio.create_task(update_event.wait()):
                        update_event.clear()
                        await websocket.send_json({
                            "type": "status",
                            "data": latest_status
                        })
                        
                        # Close the connection if the operation is complete
                        status_value = latest_status.get("status")
                        if status_value in [ProgressStatus.COMPLETED.value, 
                                          ProgressStatus.FAILED.value, 
                                          ProgressStatus.CANCELED.value,
                                          ProgressStatus.TIMEOUT.value]:
                            await websocket.send_json({
                                "type": "complete",
                                "message": f"Operation {status_value}"
                            })
                            await websocket.close()
                            break
                    
                    # If this is a message from the client
                    else:
                        # Handle client messages (e.g., cancel request)
                        try:
                            msg = json.loads(result)
                            
                            if msg.get("type") == "cancel":
                                await tracker.cancel_operation(operation_id)
                                # Status update will be sent via the update callback
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Received invalid JSON from WebSocket: {result}")
                
                except asyncio.CancelledError:
                    continue
                    
    except WebSocketDisconnect:
        # Client disconnected
        logger.info(f"Client disconnected from WebSocket for operation {operation_id}")
    finally:
        # Unsubscribe from updates
        await tracker.unsubscribe_from_updates(operation_id, update_callback)