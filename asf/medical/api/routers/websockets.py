"""WebSocket endpoints for the Medical Research Synthesizer API.

This module provides WebSocket endpoints for real-time updates.
"""

from fastapi import APIRouter, WebSocket, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from asf.medical.storage.database import get_db_session
from asf.medical.storage.models import MedicalUser
from asf.medical.api.dependencies import get_current_user_ws
from asf.medical.api.websockets.task_updates import handle_task_updates
from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["websockets"])

@router.websocket("/ws/tasks")
async def websocket_task_updates(
    websocket: WebSocket,
    db: AsyncSession = Depends(get_db_session),
    user: MedicalUser = Depends(get_current_user_ws)
):
    """
    WebSocket endpoint for task updates.
    This endpoint allows clients to subscribe to real-time task updates.
    Args:
        websocket: WebSocket connection
        db: Database session
        user: Current user (optional)
    """
    await handle_task_updates(websocket, db, user)

@router.websocket("/ws/tasks/public")
async def websocket_public_task_updates(
    websocket: WebSocket,
    db: AsyncSession = Depends(get_db_session)
):
    """
    WebSocket endpoint for public task updates.
    This endpoint allows clients to subscribe to real-time task updates
    without authentication. Only public tasks will be visible.
    Args:
        websocket: WebSocket connection
        db: Database session
    """
    await handle_task_updates(websocket, db, None)