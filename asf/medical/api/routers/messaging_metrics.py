"""
API endpoints for messaging metrics.

This module provides API endpoints for monitoring the messaging system.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel

from asf.medical.core.logging_config import get_logger
from asf.medical.core.messaging.broker import get_message_broker
from asf.medical.storage.database import get_db_session
from asf.medical.storage.repositories.task_repository import TaskRepository
from asf.medical.api.dependencies import get_admin_user
from asf.medical.storage.models import User

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/messaging-metrics", tags=["messaging-metrics"])

class SystemStatusResponse(BaseModel):
    """System status response."""
    connection_status: str
    worker_status: str
    message_processing_status: str
    dead_letter_count: int
    uptime: str

class MessageCountsResponse(BaseModel):
    """Message counts response."""
    tasks: int
    events: int
    commands: int
    total: int

class TaskStatusResponse(BaseModel):
    """Task status response."""
    pending: int
    running: int
    completed: int
    failed: int
    retrying: int
    cancelled: int
    total: int

class RecentMessage(BaseModel):
    """Recent message model."""
    id: str
    type: str
    subtype: str
    timestamp: str

class RecentMessagesResponse(BaseModel):
    """Recent messages response."""
    messages: List[RecentMessage]

class MetricsDataPoint(BaseModel):
    """Metrics data point."""
    timestamp: str
    value: float

class MetricsResponse(BaseModel):
    """Metrics response."""
    throughput: List[MetricsDataPoint]
    queue_sizes: Dict[str, List[MetricsDataPoint]]
    processing_times: Dict[str, List[MetricsDataPoint]]
    error_rates: List[MetricsDataPoint]

@router.get("/system-status", response_model=SystemStatusResponse)
async def get_system_status(
    current_user: User = Depends(get_admin_user)
):
    """
    Get the messaging system status.
    
    Args:
        current_user: Current admin user
        
    Returns:
        System status
    """
    try:
        # Get the message broker
        broker = get_message_broker()
        
        # Check connection status
        connection_status = "Connected" if broker.is_connected() else "Disconnected"
        
        # Get worker status (this would be implemented in a real system)
        worker_status = "Running (5/5)"
        
        # Get message processing status
        message_processing_status = "Normal"
        
        # Get dead letter count
        dead_letter_count = 0
        
        # Get uptime
        uptime = "3 days, 12 hours, 45 minutes"
        
        return SystemStatusResponse(
            connection_status=connection_status,
            worker_status=worker_status,
            message_processing_status=message_processing_status,
            dead_letter_count=dead_letter_count,
            uptime=uptime
        )
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting system status: {str(e)}"
        )

@router.get("/message-counts", response_model=MessageCountsResponse)
async def get_message_counts(
    current_user: User = Depends(get_admin_user)
):
    """
    Get message counts.
    
    Args:
        current_user: Current admin user
        
    Returns:
        Message counts
    """
    try:
        # In a real implementation, this would query the message broker
        # For now, we'll return dummy data
        tasks = 245
        events = 1203
        commands = 87
        total = tasks + events + commands
        
        return MessageCountsResponse(
            tasks=tasks,
            events=events,
            commands=commands,
            total=total
        )
    except Exception as e:
        logger.error(f"Error getting message counts: {str(e)}", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting message counts: {str(e)}"
        )

@router.get("/task-status", response_model=TaskStatusResponse)
async def get_task_status(
    current_user: User = Depends(get_admin_user),
    db = Depends(get_db_session)
):
    """
    Get task status counts.
    
    Args:
        current_user: Current admin user
        db: Database session
        
    Returns:
        Task status counts
    """
    try:
        # Get the task repository
        task_repository = TaskRepository()
        
        # Get task count by status
        counts = await task_repository.get_task_count_by_status(db)
        
        # Calculate total
        total = sum(counts.values())
        
        return TaskStatusResponse(
            pending=counts.get("pending", 0),
            running=counts.get("running", 0),
            completed=counts.get("completed", 0),
            failed=counts.get("failed", 0),
            retrying=counts.get("retrying", 0),
            cancelled=counts.get("cancelled", 0),
            total=total
        )
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting task status: {str(e)}"
        )

@router.get("/recent-messages", response_model=RecentMessagesResponse)
async def get_recent_messages(
    limit: int = Query(10, description="Maximum number of messages to return"),
    current_user: User = Depends(get_admin_user)
):
    """
    Get recent messages.
    
    Args:
        limit: Maximum number of messages to return
        current_user: Current admin user
        
    Returns:
        Recent messages
    """
    try:
        # In a real implementation, this would query the message broker
        # For now, we'll return dummy data
        messages = [
            RecentMessage(
                id="msg-123456",
                type="TASK",
                subtype="SEARCH_PUBMED",
                timestamp=datetime.now().isoformat()
            ),
            RecentMessage(
                id="msg-123457",
                type="EVENT",
                subtype="task.started",
                timestamp=(datetime.now() - timedelta(seconds=1)).isoformat()
            ),
            RecentMessage(
                id="msg-123458",
                type="EVENT",
                subtype="task.progress",
                timestamp=(datetime.now() - timedelta(seconds=3)).isoformat()
            ),
            RecentMessage(
                id="msg-123459",
                type="EVENT",
                subtype="task.completed",
                timestamp=(datetime.now() - timedelta(seconds=6)).isoformat()
            ),
            RecentMessage(
                id="msg-123460",
                type="TASK",
                subtype="ANALYZE_BIAS",
                timestamp=(datetime.now() - timedelta(seconds=30)).isoformat()
            )
        ]
        
        return RecentMessagesResponse(
            messages=messages[:limit]
        )
    except Exception as e:
        logger.error(f"Error getting recent messages: {str(e)}", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting recent messages: {str(e)}"
        )

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    time_range: str = Query("24h", description="Time range (1h, 6h, 24h, 7d, 30d)"),
    current_user: User = Depends(get_admin_user)
):
    """
    Get messaging system metrics.
    
    Args:
        time_range: Time range
        current_user: Current admin user
        
    Returns:
        Metrics data
    """
    try:
        # In a real implementation, this would query the metrics database
        # For now, we'll return dummy data
        
        # Parse time range
        hours = 24
        if time_range == "1h":
            hours = 1
        elif time_range == "6h":
            hours = 6
        elif time_range == "24h":
            hours = 24
        elif time_range == "7d":
            hours = 24 * 7
        elif time_range == "30d":
            hours = 24 * 30
        
        # Generate timestamps
        now = datetime.now()
        timestamps = [
            (now - timedelta(hours=i)).isoformat()
            for i in range(hours, 0, -1)
        ]
        
        # Generate throughput data
        throughput = [
            MetricsDataPoint(
                timestamp=timestamp,
                value=50 + (i % 20)  # Random-ish value between 50 and 70
            )
            for i, timestamp in enumerate(timestamps)
        ]
        
        # Generate queue sizes data
        queue_sizes = {
            "tasks": [
                MetricsDataPoint(
                    timestamp=timestamp,
                    value=10 + (i % 5)  # Random-ish value between 10 and 15
                )
                for i, timestamp in enumerate(timestamps)
            ],
            "events": [
                MetricsDataPoint(
                    timestamp=timestamp,
                    value=5 + (i % 3)  # Random-ish value between 5 and 8
                )
                for i, timestamp in enumerate(timestamps)
            ],
            "commands": [
                MetricsDataPoint(
                    timestamp=timestamp,
                    value=2 + (i % 2)  # Random-ish value between 2 and 4
                )
                for i, timestamp in enumerate(timestamps)
            ]
        }
        
        # Generate processing times data
        processing_times = {
            "search": [
                MetricsDataPoint(
                    timestamp=timestamp,
                    value=0.5 + (i % 10) / 20  # Random-ish value between 0.5 and 1.0
                )
                for i, timestamp in enumerate(timestamps)
            ],
            "analysis": [
                MetricsDataPoint(
                    timestamp=timestamp,
                    value=1.0 + (i % 10) / 10  # Random-ish value between 1.0 and 2.0
                )
                for i, timestamp in enumerate(timestamps)
            ],
            "export": [
                MetricsDataPoint(
                    timestamp=timestamp,
                    value=0.8 + (i % 10) / 15  # Random-ish value between 0.8 and 1.5
                )
                for i, timestamp in enumerate(timestamps)
            ]
        }
        
        # Generate error rates data
        error_rates = [
            MetricsDataPoint(
                timestamp=timestamp,
                value=(i % 5) / 100  # Random-ish value between 0 and 0.05
            )
            for i, timestamp in enumerate(timestamps)
        ]
        
        return MetricsResponse(
            throughput=throughput,
            queue_sizes=queue_sizes,
            processing_times=processing_times,
            error_rates=error_rates
        )
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting metrics: {str(e)}"
        )
