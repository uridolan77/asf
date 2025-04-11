API endpoints for messaging metrics.
This module provides API endpoints for monitoring the messaging system.
from typing import Dict, List
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
    System status response.
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