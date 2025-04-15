"""
API endpoints for messaging metrics.
This module provides API endpoints for monitoring the messaging system.
"""
from typing import Dict, List
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from ...core.logging_config import get_logger
from ..dependencies import get_admin_user
from ...storage.models import User
logger = get_logger(__name__)
router = APIRouter(prefix="/v1/messaging-metrics", tags=["messaging-metrics"])
class MetricsDataPoint(BaseModel):
    """Metrics data point.

    This model represents a single data point in a metrics time series.
    """
    timestamp: str = Field(..., description="Timestamp in ISO format")
    value: float = Field(..., description="Metric value")


class RecentMessage(BaseModel):
    """Recent message.

    This model represents a recent message in the messaging system.
    """
    id: str = Field(..., description="Message ID")
    type: str = Field(..., description="Message type")
    subtype: str = Field(..., description="Message subtype")
    timestamp: str = Field(..., description="Timestamp in ISO format")


class RecentMessagesResponse(BaseModel):
    """Recent messages response.

    This model represents a response containing recent messages.
    """
    messages: List[RecentMessage] = Field(..., description="List of recent messages")


class MetricsResponse(BaseModel):
    """Metrics response.

    This model represents a response containing messaging system metrics.
    """
    throughput: List[MetricsDataPoint] = Field(..., description="Message throughput over time")
    queue_sizes: Dict[str, List[MetricsDataPoint]] = Field(..., description="Queue sizes over time")
    processing_times: Dict[str, List[MetricsDataPoint]] = Field(..., description="Processing times over time")
    error_rates: List[MetricsDataPoint] = Field(..., description="Error rates over time")
@router.get("/recent-messages", response_model=RecentMessagesResponse)
async def get_recent_messages(
    limit: int = Query(5, description="Maximum number of messages to return"),
    current_user: User = Depends(get_admin_user)  # Used for authorization
):
    """Get recent messages from the messaging system.

    This endpoint retrieves the most recent messages from the messaging system.
    It requires admin privileges to access.

    Args:
        limit: Maximum number of messages to return
        current_user: The authenticated admin user

    Returns:
        RecentMessagesResponse containing the list of recent messages

    Raises:
        HTTPException: If an error occurs while retrieving messages
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
    current_user: User = Depends(get_admin_user)  # Used for authorization
):
    """Get messaging system metrics.

    This endpoint retrieves metrics about the messaging system, such as throughput,
    queue sizes, processing times, and error rates. It requires admin privileges to access.

    Args:
        time_range: Time range for metrics (1h, 6h, 24h, 7d, 30d)
        current_user: The authenticated admin user

    Returns:
        MetricsResponse containing the metrics data

    Raises:
        HTTPException: If an error occurs while retrieving metrics
    """
    try:
        # In a real implementation, this would query the message broker
        # For now, we'll return dummy data
        now = datetime.now()

        # Generate dummy data points based on time range
        if time_range == "1h":
            interval = timedelta(minutes=5)
            num_points = 12
        elif time_range == "6h":
            interval = timedelta(minutes=30)
            num_points = 12
        elif time_range == "24h":
            interval = timedelta(hours=2)
            num_points = 12
        elif time_range == "7d":
            interval = timedelta(hours=12)
            num_points = 14
        elif time_range == "30d":
            interval = timedelta(days=2)
            num_points = 15
        else:
            interval = timedelta(hours=2)
            num_points = 12

        # Generate throughput data
        throughput = []
        for i in range(num_points):
            timestamp = (now - interval * i).isoformat()
            value = 100 - i * 5 + (i % 3) * 10  # Some variation
            throughput.append(MetricsDataPoint(timestamp=timestamp, value=value))

        # Generate queue sizes data
        queue_sizes = {
            "tasks": [
                MetricsDataPoint(timestamp=(now - interval * i).isoformat(), value=20 - i + (i % 4) * 5)
                for i in range(num_points)
            ],
            "events": [
                MetricsDataPoint(timestamp=(now - interval * i).isoformat(), value=15 - i + (i % 3) * 3)
                for i in range(num_points)
            ]
        }

        # Generate processing times data
        processing_times = {
            "search": [
                MetricsDataPoint(timestamp=(now - interval * i).isoformat(), value=0.5 + (i % 5) * 0.1)
                for i in range(num_points)
            ],
            "analysis": [
                MetricsDataPoint(timestamp=(now - interval * i).isoformat(), value=2.0 + (i % 3) * 0.5)
                for i in range(num_points)
            ],
            "export": [
                MetricsDataPoint(timestamp=(now - interval * i).isoformat(), value=1.0 + (i % 4) * 0.2)
                for i in range(num_points)
            ]
        }

        # Generate error rates data
        error_rates = [
            MetricsDataPoint(timestamp=(now - interval * i).isoformat(), value=(i % 5) * 0.5)
            for i in range(num_points)
        ]

        return MetricsResponse(
            throughput=throughput,
            queue_sizes=queue_sizes,
            processing_times=processing_times,
            error_rates=error_rates
        )
    except Exception as e:
        logger.error(f"Error getting messaging metrics: {str(e)}", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting messaging metrics: {str(e)}"
        )

