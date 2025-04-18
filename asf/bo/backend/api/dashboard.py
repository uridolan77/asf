from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import random
import httpx
import os
import logging

from .auth import get_current_user, get_current_admin_user, get_db
from models.user import User

router = APIRouter()
logger = logging.getLogger(__name__)

# API URLs
MEDICAL_API_URL = os.getenv("MEDICAL_API_URL", "http://localhost:8000")

# Global variable to store system settings (in a real app, this would be in the database)
SYSTEM_SETTINGS = {
    "sessionTimeout": 60,
    "enableNotifications": True,
    "darkMode": False,
    "language": "en"
}

@router.get("/api/stats")
def get_dashboard_stats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get basic dashboard statistics."""
    # Mock data for dashboard stats
    user_count = 15  # In a real app, this would come from the database
    active_sessions = 8
    system_status = "Operational"

    # Generate mock monthly data
    monthly_data = [
        {"month": "Jan", "searches": 45, "analyses": 18},
        {"month": "Feb", "searches": 52, "analyses": 22},
        {"month": "Mar", "searches": 49, "analyses": 25},
        {"month": "Apr", "searches": 63, "analyses": 30},
        {"month": "May", "searches": 55, "analyses": 24}
    ]

    return {
        "user_count": user_count,
        "active_sessions": active_sessions,
        "system_status": system_status,
        "monthly_data": monthly_data
    }

@router.get("/api/admin/stats")
def get_admin_stats(admin_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Get detailed system statistics (admin only)."""
    # Count all users
    user_count = db.query(User).count()

    # For demo purposes, we'll simulate active sessions
    # In a real app, you'd track this in a sessions table or using Redis
    active_sessions = random.randint(1, user_count)

    return {
        "user_count": user_count,
        "active_sessions": active_sessions,
        "admin_data": {
            "api_calls_today": random.randint(500, 2000),
            "avg_response_time": round(random.uniform(0.1, 0.5), 2),
            "errors_today": random.randint(0, 50),
            "server_load": round(random.uniform(10, 60), 1)
        }
    }

@router.get("/api/settings")
def get_system_settings(admin_user: User = Depends(get_current_admin_user)):
    """Get system settings (admin only)."""
    return SYSTEM_SETTINGS

@router.put("/api/settings")
def update_system_settings(settings: dict, admin_user: User = Depends(get_current_admin_user)):
    """Update system settings (admin only)."""
    # Update global settings
    SYSTEM_SETTINGS["sessionTimeout"] = settings.get("sessionTimeout", SYSTEM_SETTINGS["sessionTimeout"])
    SYSTEM_SETTINGS["enableNotifications"] = settings.get("enableNotifications", SYSTEM_SETTINGS["enableNotifications"])
    SYSTEM_SETTINGS["darkMode"] = settings.get("darkMode", SYSTEM_SETTINGS["darkMode"])
    SYSTEM_SETTINGS["language"] = settings.get("language", SYSTEM_SETTINGS["language"])

    # In a real application, this would be saved to a database

    return SYSTEM_SETTINGS

@router.get("/api/dashboard-test")
def get_dashboard_test_data():
    """Get test dashboard data without authentication."""
    # Mock data for dashboard stats
    user_count = 15
    active_sessions = 8
    system_status = "Operational"

    # Generate mock monthly data
    monthly_data = [
        {"month": "Jan", "searches": 45, "analyses": 18},
        {"month": "Feb", "searches": 52, "analyses": 22},
        {"month": "Mar", "searches": 49, "analyses": 25},
        {"month": "Apr", "searches": 63, "analyses": 30},
        {"month": "May", "searches": 55, "analyses": 24}
    ]

    return {
        "user_count": user_count,
        "active_sessions": active_sessions,
        "system_status": system_status,
        "monthly_data": monthly_data
    }

@router.get("/api/research-metrics")
def get_research_metrics(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get research metrics for dashboard charts."""
    # Mock data for research metrics
    metrics = [
        {"category": "Clinical Trials", "count": 145},
        {"category": "Meta-analyses", "count": 72},
        {"category": "Systematic Reviews", "count": 98},
        {"category": "Cohort Studies", "count": 123},
        {"category": "Case Reports", "count": 86}
    ]

    return metrics

@router.get("/api/recent-updates")
def get_recent_updates(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get recent research updates for the dashboard."""
    # Mock data for recent updates
    updates = {
        "last_updated": datetime.utcnow().isoformat(),
        "items": [
            {
                "title": "Procalcitonin-guided antibiotic therapy in CAP shows promising results",
                "date": "Apr 2025",
                "link": "/analysis/123"
            },
            {
                "title": "New data on antibiotic resistance patterns in Streptococcus pneumoniae",
                "date": "Mar 2025",
                "link": "/knowledge-base/456"
            },
            {
                "title": "Post-COVID patterns in respiratory infections suggest modified treatment approaches",
                "date": "Feb 2025",
                "link": "/clinical-data/789"
            },
            {
                "title": "Machine learning model improves early detection of sepsis in pneumonia patients",
                "date": "Feb 2025",
                "link": "/ml-services/012"
            },
            {
                "title": "Updated guidelines for CAP management published by international consortium",
                "date": "Jan 2025",
                "link": "/knowledge-base/345"
            }
        ]
    }

    return updates

@router.get("/api/llm-usage")
async def get_llm_usage(current_user: User = Depends(get_current_user)):
    """Get LLM usage statistics for the dashboard."""
    try:
        # Call the medical API to get the LLM usage statistics
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MEDICAL_API_URL}/api/llm/usage",
                headers={"Authorization": f"Bearer {current_user.token}"}
            )

            if response.status_code != 200:
                logger.warning(f"Failed to get LLM usage from API: {response.status_code}")
                # Return fallback mock data
                usage = [
                    { "model": "gpt-4o", "usage_count": 2580 },
                    { "model": "claude-3-opus", "usage_count": 1420 },
                    { "model": "biomedlm-2-7b", "usage_count": 3850 },
                    { "model": "mistralai/Mixtral-8x7B", "usage_count": 980 }
                ]
                return {"usage": usage}

            # Process the response to match the expected format
            data = response.json()
            if isinstance(data, list):
                # If the API returns a list directly, use it
                return {"usage": data}
            elif "usage" in data:
                # If the API returns a dict with a usage key, use that
                return data
            else:
                # Try to extract model usage from the response
                usage = []
                for model, count in data.get("components", {}).get("gateway", {}).get("models", {}).items():
                    usage.append({"model": model, "usage_count": count})
                return {"usage": usage}
    except Exception as e:
        logger.error(f"Error getting LLM usage: {str(e)}")
        # Return fallback mock data
        usage = [
            { "model": "gpt-4o", "usage_count": 2580 },
            { "model": "claude-3-opus", "usage_count": 1420 },
            { "model": "biomedlm-2-7b", "usage_count": 3850 },
            { "model": "mistralai/Mixtral-8x7B", "usage_count": 980 }
        ]
        return {"usage": usage}

@router.get("/api/public/stats")
def get_public_dashboard_stats():
    """Get basic dashboard statistics without requiring authentication."""
    # Mock data for dashboard stats
    user_count = 15
    active_sessions = 8
    system_status = "Operational"

    # Generate mock monthly data
    monthly_data = [
        {"month": "Jan", "searches": 45, "analyses": 18},
        {"month": "Feb", "searches": 52, "analyses": 22},
        {"month": "Mar", "searches": 49, "analyses": 25},
        {"month": "Apr", "searches": 63, "analyses": 30},
        {"month": "May", "searches": 55, "analyses": 24}
    ]

    return {
        "user_count": user_count,
        "active_sessions": active_sessions,
        "system_status": system_status,
        "monthly_data": monthly_data
    }

@router.get("/api/public/research-metrics")
def get_public_research_metrics():
    """Get research metrics for dashboard charts without requiring authentication."""
    # Mock data for research metrics
    metrics = [
        {"category": "Clinical Trials", "count": 145},
        {"category": "Meta-analyses", "count": 72},
        {"category": "Systematic Reviews", "count": 98},
        {"category": "Cohort Studies", "count": 123},
        {"category": "Case Reports", "count": 86}
    ]

    return metrics

@router.get("/api/public/recent-updates")
def get_public_recent_updates():
    """Get recent research updates for the dashboard without requiring authentication."""
    # Mock data for recent updates
    updates = {
        "last_updated": datetime.utcnow().isoformat(),
        "items": [
            {
                "title": "Procalcitonin-guided antibiotic therapy in CAP shows promising results",
                "date": "Apr 2025",
                "link": "/analysis/123"
            },
            {
                "title": "New data on antibiotic resistance patterns in Streptococcus pneumoniae",
                "date": "Mar 2025",
                "link": "/knowledge-base/456"
            },
            {
                "title": "Post-COVID patterns in respiratory infections suggest modified treatment approaches",
                "date": "Feb 2025",
                "link": "/clinical-data/789"
            },
            {
                "title": "Machine learning model improves early detection of sepsis in pneumonia patients",
                "date": "Feb 2025",
                "link": "/ml-services/012"
            },
            {
                "title": "Updated guidelines for CAP management published by international consortium",
                "date": "Jan 2025",
                "link": "/knowledge-base/345"
            }
        ]
    }

    return updates
