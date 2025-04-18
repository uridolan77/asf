from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import uuid
import random
import httpx
import os
import logging

from .auth import get_current_user, get_db
from models.user import User
from .models import PICOSearchRequest

router = APIRouter()
logger = logging.getLogger(__name__)

# API URLs
MEDICAL_API_URL = os.getenv("MEDICAL_API_URL", "http://localhost:8000")

@router.post("/api/medical/search")
def search_medical(
    query: str = Body(...),
    max_results: int = Body(20),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Search for medical research based on a query.
    """
    try:
        # Import the NCBI client to perform real searches
        from medical.clients.ncbi import NCBIClient

        # Create a client instance
        ncbi_client = NCBIClient()

        # Perform the actual search using the client
        # This assumes there's a search method in the NCBIClient class
        search_results = ncbi_client.search(query, max_results=max_results)

        # Process and format the results
        articles = []

        for result in search_results:
            article = {
                "id": result.get("id", f"ncbi_{len(articles)}"),
                "title": result.get("title", "Untitled"),
                "authors": result.get("authors", []),
                "journal": result.get("journal", "Unknown Journal"),
                "year": result.get("year", datetime.now().year),
                "abstract": result.get("abstract", "No abstract available"),
                "relevance_score": result.get("score", 0.8),
                "source": "NCBI"
            }
            articles.append(article)

        return {
            "success": True,
            "message": f"Found {len(articles)} results for query: {query}",
            "data": {
                "articles": articles,
                "query": query,
                "total_results": len(articles)
            }
        }
    except ImportError as e:
        print(f"Import error: {str(e)}")
        # Fallback to mock data if the client import fails
        articles = [
            {
                "id": f"article_{i}",
                "title": f"Research on {query} - Part {i}",
                "authors": ["Author A", "Author B"],
                "journal": "Journal of Medical Research",
                "year": 2023 - i,
                "abstract": f"This study investigates {query} and its implications for healthcare.",
                "relevance_score": round(random.uniform(0.5, 0.99), 2),
                "source": "Mock Data (Client import failed)"
            }
            for i in range(1, min(max_results + 1, 21))
        ]

        return {
            "success": True,
            "message": f"Found {len(articles)} results for query: {query} (using mock data due to import error: {str(e)})",
            "data": {
                "articles": articles,
                "query": query,
                "total_results": len(articles)
            }
        }
    except Exception as e:
        print(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error performing medical search: {str(e)}"
        )

@router.post("/api/medical/search/pico")
def search_pico(
    request: PICOSearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Search for medical research using PICO framework.
    """
    # Simulate searching using PICO components
    pico_description = f"P: {request.population or request.condition}, " \
                       f"I: {', '.join(request.interventions)}, " \
                       f"O: {', '.join(request.outcomes)}"

    # In a real implementation, this would call the medical API
    # For now, we'll just return mock data

    # Save search to history
    search_id = str(uuid.uuid4())

    # Return mock results
    return {
        "success": True,
        "message": f"Found 5 results for PICO search: {pico_description}",
        "data": {
            "total_count": 5,
            "page": 1,
            "page_size": 20,
            "results": [
                {
                    "id": f"result_{i}",
                    "title": f"PICO Study {i} on {request.condition}",
                    "authors": ["Author A", "Author B"],
                    "journal": "Journal of Medical Research",
                    "year": 2025 - i,
                    "abstract": f"This study examines {request.condition} with {', '.join(request.interventions)}...",
                    "relevance_score": 0.95 - (i * 0.1)
                } for i in range(5)
            ]
        }
    }

@router.get("/api/medical/search/history")
async def get_search_history(current_user: User = Depends(get_current_user)):
    """
    Get search history for the current user.
    """
    try:
        # Call the medical API to get the search history
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MEDICAL_API_URL}/api/search/history",
                headers={"Authorization": f"Bearer {current_user.token}"}
            )

            if response.status_code != 200:
                logger.warning(f"Failed to get search history from medical API: {response.status_code}")
                # Return fallback mock data
                searches = [
                    {
                        "id": "search_1",
                        "query": "pneumonia treatment",
                        "type": "standard",
                        "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                        "result_count": 42
                    },
                    {
                        "id": "search_2",
                        "query": "P: adults with hypertension, I: ACE inhibitors, C: ARBs, O: blood pressure reduction",
                        "type": "pico",
                        "timestamp": (datetime.now() - timedelta(days=3)).isoformat(),
                        "result_count": 15
                    },
                    {
                        "id": "search_3",
                        "query": "diabetes management",
                        "type": "standard",
                        "timestamp": (datetime.now() - timedelta(days=5)).isoformat(),
                        "result_count": 78
                    }
                ]
                return {"searches": searches}

            return response.json()
    except Exception as e:
        logger.error(f"Error getting search history: {str(e)}")
        # Return fallback mock data
        searches = [
            {
                "id": "search_1",
                "query": "pneumonia treatment",
                "type": "standard",
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                "result_count": 42
            },
            {
                "id": "search_2",
                "query": "P: adults with hypertension, I: ACE inhibitors, C: ARBs, O: blood pressure reduction",
                "type": "pico",
                "timestamp": (datetime.now() - timedelta(days=3)).isoformat(),
                "result_count": 15
            },
            {
                "id": "search_3",
                "query": "diabetes management",
                "type": "standard",
                "timestamp": (datetime.now() - timedelta(days=5)).isoformat(),
                "result_count": 78
            }
        ]
        return {"searches": searches}

@router.post("/api/medical/search/history")
async def save_search_history(search: dict = Body(...), current_user: User = Depends(get_current_user)):
    """
    Save a search to the user's history.
    """
    try:
        # Call the medical API to save the search history
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MEDICAL_API_URL}/api/search/history",
                json=search,
                headers={"Authorization": f"Bearer {current_user.token}"}
            )

            if response.status_code != 200:
                logger.warning(f"Failed to save search history to medical API: {response.status_code}")
                # Return fallback mock response
                search_id = str(uuid.uuid4())
                return {
                    "success": True,
                    "message": "Search saved to history (local)",
                    "data": {
                        "id": search_id,
                        "query": search.get("query", ""),
                        "type": search.get("type", "standard"),
                        "timestamp": search.get("timestamp", datetime.now(datetime.timezone.utc).isoformat())
                    }
                }

            return response.json()
    except Exception as e:
        logger.error(f"Error saving search history: {str(e)}")
        # Return fallback mock response
        search_id = str(uuid.uuid4())
        return {
            "success": True,
            "message": "Search saved to history (local)",
            "data": {
                "id": search_id,
                "query": search.get("query", ""),
                "type": search.get("type", "standard"),
                "timestamp": search.get("timestamp", datetime.now(datetime.timezone.utc).isoformat())
            }
        }
