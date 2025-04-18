from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from datetime import datetime
import uuid
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

@router.get("/api/medical/clients")
def list_medical_clients(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all available medical clients.
    """
    # Mock data for medical clients
    clients = [
        {
            "id": "ncbi",
            "name": "NCBI",
            "description": "National Center for Biotechnology Information",
            "status": "connected",
            "last_checked": datetime.utcnow().isoformat(),
            "api_version": "2.0",
            "endpoints": ["pubmed", "pmc", "gene"]
        },
        {
            "id": "umls",
            "name": "UMLS",
            "description": "Unified Medical Language System",
            "status": "connected",
            "last_checked": datetime.utcnow().isoformat(),
            "api_version": "1.5",
            "endpoints": ["search", "semantic-network", "metathesaurus"]
        },
        {
            "id": "clinicaltrials",
            "name": "ClinicalTrials.gov",
            "description": "Database of clinical studies",
            "status": "connected",
            "last_checked": datetime.utcnow().isoformat(),
            "api_version": "1.0",
            "endpoints": ["search", "study", "condition"]
        },
        {
            "id": "cochrane",
            "name": "Cochrane Library",
            "description": "Collection of databases in medicine and healthcare",
            "status": "disconnected",
            "last_checked": datetime.utcnow().isoformat(),
            "api_version": "2.1",
            "endpoints": ["reviews", "trials", "meta-analyses"]
        },
        {
            "id": "crossref",
            "name": "Crossref",
            "description": "Official DOI registration agency for scholarly publications",
            "status": "connected",
            "last_checked": datetime.utcnow().isoformat(),
            "api_version": "1.0",
            "endpoints": ["works", "journals", "funders"]
        },
        {
            "id": "snomed",
            "name": "SNOMED CT",
            "description": "Systematized Nomenclature of Medicine - Clinical Terms",
            "status": "connected",
            "last_checked": datetime.utcnow().isoformat(),
            "api_version": "3.0",
            "endpoints": ["concepts", "descriptions", "relationships"]
        }
    ]

    return {
        "success": True,
        "message": f"Retrieved {len(clients)} medical clients",
        "data": {
            "clients": clients,
            "total": len(clients)
        }
    }

@router.get("/api/medical/clients/{client_id}")
def get_medical_client(
    client_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get details of a specific medical client.
    """
    # Mock data for specific clients
    clients = {
        "ncbi": {
            "id": "ncbi",
            "name": "NCBI",
            "description": "National Center for Biotechnology Information",
            "status": "connected",
            "last_checked": datetime.utcnow().isoformat(),
            "api_version": "2.0",
            "endpoints": ["pubmed", "pmc", "gene"],
            "config": {
                "api_key": "••••••••••••••••",
                "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
                "rate_limit": "3 requests per second",
                "timeout": 30
            },
            "usage": {
                "requests_today": 145,
                "requests_this_month": 2876,
                "last_request": datetime.utcnow().isoformat()
            },
            "capabilities": [
                "Full-text search",
                "Citation retrieval",
                "Author search",
                "Journal lookup"
            ]
        },
        "umls": {
            "id": "umls",
            "name": "UMLS",
            "description": "Unified Medical Language System",
            "status": "connected",
            "last_checked": datetime.utcnow().isoformat(),
            "api_version": "1.5",
            "endpoints": ["search", "semantic-network", "metathesaurus"],
            "config": {
                "api_key": "••••••••••••••••",
                "base_url": "https://uts-ws.nlm.nih.gov/rest",
                "rate_limit": "5 requests per second",
                "timeout": 20
            },
            "usage": {
                "requests_today": 87,
                "requests_this_month": 1543,
                "last_request": datetime.utcnow().isoformat()
            },
            "capabilities": [
                "Concept lookup",
                "Semantic type retrieval",
                "Term relationships",
                "Source vocabulary integration"
            ]
        },
        "clinicaltrials": {
            "id": "clinicaltrials",
            "name": "ClinicalTrials.gov",
            "description": "Database of clinical studies",
            "status": "connected",
            "last_checked": datetime.utcnow().isoformat(),
            "api_version": "1.0",
            "endpoints": ["search", "study", "condition"],
            "config": {
                "api_key": None,  # No API key required
                "base_url": "https://clinicaltrials.gov/api",
                "rate_limit": "10 requests per minute",
                "timeout": 30
            },
            "usage": {
                "requests_today": 42,
                "requests_this_month": 876,
                "last_request": datetime.utcnow().isoformat()
            },
            "capabilities": [
                "Study search",
                "Condition-based filtering",
                "Intervention search",
                "Status tracking"
            ]
        },
        "cochrane": {
            "id": "cochrane",
            "name": "Cochrane Library",
            "description": "Collection of databases in medicine and healthcare",
            "status": "disconnected",
            "last_checked": datetime.utcnow().isoformat(),
            "api_version": "2.1",
            "endpoints": ["reviews", "trials", "meta-analyses"],
            "config": {
                "api_key": "••••••••••••••••",
                "base_url": "https://mrw.interscience.wiley.com/cochrane/api",
                "rate_limit": "2 requests per second",
                "timeout": 45
            },
            "usage": {
                "requests_today": 0,
                "requests_this_month": 324,
                "last_request": (datetime.utcnow().replace(day=datetime.utcnow().day-2)).isoformat()
            },
            "capabilities": [
                "Systematic review search",
                "Meta-analysis retrieval",
                "Clinical trial lookup",
                "Evidence grading"
            ],
            "error": "Authentication failed: Invalid API key"
        },
        "crossref": {
            "id": "crossref",
            "name": "Crossref",
            "description": "Official DOI registration agency for scholarly publications",
            "status": "connected",
            "last_checked": datetime.utcnow().isoformat(),
            "api_version": "1.0",
            "endpoints": ["works", "journals", "funders"],
            "config": {
                "api_key": "••••••••••••••••",
                "base_url": "https://api.crossref.org",
                "rate_limit": "50 requests per minute",
                "timeout": 15
            },
            "usage": {
                "requests_today": 63,
                "requests_this_month": 1287,
                "last_request": datetime.utcnow().isoformat()
            },
            "capabilities": [
                "DOI lookup",
                "Citation matching",
                "Journal metadata",
                "Funder information"
            ]
        },
        "snomed": {
            "id": "snomed",
            "name": "SNOMED CT",
            "description": "Systematized Nomenclature of Medicine - Clinical Terms",
            "status": "connected",
            "last_checked": datetime.utcnow().isoformat(),
            "api_version": "3.0",
            "endpoints": ["concepts", "descriptions", "relationships"],
            "config": {
                "api_key": "••••••••••••••••",
                "base_url": "https://browser.ihtsdotools.org/snowstorm/snomed-ct/api",
                "rate_limit": "20 requests per minute",
                "timeout": 25
            },
            "usage": {
                "requests_today": 112,
                "requests_this_month": 2143,
                "last_request": datetime.utcnow().isoformat()
            },
            "capabilities": [
                "Concept lookup",
                "Hierarchical relationships",
                "Term mapping",
                "Subsumption testing"
            ]
        }
    }

    if client_id not in clients:
        raise HTTPException(status_code=404, detail=f"Medical client {client_id} not found")

    return {
        "success": True,
        "message": f"Retrieved details for {client_id}",
        "data": clients[client_id]
    }

@router.post("/api/medical/clients/{client_id}/test")
def test_medical_client(
    client_id: str,
    admin_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Test connection to a medical client (admin only).
    """
    # Mock test results
    test_results = {
        "ncbi": {
            "success": True,
            "message": "Connection successful",
            "response_time": 0.87,
            "endpoints_tested": ["pubmed", "pmc"],
            "details": {
                "pubmed": {"status": "OK", "response_time": 0.65},
                "pmc": {"status": "OK", "response_time": 1.09}
            }
        },
        "umls": {
            "success": True,
            "message": "Connection successful",
            "response_time": 1.23,
            "endpoints_tested": ["search", "semantic-network"],
            "details": {
                "search": {"status": "OK", "response_time": 1.05},
                "semantic-network": {"status": "OK", "response_time": 1.41}
            }
        },
        "clinicaltrials": {
            "success": True,
            "message": "Connection successful",
            "response_time": 1.56,
            "endpoints_tested": ["search", "study"],
            "details": {
                "search": {"status": "OK", "response_time": 1.32},
                "study": {"status": "OK", "response_time": 1.80}
            }
        },
        "cochrane": {
            "success": False,
            "message": "Connection failed: Authentication error",
            "response_time": None,
            "endpoints_tested": ["reviews"],
            "details": {
                "reviews": {"status": "ERROR", "error": "Authentication failed: Invalid API key"}
            }
        },
        "crossref": {
            "success": True,
            "message": "Connection successful",
            "response_time": 0.95,
            "endpoints_tested": ["works", "journals"],
            "details": {
                "works": {"status": "OK", "response_time": 0.88},
                "journals": {"status": "OK", "response_time": 1.02}
            }
        },
        "snomed": {
            "success": True,
            "message": "Connection successful",
            "response_time": 1.34,
            "endpoints_tested": ["concepts", "descriptions"],
            "details": {
                "concepts": {"status": "OK", "response_time": 1.21},
                "descriptions": {"status": "OK", "response_time": 1.47}
            }
        }
    }

    if client_id not in test_results:
        raise HTTPException(status_code=404, detail=f"Medical client {client_id} not found")

    return {
        "success": True,
        "message": f"Test results for {client_id}",
        "data": test_results[client_id]
    }

@router.put("/api/medical/clients/{client_id}/config")
def update_client_config(
    client_id: str,
    config: dict = Body(...),
    admin_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Update configuration for a medical client (admin only).
    """
    # Check if client exists
    clients = ["ncbi", "umls", "clinicaltrials", "cochrane", "crossref", "snomed"]
    if client_id not in clients:
        raise HTTPException(status_code=404, detail=f"Medical client {client_id} not found")

    # In a real implementation, this would update the configuration in a database
    # For now, we'll just return a mock response
    return {
        "success": True,
        "message": f"Updated configuration for {client_id}",
        "data": {
            "id": client_id,
            "config": {
                **config,
                # Mask API key in response
                "api_key": "••••••••••••••••" if "api_key" in config else None
            },
            "updated_at": datetime.utcnow().isoformat()
        }
    }

@router.get("/api/medical/clients/{client_id}/usage")
def get_client_usage(
    client_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get usage statistics for a medical client.
    """
    # Check if client exists
    clients = ["ncbi", "umls", "clinicaltrials", "cochrane", "crossref", "snomed"]
    if client_id not in clients:
        raise HTTPException(status_code=404, detail=f"Medical client {client_id} not found")

    # Mock usage data
    daily_usage = [
        {"date": (datetime.utcnow().replace(day=datetime.utcnow().day-i)).strftime("%Y-%m-%d"), "requests": random.randint(50, 200)}
        for i in range(30)
    ]

    endpoint_usage = {
        "ncbi": {
            "pubmed": random.randint(500, 1500),
            "pmc": random.randint(200, 800),
            "gene": random.randint(50, 300)
        },
        "umls": {
            "search": random.randint(400, 1200),
            "semantic-network": random.randint(100, 500),
            "metathesaurus": random.randint(200, 700)
        },
        "clinicaltrials": {
            "search": random.randint(300, 900),
            "study": random.randint(200, 600),
            "condition": random.randint(100, 400)
        },
        "cochrane": {
            "reviews": random.randint(100, 500),
            "trials": random.randint(50, 300),
            "meta-analyses": random.randint(30, 200)
        },
        "crossref": {
            "works": random.randint(400, 1000),
            "journals": random.randint(200, 600),
            "funders": random.randint(50, 250)
        },
        "snomed": {
            "concepts": random.randint(500, 1500),
            "descriptions": random.randint(300, 900),
            "relationships": random.randint(200, 700)
        }
    }

    return {
        "success": True,
        "message": f"Retrieved usage statistics for {client_id}",
        "data": {
            "id": client_id,
            "total_requests": sum(day["requests"] for day in daily_usage),
            "daily_usage": daily_usage,
            "endpoint_usage": endpoint_usage.get(client_id, {}),
            "usage_period": {
                "start": daily_usage[-1]["date"],
                "end": daily_usage[0]["date"]
            }
        }
    }
