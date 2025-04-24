"""
Medical Clients API endpoints for the BO backend.

This module provides endpoints for managing medical clients,
including NCBI, UMLS, ClinicalTrials, Cochrane, Crossref, and SNOMED.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import httpx
import logging
import os
import json
from datetime import datetime

from .auth import get_current_user, User
from .utils import handle_api_error

router = APIRouter(prefix="/api/medical/clients", tags=["clients"])

logger = logging.getLogger(__name__)

# Models
class ClientStatus(BaseModel):
    client_id: str
    name: str
    status: str
    last_checked: str
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    api_version: Optional[str] = None
    config: Dict[str, Any]

class ClientConfig(BaseModel):
    api_key: Optional[str] = None
    email: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    cache_ttl: Optional[int] = None
    use_cache: Optional[bool] = None
    access_mode: Optional[str] = None
    edition: Optional[str] = None
    version: Optional[str] = None
    cache_dir: Optional[str] = None

class ClientUpdateRequest(BaseModel):
    config: ClientConfig

class ClientUsageStats(BaseModel):
    client_id: str
    name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    cache_hit_rate: float
    last_used: str
    usage_by_endpoint: Dict[str, int]

# Client configuration file path
CLIENT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "clients.json")

# Ensure config directory exists
os.makedirs(os.path.dirname(CLIENT_CONFIG_PATH), exist_ok=True)

# Default client configurations
DEFAULT_CLIENTS = {
    "ncbi": {
        "name": "NCBI",
        "config": {
            "api_key": os.environ.get("NCBI_API_KEY", ""),
            "email": os.environ.get("NCBI_EMAIL", "your.email@example.com"),
            "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            "timeout": 30.0,
            "max_retries": 3,
            "cache_ttl": 3600,
            "use_cache": True
        }
    },
    "umls": {
        "name": "UMLS",
        "config": {
            "api_key": os.environ.get("UMLS_API_KEY", ""),
            "base_url": "https://uts-ws.nlm.nih.gov/rest",
            "timeout": 30.0,
            "max_retries": 3,
            "cache_ttl": 86400,
            "use_cache": True
        }
    },
    "clinical_trials": {
        "name": "ClinicalTrials.gov",
        "config": {
            "base_url": "https://clinicaltrials.gov/api/v2",
            "timeout": 30.0,
            "max_retries": 3,
            "cache_ttl": 3600,
            "use_cache": True
        }
    },
    "cochrane": {
        "name": "Cochrane Library",
        "config": {
            "api_key": os.environ.get("COCHRANE_API_KEY", ""),
            "base_url": "https://www.cochranelibrary.com",
            "timeout": 30.0,
            "max_retries": 3,
            "cache_ttl": 86400,
            "use_cache": True
        }
    },
    "crossref": {
        "name": "Crossref",
        "config": {
            "email": os.environ.get("CROSSREF_EMAIL", "your.email@example.com"),
            "plus_api_token": os.environ.get("CROSSREF_API_TOKEN", ""),
            "base_url": "https://api.crossref.org",
            "timeout": 30.0,
            "max_retries": 3,
            "cache_ttl": 86400,
            "use_cache": True
        }
    },
    "snomed": {
        "name": "SNOMED CT",
        "config": {
            "api_key": os.environ.get("UMLS_API_KEY", ""),
            "access_mode": "umls",
            "edition": "US",
            "cache_dir": "./snomed_cache",
            "cache_ttl": 86400,
            "use_cache": True
        }
    }
}

def load_client_configs():
    """Load client configurations from file or create default if not exists."""
    try:
        if os.path.exists(CLIENT_CONFIG_PATH):
            with open(CLIENT_CONFIG_PATH, "r") as f:
                return json.load(f)
        else:
            # Create default config file
            with open(CLIENT_CONFIG_PATH, "w") as f:
                json.dump(DEFAULT_CLIENTS, f, indent=2)
            return DEFAULT_CLIENTS
    except Exception as e:
        logger.error(f"Error loading client configurations: {str(e)}")
        return DEFAULT_CLIENTS

def save_client_configs(configs):
    """Save client configurations to file."""
    try:
        with open(CLIENT_CONFIG_PATH, "w") as f:
            json.dump(configs, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving client configurations: {str(e)}")
        return False

@router.get("/", response_model=List[ClientStatus])
async def get_all_clients():
    """
    Get status of all medical clients.

    This endpoint returns the status of all configured medical clients,
    including NCBI, UMLS, ClinicalTrials, Cochrane, Crossref, and SNOMED.
    """
    clients = load_client_configs()
    result = []

    for client_id, client_data in clients.items():
        # Check client status
        status_info = await check_client_status(client_id, client_data)
        result.append(status_info)

    return result

@router.get("/{client_id}", response_model=ClientStatus)
async def get_client(client_id: str, current_user: User = Depends(get_current_user)):
    """
    Get status of a specific medical client.

    This endpoint returns the status of a specific medical client,
    including configuration and connection status.
    """
    clients = load_client_configs()

    if client_id not in clients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client '{client_id}' not found"
        )

    # Check client status
    status_info = await check_client_status(client_id, clients[client_id])
    return status_info

@router.put("/{client_id}", response_model=ClientStatus)
async def update_client(
    client_id: str,
    request: ClientUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Update configuration for a specific medical client.

    This endpoint updates the configuration for a specific medical client
    and returns the updated status.
    """
    clients = load_client_configs()

    if client_id not in clients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client '{client_id}' not found"
        )

    # Update client configuration
    updated_config = {**clients[client_id]["config"]}

    # Update only provided fields
    for key, value in request.config.dict(exclude_unset=True).items():
        if value is not None:
            updated_config[key] = value

    clients[client_id]["config"] = updated_config

    # Save updated configurations
    if not save_client_configs(clients):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save client configuration"
        )

    # Check client status with updated configuration
    status_info = await check_client_status(client_id, clients[client_id])
    return status_info

@router.get("/{client_id}/usage", response_model=ClientUsageStats)
async def get_client_usage(client_id: str, current_user: User = Depends(get_current_user)):
    """
    Get usage statistics for a specific medical client.

    This endpoint returns usage statistics for a specific medical client,
    including total requests, success rate, and cache hit rate.
    """
    clients = load_client_configs()

    if client_id not in clients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client '{client_id}' not found"
        )

    # Get client usage statistics
    usage_stats = await get_client_usage_stats(client_id, clients[client_id])
    return usage_stats

@router.post("/{client_id}/test", response_model=Dict[str, Any])
async def test_client_connection(client_id: str, current_user: User = Depends(get_current_user)):
    """
    Test connection to a specific medical client.

    This endpoint tests the connection to a specific medical client
    and returns the test results.
    """
    clients = load_client_configs()

    if client_id not in clients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client '{client_id}' not found"
        )

    # Test client connection
    test_result = await test_client_connection_impl(client_id, clients[client_id])
    return test_result

async def check_client_status(client_id: str, client_data: Dict[str, Any]) -> ClientStatus:
    """Check the status of a medical client."""
    try:
        # Test connection to client
        start_time = datetime.now()
        is_connected = False
        error_message = None
        api_version = None

        try:
            # Implement client-specific status checks
            if client_id == "ncbi":
                is_connected, api_version = await check_ncbi_status(client_data["config"])
            elif client_id == "umls":
                is_connected, api_version = await check_umls_status(client_data["config"])
            elif client_id == "clinical_trials":
                is_connected, api_version = await check_clinical_trials_status(client_data["config"])
            elif client_id == "cochrane":
                is_connected, api_version = await check_cochrane_status(client_data["config"])
            elif client_id == "crossref":
                is_connected, api_version = await check_crossref_status(client_data["config"])
            elif client_id == "snomed":
                is_connected, api_version = await check_snomed_status(client_data["config"])
            else:
                is_connected = False
                error_message = "Unknown client type"
        except Exception as e:
            is_connected = False
            error_message = str(e)

        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()

        # Determine status
        status_value = "connected" if is_connected else "disconnected"

        return ClientStatus(
            client_id=client_id,
            name=client_data["name"],
            status=status_value,
            last_checked=datetime.now().isoformat(),
            response_time=response_time,
            error_message=error_message,
            api_version=api_version,
            config=client_data["config"]
        )
    except Exception as e:
        logger.error(f"Error checking status for client '{client_id}': {str(e)}")
        return ClientStatus(
            client_id=client_id,
            name=client_data["name"],
            status="error",
            last_checked=datetime.now().isoformat(),
            error_message=str(e),
            config=client_data["config"]
        )

async def get_client_usage_stats(client_id: str, client_data: Dict[str, Any]) -> ClientUsageStats:
    """Get usage statistics for a medical client."""
    # In a real implementation, this would retrieve actual usage statistics
    # For now, we'll return mock data
    return ClientUsageStats(
        client_id=client_id,
        name=client_data["name"],
        total_requests=1000,
        successful_requests=950,
        failed_requests=50,
        average_response_time=0.25,
        cache_hit_rate=0.75,
        last_used=datetime.now().isoformat(),
        usage_by_endpoint={
            "search": 500,
            "get": 300,
            "other": 200
        }
    )

async def test_client_connection_impl(client_id: str, client_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test connection to a medical client."""
    try:
        start_time = datetime.now()
        is_connected = False
        error_message = None
        api_version = None
        details = {}

        try:
            # Implement client-specific connection tests
            if client_id == "ncbi":
                is_connected, api_version, details = await test_ncbi_connection(client_data["config"])
            elif client_id == "umls":
                is_connected, api_version, details = await test_umls_connection(client_data["config"])
            elif client_id == "clinical_trials":
                is_connected, api_version, details = await test_clinical_trials_connection(client_data["config"])
            elif client_id == "cochrane":
                is_connected, api_version, details = await test_cochrane_connection(client_data["config"])
            elif client_id == "crossref":
                is_connected, api_version, details = await test_crossref_connection(client_data["config"])
            elif client_id == "snomed":
                is_connected, api_version, details = await test_snomed_connection(client_data["config"])
            else:
                is_connected = False
                error_message = "Unknown client type"
        except Exception as e:
            is_connected = False
            error_message = str(e)

        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()

        return {
            "success": is_connected,
            "message": "Connection successful" if is_connected else f"Connection failed: {error_message}",
            "response_time": response_time,
            "api_version": api_version,
            "details": details
        }
    except Exception as e:
        logger.error(f"Error testing connection for client '{client_id}': {str(e)}")
        return {
            "success": False,
            "message": f"Error testing connection: {str(e)}",
            "response_time": None,
            "api_version": None,
            "details": {}
        }

# Client-specific status check implementations
async def check_ncbi_status(config: Dict[str, Any]) -> tuple:
    """Check NCBI status."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            params = {
                "tool": "MedicalResearchSynthesizer",
                "email": config.get("email", "your.email@example.com")
            }
            if config.get("api_key"):
                params["api_key"] = config["api_key"]

            response = await client.get(
                f"{config.get('base_url', 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/')}einfo.fcgi",
                params=params
            )

            if response.status_code == 200:
                return True, "2.0"
            else:
                return False, None
    except Exception as e:
        logger.error(f"Error checking NCBI status: {str(e)}")
        return False, None

async def check_umls_status(config: Dict[str, Any]) -> tuple:
    """Check UMLS status."""
    try:
        if not config.get("api_key"):
            return False, None

        async with httpx.AsyncClient(timeout=10.0) as client:
            # First, get a ticket granting ticket
            tgt_response = await client.post(
                "https://utslogin.nlm.nih.gov/cas/v1/api-key",
                data={"apikey": config["api_key"]}
            )

            if tgt_response.status_code != 201:
                return False, None

            tgt = tgt_response.headers.get("location")

            # Then, get a service ticket
            st_response = await client.post(
                tgt,
                data={"service": "http://umlsks.nlm.nih.gov"}
            )

            if st_response.status_code != 200:
                return False, None

            return True, "current"
    except Exception as e:
        logger.error(f"Error checking UMLS status: {str(e)}")
        return False, None

async def check_clinical_trials_status(config: Dict[str, Any]) -> tuple:
    """Check ClinicalTrials.gov status."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{config.get('base_url', 'https://clinicaltrials.gov/api/v2')}/info"
            )

            if response.status_code == 200:
                data = response.json()
                return True, data.get("version", "2.0")
            else:
                return False, None
    except Exception as e:
        logger.error(f"Error checking ClinicalTrials.gov status: {str(e)}")
        return False, None

async def check_cochrane_status(config: Dict[str, Any]) -> tuple:
    """Check Cochrane Library status."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{config.get('base_url', 'https://www.cochranelibrary.com')}/api/v1/search"
            )

            if response.status_code in [200, 400]:  # 400 is expected without search parameters
                return True, "1.0"
            else:
                return False, None
    except Exception as e:
        logger.error(f"Error checking Cochrane Library status: {str(e)}")
        return False, None

async def check_crossref_status(config: Dict[str, Any]) -> tuple:
    """Check Crossref status."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            params = {}
            if config.get("email"):
                params["mailto"] = config["email"]

            response = await client.get(
                f"{config.get('base_url', 'https://api.crossref.org')}/works",
                params=params
            )

            if response.status_code == 200:
                return True, "1.0"
            else:
                return False, None
    except Exception as e:
        logger.error(f"Error checking Crossref status: {str(e)}")
        return False, None

async def check_snomed_status(config: Dict[str, Any]) -> tuple:
    """Check SNOMED CT status."""
    try:
        # For SNOMED CT, we check UMLS status if access_mode is "umls"
        if config.get("access_mode") == "umls":
            return await check_umls_status(config)
        elif config.get("access_mode") == "api" and config.get("api_url"):
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(config["api_url"])

                if response.status_code == 200:
                    return True, "1.0"
                else:
                    return False, None
        else:
            return False, None
    except Exception as e:
        logger.error(f"Error checking SNOMED CT status: {str(e)}")
        return False, None

# Client-specific connection test implementations
async def test_ncbi_connection(config: Dict[str, Any]) -> tuple:
    """Test NCBI connection."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            params = {
                "tool": "MedicalResearchSynthesizer",
                "email": config.get("email", "your.email@example.com"),
                "db": "pubmed",
                "term": "covid-19",
                "retmax": "1"
            }
            if config.get("api_key"):
                params["api_key"] = config["api_key"]

            response = await client.get(
                f"{config.get('base_url', 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/')}esearch.fcgi",
                params=params
            )

            if response.status_code == 200:
                return True, "2.0", {"response": response.text[:200] + "..."}
            else:
                return False, None, {"status_code": response.status_code, "response": response.text}
    except Exception as e:
        logger.error(f"Error testing NCBI connection: {str(e)}")
        return False, None, {"error": str(e)}

async def test_umls_connection(config: Dict[str, Any]) -> tuple:
    """Test UMLS connection."""
    try:
        if not config.get("api_key"):
            return False, None, {"error": "API key not provided"}

        async with httpx.AsyncClient(timeout=10.0) as client:
            # First, get a ticket granting ticket
            tgt_response = await client.post(
                "https://utslogin.nlm.nih.gov/cas/v1/api-key",
                data={"apikey": config["api_key"]}
            )

            if tgt_response.status_code != 201:
                return False, None, {"status_code": tgt_response.status_code, "response": tgt_response.text}

            tgt = tgt_response.headers.get("location")

            # Then, get a service ticket
            st_response = await client.post(
                tgt,
                data={"service": "http://umlsks.nlm.nih.gov"}
            )

            if st_response.status_code != 200:
                return False, None, {"status_code": st_response.status_code, "response": st_response.text}

            service_ticket = st_response.text

            # Test a simple API call
            search_response = await client.get(
                f"{config.get('base_url', 'https://uts-ws.nlm.nih.gov/rest')}/search/current",
                params={"string": "heart attack", "ticket": service_ticket}
            )

            if search_response.status_code == 200:
                return True, "current", {"response": str(search_response.json())[:200] + "..."}
            else:
                return False, None, {"status_code": search_response.status_code, "response": search_response.text}
    except Exception as e:
        logger.error(f"Error testing UMLS connection: {str(e)}")
        return False, None, {"error": str(e)}

async def test_clinical_trials_connection(config: Dict[str, Any]) -> tuple:
    """Test ClinicalTrials.gov connection."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test API info endpoint
            info_response = await client.get(
                f"{config.get('base_url', 'https://clinicaltrials.gov/api/v2')}/info"
            )

            if info_response.status_code != 200:
                return False, None, {"status_code": info_response.status_code, "response": info_response.text}

            info_data = info_response.json()

            # Test a simple search
            search_response = await client.get(
                f"{config.get('base_url', 'https://clinicaltrials.gov/api/v2')}/studies",
                params={"query.term": "covid-19", "pageSize": 1}
            )

            if search_response.status_code == 200:
                return True, info_data.get("version", "2.0"), {
                    "info": info_data,
                    "search_response": str(search_response.json())[:200] + "..."
                }
            else:
                return False, info_data.get("version", "2.0"), {
                    "info": info_data,
                    "search_status_code": search_response.status_code,
                    "search_response": search_response.text
                }
    except Exception as e:
        logger.error(f"Error testing ClinicalTrials.gov connection: {str(e)}")
        return False, None, {"error": str(e)}

async def test_cochrane_connection(config: Dict[str, Any]) -> tuple:
    """Test Cochrane Library connection."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test a simple search
            search_response = await client.get(
                f"{config.get('base_url', 'https://www.cochranelibrary.com')}/api/v1/search/cdsr",
                params={"q": "diabetes"}
            )

            if search_response.status_code == 200:
                return True, "1.0", {"response": str(search_response.json())[:200] + "..."}
            else:
                return False, None, {"status_code": search_response.status_code, "response": search_response.text}
    except Exception as e:
        logger.error(f"Error testing Cochrane Library connection: {str(e)}")
        return False, None, {"error": str(e)}

async def test_crossref_connection(config: Dict[str, Any]) -> tuple:
    """Test Crossref connection."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            params = {"rows": 1}
            if config.get("email"):
                params["mailto"] = config["email"]

            response = await client.get(
                f"{config.get('base_url', 'https://api.crossref.org')}/works",
                params=params
            )

            if response.status_code == 200:
                return True, "1.0", {"response": str(response.json())[:200] + "..."}
            else:
                return False, None, {"status_code": response.status_code, "response": response.text}
    except Exception as e:
        logger.error(f"Error testing Crossref connection: {str(e)}")
        return False, None, {"error": str(e)}

async def test_snomed_connection(config: Dict[str, Any]) -> tuple:
    """Test SNOMED CT connection."""
    try:
        # For SNOMED CT, we test UMLS connection if access_mode is "umls"
        if config.get("access_mode") == "umls":
            return await test_umls_connection(config)
        elif config.get("access_mode") == "api" and config.get("api_url"):
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{config['api_url']}/concepts",
                    params={"term": "myocardial infarction", "limit": 1}
                )

                if response.status_code == 200:
                    return True, "1.0", {"response": str(response.json())[:200] + "..."}
                else:
                    return False, None, {"status_code": response.status_code, "response": response.text}
        else:
            return False, None, {"error": "Invalid access mode or API URL not provided"}
    except Exception as e:
        logger.error(f"Error testing SNOMED CT connection: {str(e)}")
        return False, None, {"error": str(e)}
