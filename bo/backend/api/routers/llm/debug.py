"""
Debug endpoints for the LLM Gateway.
These endpoints provide detailed information for debugging purposes.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Body, Query, Path
from typing import Dict, Any, List, Optional
import logging
import os
import yaml
import json
from datetime import datetime
import asyncio
import sys
import traceback

from ...auth import get_current_user, User
from .utils import load_config, GATEWAY_CONFIG_PATH

# Import LLM Gateway components if available
try:
    from asf.medical.llm_gateway.core.client import LLMGatewayClient
    from asf.medical.llm_gateway.core.models import (
        LLMRequest, LLMConfig, InterventionContext, ContentItem,
        GatewayConfig, ProviderConfig, MCPRole
    )
    from asf.medical.llm_gateway.core.factory import ProviderFactory
    from asf.medical.llm_gateway.providers.openai_client import OpenAIClient
    LLM_GATEWAY_AVAILABLE = True
except ImportError:
    LLM_GATEWAY_AVAILABLE = False

router = APIRouter(prefix="/debug", tags=["llm-gateway-debug"])

logger = logging.getLogger(__name__)

@router.get("/config", response_model=Dict[str, Any])
async def get_debug_config(current_user: User = Depends(get_current_user)):
    """
    Get the raw LLM Gateway configuration.
    
    This endpoint returns the raw configuration loaded from the config file.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )
    
    try:
        # Load config from file
        config = load_config(GATEWAY_CONFIG_PATH)
        
        # Mask sensitive information
        masked_config = mask_sensitive_info(config)
        
        return {
            "config_path": GATEWAY_CONFIG_PATH,
            "config": masked_config,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting debug config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get debug config: {str(e)}"
        )

@router.get("/environment", response_model=Dict[str, Any])
async def get_debug_environment(current_user: User = Depends(get_current_user)):
    """
    Get environment information for debugging.
    
    This endpoint returns information about the environment, including
    Python version, installed packages, and environment variables.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )
    
    try:
        # Get Python version
        python_version = sys.version
        
        # Get installed packages
        import pkg_resources
        installed_packages = [
            {"name": pkg.key, "version": pkg.version}
            for pkg in pkg_resources.working_set
        ]
        
        # Get environment variables (mask sensitive ones)
        env_vars = {}
        for key, value in os.environ.items():
            if any(sensitive in key.lower() for sensitive in ["key", "secret", "password", "token"]):
                env_vars[key] = "***MASKED***"
            else:
                env_vars[key] = value
        
        # Get system info
        import platform
        system_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }
        
        return {
            "python_version": python_version,
            "installed_packages": installed_packages,
            "environment_variables": env_vars,
            "system_info": system_info,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting debug environment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get debug environment: {str(e)}"
        )

@router.post("/test-openai", response_model=Dict[str, Any])
async def test_openai_connection(
    api_key: Optional[str] = Body(None, description="OpenAI API key (optional)"),
    current_user: User = Depends(get_current_user)
):
    """
    Test the OpenAI connection.
    
    This endpoint tests the connection to the OpenAI API and returns detailed results.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )
    
    try:
        # Import the test script
        from asf.medical.llm_gateway.test_openai_connection_detailed import OpenAIConnectionTester
        
        # Run test
        tester = OpenAIConnectionTester(api_key, GATEWAY_CONFIG_PATH)
        results = await tester.run_test()
        
        return results
    except Exception as e:
        logger.error(f"Error testing OpenAI connection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test OpenAI connection: {str(e)}"
        )

@router.get("/diagnostics", response_model=Dict[str, Any])
async def run_diagnostics(current_user: User = Depends(get_current_user)):
    """
    Run diagnostics on the LLM Gateway.
    
    This endpoint runs a series of diagnostic tests on the LLM Gateway
    and returns the results.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )
    
    try:
        # Import the diagnostic tool
        from asf.medical.llm_gateway.diagnostic import LLMGatewayDiagnostic
        
        # Run diagnostics
        diagnostic = LLMGatewayDiagnostic(GATEWAY_CONFIG_PATH)
        results = await diagnostic.run_all_tests()
        
        return results
    except Exception as e:
        logger.error(f"Error running diagnostics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run diagnostics: {str(e)}"
        )

@router.get("/logs", response_model=Dict[str, Any])
async def get_logs(
    lines: int = Query(100, description="Number of log lines to return"),
    current_user: User = Depends(get_current_user)
):
    """
    Get recent log entries.
    
    This endpoint returns recent log entries from the application log file.
    """
    if not LLM_GATEWAY_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM Gateway is not available. Please check your installation."
        )
    
    try:
        # Find log file
        log_paths = [
            os.path.join(os.getcwd(), "logs", "app.log"),
            os.path.join(os.getcwd(), "app.log"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "logs", "app.log"),
        ]
        
        log_file = None
        for path in log_paths:
            if os.path.exists(path):
                log_file = path
                break
        
        if not log_file:
            return {
                "error": "Log file not found",
                "searched_paths": log_paths,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Read log file
        with open(log_file, 'r') as f:
            log_lines = f.readlines()
        
        # Get last N lines
        last_lines = log_lines[-lines:] if lines < len(log_lines) else log_lines
        
        return {
            "log_file": log_file,
            "lines": lines,
            "total_lines": len(log_lines),
            "log_entries": last_lines,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting logs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get logs: {str(e)}"
        )

def mask_sensitive_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mask sensitive information in the configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with sensitive information masked
    """
    # Create a deep copy of the config
    import copy
    masked_config = copy.deepcopy(config)
    
    # Mask sensitive information in provider connection params
    providers = masked_config.get("additional_config", {}).get("providers", {})
    for provider_id, provider_config in providers.items():
        connection_params = provider_config.get("connection_params", {})
        for key in connection_params:
            if any(sensitive in key.lower() for sensitive in ["key", "secret", "password", "token"]):
                if isinstance(connection_params[key], str) and len(connection_params[key]) > 8:
                    # Mask the value but keep first and last few characters
                    value = connection_params[key]
                    connection_params[key] = f"{value[:5]}...{value[-4:]}"
                else:
                    connection_params[key] = "***MASKED***"
    
    return masked_config
